// -*- C++ -*-
//
// Package:    APVShotsFilter
// Class:      APVShotsFilter
//
/**\class APVShotsFilter APVShotsFilter.cc DPGAnalysis/SiStripTools/src/APVShotsFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Mia Tosi,40 3-B32,+41227671551,
//         Created:  Sun Nov 10 11:30:51 CET 2013
// $Id$
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"

#include "DQM/SiStripCommon/interface/APVShotFinder.h"
#include "DQM/SiStripCommon/interface/APVShot.h"

#include "DPGAnalysis/SiStripTools/interface/EventWithHistory.h"
#include "DPGAnalysis/SiStripTools/interface/APVCyclePhaseCollection.h"

//******** includes for the cabling *************
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
//***************************************************

//
// class declaration
//

class APVShotsFilter : public edm::EDFilter {
public:
  explicit APVShotsFilter(const edm::ParameterSet&);
  ~APVShotsFilter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  void updateDetCabling(const SiStripDetCablingRcd& iRcd);
  // ----------member data ---------------------------

  edm::EDGetTokenT<EventWithHistory> heToken_;
  edm::EDGetTokenT<APVCyclePhaseCollection> apvphaseToken_;
  edm::EDGetTokenT<edm::DetSetVector<SiStripDigi> > digisToken_;

  bool _selectAPVshots;

  bool _zs;
  int _nevents;

  // DetCabling
  bool _useCabling;
  edm::ESWatcher<SiStripDetCablingRcd> _detCablingWatcher;
  edm::ESGetToken<SiStripDetCabling, SiStripDetCablingRcd> _detCablingToken;
  const SiStripDetCabling* _detCabling = nullptr;  //!< The cabling object.
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
APVShotsFilter::APVShotsFilter(const edm::ParameterSet& iConfig)
    : _selectAPVshots(iConfig.getUntrackedParameter<bool>("selectAPVshots", true)),
      _zs(iConfig.getUntrackedParameter<bool>("zeroSuppressed", true)),
      _nevents(0),
      _useCabling(iConfig.getUntrackedParameter<bool>("useCabling", true)),
      _detCablingWatcher(_useCabling ? decltype(_detCablingWatcher){this, &APVShotsFilter::updateDetCabling}
                                     : decltype(_detCablingWatcher){}),
      _detCablingToken(_useCabling ? decltype(_detCablingToken){esConsumes()} : decltype(_detCablingToken){}) {
  //now do what ever initialization is needed
  edm::InputTag digicollection = iConfig.getParameter<edm::InputTag>("digiCollection");
  edm::InputTag historyProduct = iConfig.getParameter<edm::InputTag>("historyProduct");
  edm::InputTag apvphasecoll = iConfig.getParameter<edm::InputTag>("apvPhaseCollection");

  heToken_ = consumes<EventWithHistory>(historyProduct);
  apvphaseToken_ = consumes<APVCyclePhaseCollection>(apvphasecoll);
  digisToken_ = consumes<edm::DetSetVector<SiStripDigi> >(digicollection);
}

APVShotsFilter::~APVShotsFilter() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  if (_detCabling)
    _detCabling = nullptr;
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool APVShotsFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  if (_useCabling) {
    //retrieve cabling
    _detCablingWatcher.check(iSetup);
  }
  _nevents++;

  edm::Handle<EventWithHistory> he;
  iEvent.getByToken(heToken_, he);

  edm::Handle<APVCyclePhaseCollection> apvphase;
  iEvent.getByToken(apvphaseToken_, apvphase);

  edm::Handle<edm::DetSetVector<SiStripDigi> > digis;
  iEvent.getByToken(digisToken_, digis);

  // loop on detector with digis
  int nshots = 0;
  std::vector<int> nshotsperFed;

  const int siStripFedIdMin = FEDNumbering::MINSiStripFEDID;
  const int siStripFedIdMax = FEDNumbering::MAXSiStripFEDID;
  const uint16_t lNumFeds = (siStripFedIdMax - siStripFedIdMin) + 1;
  if (_useCabling) {
    nshotsperFed.resize(lNumFeds, 0);
  }

  APVShotFinder apvsf(*digis, _zs);
  const std::vector<APVShot>& shots = apvsf.getShots();

  for (std::vector<APVShot>::const_iterator shot = shots.begin(); shot != shots.end(); ++shot) {
    if (!shot->isGenuine())
      continue;
    ++nshots;

    //get the fedid from the detid
    uint32_t det = shot->detId();
    if (_useCabling) {
      int apvPair = shot->apvNumber() / 2;
      LogDebug("APVShotsFilter") << apvPair;

      const FedChannelConnection& theConn = _detCabling->getConnection(det, apvPair);

      int lChannelId = -1;
      int thelFEDId = -1;
      if (theConn.isConnected()) {
        lChannelId = theConn.fedCh();
        thelFEDId = theConn.fedId();
      } else {
        edm::LogWarning("APVShotsFilter") << "connection of det " << det << " APV pair " << apvPair << " not found";
      }
      LogDebug("APVShotsFilter") << thelFEDId << " " << lChannelId;

      const std::vector<const FedChannelConnection*>& conns = _detCabling->getConnections(det);

      if (!(conns.size()))
        continue;
      uint16_t lFedId = 0;
      for (uint32_t ch = 0; ch < conns.size(); ch++) {
        if (conns[ch] && conns[ch]->isConnected()) {
          LogDebug("APVShotsFilter") << *(conns[ch]);
          LogDebug("APVShotsFilter") << "Ready for FED id " << ch;
          lFedId = conns[ch]->fedId();
          LogDebug("APVShotsFilter") << "obtained FED id " << ch << " " << lFedId;
          //uint16_t lFedCh = conns[ch]->fedCh();

          if (lFedId < sistrip::FED_ID_MIN || lFedId > sistrip::FED_ID_MAX) {
            edm::LogWarning("APVShotsFilter") << lFedId << " for detid " << det << " connection " << ch;
            continue;
          } else
            break;
        }
      }
      if (lFedId < sistrip::FED_ID_MIN || lFedId > sistrip::FED_ID_MAX) {
        edm::LogWarning("APVShotsFilter") << lFedId << "found for detid " << det;
        continue;
      }

      if (lFedId != thelFEDId) {
        edm::LogWarning("APVShotsFilter") << " Mismatch in FED id for det " << det << " APV pair " << apvPair << " : "
                                          << lFedId << " vs " << thelFEDId;
      }

      //       LogDebug("APVShotsFilter") << nshotsperfed.size() << " " << lFedId-sistrip::FED_ID_MIN;
      //       ++nshotsperFed[lFedId-FEDNumbering::MINSiStripFEDID];

      LogDebug("APVShotsFilter") << " ready to be filled with " << thelFEDId << " " << lChannelId;
      LogDebug("APVShotsFilter") << " filled with " << thelFEDId << " " << lChannelId;
    }
  }

  bool foundAPVshots = (nshots > 0);
  bool pass = (_selectAPVshots ? foundAPVshots : !foundAPVshots);
  return pass;
}

// ------------ method called once each job just after ending the event loop  ------------
void APVShotsFilter::endJob() { edm::LogInfo("APVShotsFilter") << _nevents << " analyzed events"; }

void APVShotsFilter::updateDetCabling(const SiStripDetCablingRcd& iRcd) { _detCabling = &iRcd.get(_detCablingToken); }

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void APVShotsFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(APVShotsFilter);

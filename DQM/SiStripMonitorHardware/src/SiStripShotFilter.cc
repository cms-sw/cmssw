
// -*- C++ -*-
//
// Class:      SiStripShotFilterPlugins
//
/* Description: DQM source application to filter "shots" for SiStrip data
*/
//
//         Created:  2009/12/07
//

#include <sstream>
#include <fstream>
#include <iostream>
#include <memory>
#include <list>
#include <algorithm>
#include <cassert>

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"

#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"

#include "DQM/SiStripCommon/interface/APVShotFinder.h"
#include "DQM/SiStripCommon/interface/APVShot.h"

//
// Class declaration
//

class SiStripShotFilter : public edm::EDFilter {
public:
  explicit SiStripShotFilter(const edm::ParameterSet&);
  ~SiStripShotFilter() override;

private:
  void beginJob() override;
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  //update the cabling if necessary
  void updateCabling(const SiStripFedCablingRcd& cablingRcd);

  //path to output file
  std::ofstream fOut_;
  std::string fOutPath_;
  //FED cabling
  const SiStripFedCabling* cabling_;
  edm::ESWatcher<SiStripFedCablingRcd> fedCablingWatcher_;
  edm::ESGetToken<SiStripFedCabling, SiStripFedCablingRcd> fedCablingToken_;

  edm::InputTag digicollection_;
  edm::EDGetTokenT<edm::DetSetVector<SiStripDigi> > digiToken_;
  bool zs_;
};

//
// Constructors and destructor
//

SiStripShotFilter::SiStripShotFilter(const edm::ParameterSet& iConfig)
    : fOutPath_(iConfig.getUntrackedParameter<std::string>("OutputFilePath", "shotChannels.dat")),
      fedCablingWatcher_(this, &SiStripShotFilter::updateCabling),
      fedCablingToken_(esConsumes<>()),
      digicollection_(iConfig.getParameter<edm::InputTag>("DigiCollection")),
      zs_(iConfig.getUntrackedParameter<bool>("ZeroSuppressed", true))

{
  digiToken_ = consumes<edm::DetSetVector<SiStripDigi> >(digicollection_);
}

SiStripShotFilter::~SiStripShotFilter() {}

//
// Member functions
//

// ------------ method called to for each event  ------------
bool SiStripShotFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  fedCablingWatcher_.check(iSetup);
  //get digi data
  edm::Handle<edm::DetSetVector<SiStripDigi> > digis;
  iEvent.getByToken(digiToken_, digis);

  // loop on detector with digis

  APVShotFinder apvsf(*digis, zs_);
  const std::vector<APVShot>& shots = apvsf.getShots();

  //loop on feds first: there should be only a small number of shots...
  //better to loop only once on all channels....
  //to be able to output both fed/ch and module/APV.

  unsigned int lShots = 0;

  for (unsigned int fedId = FEDNumbering::MINSiStripFEDID; fedId <= FEDNumbering::MAXSiStripFEDID;
       fedId++) {  //loop over FED IDs

    for (unsigned int iCh = 0; iCh < sistrip::FEDCH_PER_FED; iCh++) {  //loop on channels

      const FedChannelConnection& lConnection = cabling_->fedConnection(fedId, iCh);

      uint32_t lDetId = lConnection.detId();
      short lAPVPair = lConnection.apvPairNumber();

      for (std::vector<APVShot>::const_iterator shot = shots.begin(); shot != shots.end(); ++shot) {  //loop on shots

        if (shot->detId() == lDetId && static_cast<short>(shot->apvNumber() / 2.) == lAPVPair) {
          if (shot->isGenuine()) {  //genuine shot

            fOut_ << fedId << " " << iCh << " " << shot->detId() << " " << shot->apvNumber() << std::endl;
            lShots++;
          }  //genuine shot
          if (shot->apvNumber() % 2 == 1)
            break;
        }
      }  //loop on shots
    }    //loop on channels
  }      //loop on FEDs.

  if (lShots > 0)
    fOut_ << "### " << iEvent.id().event() << " " << lShots << std::endl;

  return lShots;

}  //analyze method

// ------------ method called once each job just before starting event loop  ------------
void SiStripShotFilter::beginJob() {
  fOut_.open(fOutPath_.c_str(), std::ios::out);
  if (!fOut_)
    std::cout << " WARNING ! Cannot open file " << fOutPath_
              << " for writting. List of shot channels will not be saved." << std::endl;
}

// ------------ method called once each job just after ending the event loop  ------------
void SiStripShotFilter::endJob() { fOut_.close(); }

void SiStripShotFilter::updateCabling(const SiStripFedCablingRcd& cablingRcd) {
  cabling_ = &cablingRcd.get(fedCablingToken_);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripShotFilter);

// -*- C++ -*-
//
// Package:    SiStripTools
// Class:      APVCyclePhaseProducerFromL1ABC
//
/**\class APVCyclePhaseProducerFromL1ABC APVCyclePhaseProducerFromL1ABC.cc DPGAnalysis/SiStripTools/plugins/APVCyclePhaseProducerFromL1ABC.cc

 Description: EDproducer for APVCyclePhaseCollection which uses the configuration file to assign a phase to the run

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Venturi
//         Created:  Mon Jan 12 09:05:45 CET 2009
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <map>
#include <vector>
#include <string>

#include "TH1F.h"
#include "TProfile.h"

#include "DataFormats/Scalers/interface/L1AcceptBunchCrossing.h"
#include "DataFormats/TCDS/interface/TCDSRecord.h"
#include "DPGAnalysis/SiStripTools/interface/APVCyclePhaseCollection.h"

#include "DPGAnalysis/SiStripTools/interface/RunHistogramManager.h"

//
// class decleration
//

class APVCyclePhaseProducerFromL1ABC : public edm::EDProducer {
public:
  explicit APVCyclePhaseProducerFromL1ABC(const edm::ParameterSet&);
  ~APVCyclePhaseProducerFromL1ABC() override;

private:
  void beginJob() override;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<L1AcceptBunchCrossingCollection> _l1abccollectionToken;
  edm::EDGetTokenT<TCDSRecord> _tcdsRecordToken;
  const std::vector<std::string> _defpartnames;
  const std::vector<int> _defphases;
  const int _orbitoffsetSOR;
  const bool _wantHistos;
  const bool _forceSCAL;
  RunHistogramManager m_rhm;

  TH1F** _hbx;
  TH1F** _hdbx;
  TH1F** _hdorbit;
  const unsigned int _firstgoodrun;
  std::map<edm::EventNumber_t, long long> _offsets;
  long long _curroffset;
  edm::EventNumber_t _curroffevent;
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
APVCyclePhaseProducerFromL1ABC::APVCyclePhaseProducerFromL1ABC(const edm::ParameterSet& iConfig)
    : _l1abccollectionToken(
          mayConsume<L1AcceptBunchCrossingCollection>(iConfig.getParameter<edm::InputTag>("l1ABCCollection"))),
      _tcdsRecordToken(mayConsume<TCDSRecord>(iConfig.getParameter<edm::InputTag>("tcdsRecordLabel"))),
      _defpartnames(iConfig.getParameter<std::vector<std::string> >("defaultPartitionNames")),
      _defphases(iConfig.getParameter<std::vector<int> >("defaultPhases")),
      _orbitoffsetSOR(iConfig.getParameter<int>("StartOfRunOrbitOffset")),
      _wantHistos(iConfig.getUntrackedParameter<bool>("wantHistos", false)),
      _forceSCAL(iConfig.getParameter<bool>("forceSCAL")),
      m_rhm(consumesCollector()),
      _hbx(nullptr),
      _hdbx(nullptr),
      _hdorbit(nullptr),
      _firstgoodrun(110878),
      _offsets(),
      _curroffset(0),
      _curroffevent(0) {
  produces<APVCyclePhaseCollection, edm::InEvent>();

  //now do what ever other initialization is needed

  if (_wantHistos) {
    _hbx = m_rhm.makeTH1F("l1abcbx", "BX number from TCDS (or L1ABC as fallback)", 4096, -0.5, 4095.5);
    _hdbx = m_rhm.makeTH1F("dbx", "BX number difference", 4096 * 2 - 1, -4095.5, 4095.5);
    _hdorbit = m_rhm.makeTH1F("dorbit", "Orbit Number difference", 9999, -4999.5, 4999.5);
  }
}

APVCyclePhaseProducerFromL1ABC::~APVCyclePhaseProducerFromL1ABC() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void APVCyclePhaseProducerFromL1ABC::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)

{
  // reset offset vector

  _offsets.clear();
  edm::LogInfo("AbsoluteBXOffsetReset") << "Absolute BX offset map reset";

  if (_wantHistos) {
    m_rhm.beginRun(iRun);

    if (_hbx && *_hbx) {
      (*_hbx)->GetXaxis()->SetTitle("BX");
      (*_hbx)->GetYaxis()->SetTitle("Events");
    }

    if (_hdbx && *_hdbx) {
      (*_hdbx)->GetXaxis()->SetTitle("#DeltaBX");
      (*_hdbx)->GetYaxis()->SetTitle("Events");
    }

    if (_hdorbit && *_hdorbit) {
      (*_hdorbit)->GetXaxis()->SetTitle("#Deltaorbit");
      (*_hdorbit)->GetYaxis()->SetTitle("Events");
    }
  }

  if (iRun.run() < _firstgoodrun) {
    edm::LogInfo("UnreliableMissingL1AcceptBunchCrossingCollection")
        << "In this run L1AcceptBunchCrossingCollection is missing or unreliable: default phases will be used";
  }
}

void APVCyclePhaseProducerFromL1ABC::endRun(const edm::Run&, const edm::EventSetup&) {
  // summary of absolute bx offset vector

  edm::LogInfo("L1AcceptBunchCrossingAbsoluteBXOffsetSummary") << "Absolute BX offset summary:";
  for (std::map<edm::EventNumber_t, long long>::const_iterator offset = _offsets.begin(); offset != _offsets.end();
       ++offset) {
    edm::LogVerbatim("L1AcceptBunchCrossingAbsoluteBXOffsetSummary") << offset->first << " " << offset->second;
  }
}

void APVCyclePhaseProducerFromL1ABC::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  std::unique_ptr<APVCyclePhaseCollection> apvphases(new APVCyclePhaseCollection());

  const std::vector<int>& phases = _defphases;
  const std::vector<std::string>& partnames = _defpartnames;

  // Look for the L1AcceptBunchCrossingCollection

  int phasechange = 0;

  if (iEvent.run() >= _firstgoodrun) {
    Handle<L1AcceptBunchCrossingCollection> pIn;
    iEvent.getByToken(_l1abccollectionToken, pIn);
    Handle<TCDSRecord> tcds_pIn;
    iEvent.getByToken(_tcdsRecordToken, tcds_pIn);
    bool useTCDS(tcds_pIn.isValid() && !_forceSCAL);
    const auto* tcdsRecord = useTCDS ? tcds_pIn.product() : nullptr;
    // offset computation

    long long orbitoffset = _orbitoffsetSOR;
    int bxoffset = 0;

    if (useTCDS) {
      if (tcdsRecord->getL1aHistoryEntry(0).getIndex() == 0) {
        if (tcdsRecord->getEventType() != 0) {
          orbitoffset = (long long)tcdsRecord->getOrbitNr() - (long long)iEvent.orbitNumber();
          // In EventWithHistoryProducerFromL1ABC, it's -1\times this, following the corresponding L1ABC line below. I keep this difference here, because I am not sure if it should be kept or not
          bxoffset = iEvent.bunchCrossing() - tcdsRecord->getBXID();
          // If I understand correctly, tcdsRecord has no l1AcceptOffset thing, so the control done for l1abc can be safely skipped
          if (_wantHistos) {
            if (_hbx && *_hbx)
              (*_hbx)->Fill(tcdsRecord->getBXID());
            if (_hdbx && *_hdbx)
              (*_hdbx)->Fill(bxoffset);
            if (_hdorbit && *_hdorbit)
              (*_hdorbit)->Fill(orbitoffset);
          }
        }
      } else {
        edm::LogWarning("L1AcceptBunchCrossingNoType") << "L1AcceptBunchCrossing with no type found: ";
        edm::LogPrint("L1AcceptBunchCrossingNoType") << *tcdsRecord;
      }
    } else {
      for (L1AcceptBunchCrossingCollection::const_iterator l1abc = pIn->begin(); l1abc != pIn->end(); ++l1abc) {
        if (l1abc->l1AcceptOffset() == 0) {
          if (l1abc->eventType() != 0) {
            orbitoffset = (long long)iEvent.orbitNumber() - (long long)l1abc->orbitNumber();
            bxoffset =
                iEvent.bunchCrossing() -
                l1abc->bunchCrossing();  // In EventWithHistoryProducerFromL1ABC, it's l1abc - iEvent. I keep this difference here, because I am not sure if it should be kept or not

            if (_wantHistos) {
              if (_hbx && *_hbx)
                (*_hbx)->Fill(l1abc->bunchCrossing());
              if (_hdbx && *_hdbx)
                (*_hdbx)->Fill(bxoffset);
              if (_hdorbit && *_hdorbit)
                (*_hdorbit)->Fill(orbitoffset);
            }
          } else {
            edm::LogWarning("L1AcceptBunchCrossingNoType") << "L1AcceptBunchCrossing with no type found: ";
            for (L1AcceptBunchCrossingCollection::const_iterator debu = pIn->begin(); debu != pIn->end(); ++debu) {
              edm::LogPrint("L1AcceptBunchCrossingNoType") << *debu;
            }
          }
        }
      }
    }

    long long absbxoffset = orbitoffset * 3564 + bxoffset;

    if (orbitoffset != _orbitoffsetSOR)
      phasechange = (orbitoffset * 3564) % 70;

    if (_offsets.empty()) {
      _curroffset = absbxoffset;
      _curroffevent = iEvent.id().event();
      _offsets[iEvent.id().event()] = absbxoffset;
    } else {
      if (_curroffset != absbxoffset || iEvent.id().event() < _curroffevent) {
        if (_curroffset != absbxoffset) {
          edm::LogInfo("L1AcceptBunchCrossingAbsoluteBXOffsetChanged")
              << "Absolute BX offset changed from " << _curroffset << " to " << absbxoffset << " at orbit "
              << iEvent.orbitNumber() << " and BX " << iEvent.bunchCrossing();
          if (useTCDS) {
            edm::LogVerbatim("AbsoluteBXOffsetChanged") << *tcdsRecord;  // Not sure about this
          } else {
            for (L1AcceptBunchCrossingCollection::const_iterator l1abc = pIn->begin(); l1abc != pIn->end(); ++l1abc) {
              edm::LogVerbatim("L1AcceptBunchCrossingAbsoluteBXOffsetChanged") << *l1abc;
            }
          }
        }

        _curroffset = absbxoffset;
        _curroffevent = iEvent.id().event();
        _offsets[iEvent.id().event()] = absbxoffset;
      }
    }
  }

  if (phases.size() < partnames.size()) {
    // throw exception
    throw cms::Exception("InvalidAPVCyclePhases")
        << " Inconsistent phases/partitions vector sizes: " << phases.size() << " " << partnames.size();
  }

  for (unsigned int ipart = 0; ipart < partnames.size(); ++ipart) {
    if (phases[ipart] >= 0) {
      apvphases->get()[partnames[ipart]] = (phases[ipart] + phasechange) % 70;
    }
  }

  iEvent.put(std::move(apvphases));
}

// ------------ method called once each job just before starting event loop  ------------
void APVCyclePhaseProducerFromL1ABC::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void APVCyclePhaseProducerFromL1ABC::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(APVCyclePhaseProducerFromL1ABC);

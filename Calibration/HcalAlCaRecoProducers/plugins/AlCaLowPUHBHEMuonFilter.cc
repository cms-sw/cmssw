// system include files
#include <atomic>
#include <memory>
#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

//#define EDM_ML_DEBUG
//
// class declaration
//

namespace alCaLowPUHBHEMuonFilter {
  struct Counters {
    Counters() : nAll_(0), nGood_(0) {}
    mutable std::atomic<unsigned int> nAll_, nGood_;
  };
}  // namespace alCaLowPUHBHEMuonFilter

class AlCaLowPUHBHEMuonFilter : public edm::stream::EDFilter<edm::GlobalCache<alCaLowPUHBHEMuonFilter::Counters> > {
public:
  explicit AlCaLowPUHBHEMuonFilter(edm::ParameterSet const&, const alCaLowPUHBHEMuonFilter::Counters* count);
  ~AlCaLowPUHBHEMuonFilter() override;

  static std::unique_ptr<alCaLowPUHBHEMuonFilter::Counters> initializeGlobalCache(edm::ParameterSet const&) {
    return std::make_unique<alCaLowPUHBHEMuonFilter::Counters>();
  }

  bool filter(edm::Event&, edm::EventSetup const&) override;
  void endStream() override;
  static void globalEndJob(const alCaLowPUHBHEMuonFilter::Counters* counters);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
  HLTConfigProvider hltConfig_;
  std::vector<std::string> trigNames_, HLTNames_;
  std::string processName_;
  bool pfCut_;
  double trackIsoCut_, caloIsoCut_, pfIsoCut_, minimumMuonpT_, minimumMuoneta_;
  int preScale_;
  unsigned int nRun_, nAll_, nGood_;
  edm::InputTag triggerResults_, labelMuon_;
  edm::EDGetTokenT<trigger::TriggerEvent> tok_trigEvt;
  edm::EDGetTokenT<edm::TriggerResults> tok_trigRes_;
  edm::EDGetTokenT<reco::MuonCollection> tok_Muon_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tok_magField_;
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
AlCaLowPUHBHEMuonFilter::AlCaLowPUHBHEMuonFilter(edm::ParameterSet const& iConfig,
                                                 const alCaLowPUHBHEMuonFilter::Counters* count)
    : nRun_(0), nAll_(0), nGood_(0) {
  //now do what ever initialization is needed
  trigNames_ = iConfig.getParameter<std::vector<std::string> >("triggers");
  processName_ = iConfig.getParameter<std::string>("processName");
  triggerResults_ = iConfig.getParameter<edm::InputTag>("triggerResultLabel");
  labelMuon_ = iConfig.getParameter<edm::InputTag>("muonLabel");
  pfIsoCut_ = iConfig.getParameter<double>("pfIsolationCut");
  minimumMuonpT_ = iConfig.getParameter<double>("minimumMuonpT");
  minimumMuoneta_ = iConfig.getParameter<double>("minimumMuoneta");

  // define tokens for access
  tok_trigRes_ = consumes<edm::TriggerResults>(triggerResults_);
  tok_Muon_ = consumes<reco::MuonCollection>(labelMuon_);

  tok_geom_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
  tok_magField_ = esConsumes<MagneticField, IdealMagneticFieldRecord>();

  edm::LogVerbatim("LowPUHBHEMuon") << "Parameters read from config file \n"
                                    << "Process " << processName_ << "  PF Isolation Cuts " << pfIsoCut_
                                    << " minimum Muon pT cut " << minimumMuonpT_ << " minimum Muon eta cut "
                                    << minimumMuoneta_;
  for (unsigned int k = 0; k < trigNames_.size(); ++k)
    edm::LogVerbatim("LowPUHBHEMuon") << "Trigger[" << k << "] " << trigNames_[k];
}  // AlCaLowPUHBHEMuonFilter::AlCaLowPUHBHEMuonFilter  constructor

AlCaLowPUHBHEMuonFilter::~AlCaLowPUHBHEMuonFilter() {}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool AlCaLowPUHBHEMuonFilter::filter(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  bool accept(false);
  ++nAll_;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("LowPUHBHEMuon") << "AlCaLowPUHBHEMuonFilter::Run " << iEvent.id().run() << " Event "
                                    << iEvent.id().event() << " Luminosity " << iEvent.luminosityBlock() << " Bunch "
                                    << iEvent.bunchCrossing();
#endif
  //Step1: Find if the event passes one of the chosen triggers
  /////////////////////////////TriggerResults
  auto const& triggerResults = iEvent.getHandle(tok_trigRes_);
  if (triggerResults.isValid()) {
    bool ok(false);
    std::vector<std::string> modules;
    const edm::TriggerNames& triggerNames = iEvent.triggerNames(*triggerResults);
    const std::vector<std::string>& triggerNames_ = triggerNames.triggerNames();
    for (unsigned int iHLT = 0; iHLT < triggerResults->size(); iHLT++) {
      int hlt = triggerResults->accept(iHLT);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("LowPUHBHEMuonX") << "trigger names: " << iHLT << " " << triggerNames_[iHLT];
#endif
      for (auto const& trigName : trigNames_) {
        if (triggerNames_[iHLT].find(trigName) != std::string::npos) {
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("LowPUHBHEMuonX") << "find trigger names: " << trigName;
#endif
          if (hlt > 0)
            ok = true;
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("LowPUHBHEMuon")
              << "AlCaLowPUHBHEMuonFilter::Trigger " << triggerNames_[iHLT] << " Flag " << hlt << ":" << ok;
#endif
        }
      }
    }
    if (ok) {
      //Step2: Get geometry/B-field information
      const MagneticField* bField = &(iSetup.getData(tok_magField_));
      const CaloGeometry* geo = &(iSetup.getData(tok_geom_));

      // Relevant blocks from iEvent
      auto muonHandle = iEvent.getHandle(tok_Muon_);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("LowPUHBHEMuon") << "AlCaLowPUHBHEMuonFilter::Muon Handle " << muonHandle.isValid();
#endif
      if (muonHandle.isValid()) {
        for (reco::MuonCollection::const_iterator RecMuon = muonHandle->begin(); RecMuon != muonHandle->end();
             ++RecMuon) {
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("LowPUHBHEMuon")
              << "AlCaLowPUHBHEMuonFilter::Muon:Track " << RecMuon->track().isNonnull() << " innerTrack "
              << RecMuon->innerTrack().isNonnull() << " outerTrack " << RecMuon->outerTrack().isNonnull()
              << " globalTrack " << RecMuon->globalTrack().isNonnull();
#endif
          if ((RecMuon->pt() < minimumMuonpT_) || std::abs(RecMuon->eta()) < minimumMuoneta_)
            continue;
          if ((RecMuon->track().isNonnull()) && (RecMuon->innerTrack().isNonnull()) &&
              (RecMuon->outerTrack().isNonnull()) && (RecMuon->globalTrack().isNonnull())) {
            const reco::Track* pTrack = (RecMuon->innerTrack()).get();
            spr::propagatedTrackID trackID = spr::propagateCALO(pTrack, geo, bField, false);
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("LowPUHBHEMuon")
                << "AlCaLowPUHBHEMuonFilter::Propagate: ECAL " << trackID.okECAL << " to HCAL " << trackID.okHCAL;
#endif
            double isolR04 =
                ((RecMuon->pfIsolationR04().sumChargedHadronPt +
                  std::max(0.,
                           RecMuon->pfIsolationR04().sumNeutralHadronEt + RecMuon->pfIsolationR04().sumPhotonEt -
                               (0.5 * RecMuon->pfIsolationR04().sumPUPt))) /
                 RecMuon->pt());
            bool isoCut = (isolR04 < pfIsoCut_);
            if ((trackID.okECAL) && (trackID.okHCAL) && isoCut) {
              accept = true;
              break;
            }
          }
        }
      }
    }
  }
  // Step 4:  Return the acceptance flag
  if (accept) {
    ++nGood_;
  }
  return accept;

}  // AlCaLowPUHBHEMuonFilter::filter

// ------------ method called once each job just after ending the event loop  ------------
void AlCaLowPUHBHEMuonFilter::endStream() {
  globalCache()->nAll_ += nAll_;
  globalCache()->nGood_ += nGood_;
}

void AlCaLowPUHBHEMuonFilter::globalEndJob(const alCaLowPUHBHEMuonFilter::Counters* count) {
  edm::LogVerbatim("LowPUHBHEMuon") << "Selects " << count->nGood_ << " good events out of " << count->nAll_
                                    << " total # of events";
}

// ------------ method called when starting to processes a run  ------------
void AlCaLowPUHBHEMuonFilter::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  bool changed(false);
  bool flag = hltConfig_.init(iRun, iSetup, processName_, changed);
  edm::LogVerbatim("LowPUHBHEMuon") << "Run[" << nRun_ << "] " << iRun.run() << " hltconfig.init " << flag;
}

// ------------ method called when ending the processing of a run  ------------
void AlCaLowPUHBHEMuonFilter::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  edm::LogVerbatim("LowPUHBHEMuon") << "endRun[" << nRun_ << "] " << iRun.run();
  nRun_++;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void AlCaLowPUHBHEMuonFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<std::string> triggers = {"HLT_L1DoubleMu", "HLT_L1SingleMu"};
  desc.add<std::string>("processName", "HLT");
  desc.add<edm::InputTag>("triggerResultLabel", edm::InputTag("TriggerResults", "", "HLT"));
  desc.add<edm::InputTag>("muonLabel", edm::InputTag("muons"));
  desc.add<double>("minimumMuonpT", 10.0);
  desc.add<double>("minimumMuoneta", 1.305);
  desc.add<std::vector<std::string> >("triggers", triggers);
  desc.add<double>("pfIsolationCut", 0.15);
  descriptions.add("alcaLowPUHBHEMuonFilter", desc);
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(AlCaLowPUHBHEMuonFilter);

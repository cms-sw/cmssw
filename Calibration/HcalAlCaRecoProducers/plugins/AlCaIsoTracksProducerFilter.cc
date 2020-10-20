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
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/Common/interface/Handle.h"

//Triggers
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

//
// class declaration
//

namespace AlCaIsoTracksProdFilter {
  struct Counters {
    Counters() : nAll_(0), nGood_(0) {}
    mutable std::atomic<unsigned int> nAll_, nGood_;
  };
}  // namespace AlCaIsoTracksProdFilter

class AlCaIsoTracksProducerFilter : public edm::stream::EDFilter<edm::GlobalCache<AlCaIsoTracksProdFilter::Counters> > {
public:
  explicit AlCaIsoTracksProducerFilter(edm::ParameterSet const&, const AlCaIsoTracksProdFilter::Counters* count);
  ~AlCaIsoTracksProducerFilter() override;

  static std::unique_ptr<AlCaIsoTracksProdFilter::Counters> initializeGlobalCache(edm::ParameterSet const& iConfig) {
    return std::make_unique<AlCaIsoTracksProdFilter::Counters>();
  }

  bool filter(edm::Event&, edm::EventSetup const&) override;
  void endStream() override;
  static void globalEndJob(const AlCaIsoTracksProdFilter::Counters* counters);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
  HLTConfigProvider hltConfig_;
  std::vector<std::string> trigNames_;
  unsigned int nRun_, nAll_, nGood_;
  edm::EDGetTokenT<edm::TriggerResults> tok_trigRes_;
  std::string processName_;
  edm::InputTag triggerResultsLabel_;
};

AlCaIsoTracksProducerFilter::AlCaIsoTracksProducerFilter(const edm::ParameterSet& iConfig,
                                                         const AlCaIsoTracksProdFilter::Counters* count)
    : nRun_(0), nAll_(0), nGood_(0) {
  //now do what ever initialization is needed
  trigNames_ = iConfig.getParameter<std::vector<std::string> >("triggers");
  processName_ = iConfig.getParameter<std::string>("processName");
  triggerResultsLabel_ = iConfig.getParameter<edm::InputTag>("triggerResultLabel");

  // define tokens for access
  tok_trigRes_ = consumes<edm::TriggerResults>(triggerResultsLabel_);

  edm::LogVerbatim("HcalIsoTrack") << "Use process name " << processName_ << " Labels " << triggerResultsLabel_
                                   << " selecting " << trigNames_.size() << " triggers\n";
  for (unsigned int k = 0; k < trigNames_.size(); ++k) {
    edm::LogVerbatim("HcalIsoTrack") << "Trigger[" << k << "] " << trigNames_[k] << std::endl;
  }
}

AlCaIsoTracksProducerFilter::~AlCaIsoTracksProducerFilter() {}

bool AlCaIsoTracksProducerFilter::filter(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  ++nAll_;
  edm::LogVerbatim("HcalIsoTrack") << "Run " << iEvent.id().run() << " Event " << iEvent.id().event() << " Luminosity "
                                   << iEvent.luminosityBlock() << " Bunch " << iEvent.bunchCrossing() << std::endl;

  //Find if the event passes one of the chosen triggers
  bool triggerSatisfied(false);
  if (trigNames_.empty()) {
    triggerSatisfied = true;
  } else {
    edm::Handle<edm::TriggerResults> triggerResults;
    iEvent.getByToken(tok_trigRes_, triggerResults);
    if (triggerResults.isValid()) {
      std::vector<std::string> modules;
      const edm::TriggerNames& triggerNames = iEvent.triggerNames(*triggerResults);
      const std::vector<std::string>& triggerNames_ = triggerNames.triggerNames();
      for (unsigned int iHLT = 0; iHLT < triggerResults->size(); iHLT++) {
        int hlt = triggerResults->accept(iHLT);
        for (unsigned int i = 0; i < trigNames_.size(); ++i) {
          if (triggerNames_[iHLT].find(trigNames_[i]) != std::string::npos) {
            edm::LogVerbatim("HcalIsoTrack")
                << triggerNames_[iHLT] << " has got HLT flag " << hlt << ":" << triggerSatisfied << std::endl;
            if (hlt > 0) {
              triggerSatisfied = true;
              break;
            }
          }
        }
        if (triggerSatisfied)
          break;
      }
    }
  }
  if (triggerSatisfied)
    ++nGood_;
  return triggerSatisfied;
}

void AlCaIsoTracksProducerFilter::endStream() {
  globalCache()->nAll_ += nAll_;
  globalCache()->nGood_ += nGood_;
}

void AlCaIsoTracksProducerFilter::globalEndJob(const AlCaIsoTracksProdFilter::Counters* count) {
  edm::LogVerbatim("HcalIsoTrack") << "Selects " << count->nGood_ << " in " << count->nAll_ << " events " << std::endl;
}

void AlCaIsoTracksProducerFilter::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  bool changed(false);
  bool flag = hltConfig_.init(iRun, iSetup, processName_, changed);
  edm::LogVerbatim("HcalIsoTrack") << "Run[" << nRun_ << "] " << iRun.run() << " hltconfig.init " << flag << std::endl;
}

void AlCaIsoTracksProducerFilter::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  ++nRun_;
  edm::LogVerbatim("HcalIsoTrack") << "endRun[" << nRun_ << "] " << iRun.run() << std::endl;
}

void AlCaIsoTracksProducerFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("triggerResultLabel", edm::InputTag("TriggerResults", "", "HLT"));
  std::vector<std::string> trigger = {"HLT_IsoTrackHB", "HLT_IsoTrackHE"};
  desc.add<std::vector<std::string> >("triggers", trigger);
  desc.add<std::string>("processName", "HLT");
  descriptions.add("alcaIsoTracksProducerFilter", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(AlCaIsoTracksProducerFilter);

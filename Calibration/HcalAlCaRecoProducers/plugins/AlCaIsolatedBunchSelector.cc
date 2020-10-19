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

//#define EDM_ML_DEBUG

namespace AlCaIsolatedBunch {
  struct Counters {
    Counters() : nAll_(0), nGood_(0) {}
    mutable std::atomic<unsigned int> nAll_, nGood_;
  };
}  // namespace AlCaIsolatedBunch

class AlCaIsolatedBunchSelector : public edm::stream::EDFilter<edm::GlobalCache<AlCaIsolatedBunch::Counters> > {
public:
  explicit AlCaIsolatedBunchSelector(edm::ParameterSet const&, const AlCaIsolatedBunch::Counters* count);
  ~AlCaIsolatedBunchSelector() override;

  static std::unique_ptr<AlCaIsolatedBunch::Counters> initializeGlobalCache(edm::ParameterSet const& iConfig) {
    return std::make_unique<AlCaIsolatedBunch::Counters>();
  }

  bool filter(edm::Event&, edm::EventSetup const&) override;
  void endStream() override;
  static void globalEndJob(const AlCaIsolatedBunch::Counters* counters);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
  HLTConfigProvider hltConfig_;
  std::string trigName_;
  edm::InputTag theTriggerResultsLabel_;
  std::string processName_;
  unsigned int nRun_, nAll_, nGood_;
  edm::EDGetTokenT<edm::TriggerResults> tok_trigRes_;
};

//
// constructors and destructor
//
AlCaIsolatedBunchSelector::AlCaIsolatedBunchSelector(const edm::ParameterSet& iConfig,
                                                     const AlCaIsolatedBunch::Counters* count)
    : nRun_(0), nAll_(0), nGood_(0) {
  //now do what ever initialization is needed
  trigName_ = iConfig.getParameter<std::string>("triggerName");
  processName_ = iConfig.getParameter<std::string>("processName");
  theTriggerResultsLabel_ = iConfig.getParameter<edm::InputTag>("triggerResultLabel");

  // define tokens for access
  tok_trigRes_ = consumes<edm::TriggerResults>(theTriggerResultsLabel_);

  edm::LogVerbatim("AlCaIsoBunch") << "Input tag for trigger results " << theTriggerResultsLabel_
                                   << " with trigger name " << trigName_ << " and process " << processName_
                                   << std::endl;
}

AlCaIsolatedBunchSelector::~AlCaIsolatedBunchSelector() {}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool AlCaIsolatedBunchSelector::filter(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  bool accept(false);
  ++nAll_;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("AlCaIsoBunch") << "Run " << iEvent.id().run() << " Event " << iEvent.id().event() << " Luminosity "
                                   << iEvent.luminosityBlock() << " Bunch " << iEvent.bunchCrossing() << std::endl;
#endif
  //Step1: Find if the event passes the chosen trigger
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(tok_trigRes_, triggerResults);
  if (triggerResults.isValid()) {
    const edm::TriggerNames& triggerNames = iEvent.triggerNames(*triggerResults);
    const std::vector<std::string>& triggerNames_ = triggerNames.triggerNames();
    for (unsigned int iHLT = 0; iHLT < triggerResults->size(); iHLT++) {
      int hlt = triggerResults->accept(iHLT);
      if (triggerNames_[iHLT].find(trigName_) != std::string::npos) {
        if (hlt > 0) {
          accept = true;
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("AlCaIsoBunch")
              << triggerNames_[iHLT] << " has got HLT flag " << hlt << ":" << accept << std::endl;
#endif
          break;
        }
      }
    }
  }

  // Step 2:  Return the acceptance flag
  if (accept)
    ++nGood_;
  return accept;

}  // AlCaIsolatedBunchSelector::filter
// ------------ method called once each job just after ending the event loop  ------------
void AlCaIsolatedBunchSelector::endStream() {
  globalCache()->nAll_ += nAll_;
  globalCache()->nGood_ += nGood_;
}

void AlCaIsolatedBunchSelector::globalEndJob(const AlCaIsolatedBunch::Counters* count) {
  edm::LogVerbatim("AlCaIsoBunch") << "Selects " << count->nGood_ << " in " << count->nAll_ << " events" << std::endl;
}

// ------------ method called when starting to processes a run  ------------
void AlCaIsolatedBunchSelector::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  bool changed(false);
  edm::LogVerbatim("AlCaIsoBunch") << "Run[" << nRun_ << "] " << iRun.run() << " hltconfig.init "
                                   << hltConfig_.init(iRun, iSetup, processName_, changed) << std::endl;
}
// ------------ method called when ending the processing of a run  ------------
void AlCaIsolatedBunchSelector::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  ++nRun_;
  edm::LogVerbatim("AlCaIsoBunch") << "endRun[" << nRun_ << "] " << iRun.run() << std::endl;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void AlCaIsolatedBunchSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("triggerResultLabel", edm::InputTag("TriggerResults", "", "HLT"));
  desc.add<std::string>("processName", "HLT");
  desc.add<std::string>("triggerName", "HLT_HcalIsolatedBunch");
  descriptions.add("alcaIsolatedBunchSelector", desc);
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(AlCaIsolatedBunchSelector);

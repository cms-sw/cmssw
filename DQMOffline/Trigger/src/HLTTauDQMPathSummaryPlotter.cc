#include "DQMOffline/Trigger/interface/HLTTauDQMPathSummaryPlotter.h"
#include "DQMOffline/Trigger/interface/HLTTauDQMPath.h"

HLTTauDQMPathSummaryPlotter::HLTTauDQMPathSummaryPlotter(const edm::ParameterSet& pset, bool doRefAnalysis, const std::string& dqmBaseFolder, double hltMatchDr):
  hltMatchDr_(hltMatchDr),
  doRefAnalysis_(doRefAnalysis)
{
  dqmBaseFolder_ = dqmBaseFolder;

  try {
    triggerTag_         = pset.getUntrackedParameter<std::string>("DQMFolder");
  } catch(cms::Exception& e) {
    edm::LogInfo("HLTTauDQMOffline") << "HLTTauDQMPathSummaryPlotter::HLTTauDQMPathSummaryPlotter(): " << e.what();
    validity_ = false;
    return;
  }
  validity_ = true;
}

HLTTauDQMPathSummaryPlotter::~HLTTauDQMPathSummaryPlotter() {
}

void HLTTauDQMPathSummaryPlotter::beginRun(const std::vector<const HLTTauDQMPath *>& pathObjects) {
  pathObjects_ = pathObjects;

  if (store_) {
    //Create the histograms
    store_->setCurrentFolder(triggerTag()+"/helpers");
    store_->removeContents();

    all_events = store_->book1D("RefEvents", "All events", pathObjects_.size(), 0, pathObjects_.size());
    accepted_events = store_->book1D("PathTriggerBits","Accepted Events per Path;;entries", pathObjects_.size(), 0, pathObjects_.size());
    for(size_t i=0; i<pathObjects_.size(); ++i) {
      accepted_events->setBinLabel(i+1, pathObjects_[i]->getPathName());
    }

    store_->setCurrentFolder(triggerTag());
    store_->removeContents();
  }
}

void HLTTauDQMPathSummaryPlotter::analyze(const edm::TriggerResults& triggerResults, const trigger::TriggerEvent& triggerEvent, const HLTTauDQMOfflineObjects& refCollection) {
  if(doRefAnalysis_) {
    std::vector<HLTTauDQMPath::Object> triggerObjs;
    std::vector<HLTTauDQMPath::Object> matchedTriggerObjs;
    HLTTauDQMOfflineObjects matchedOfflineObjs;

    for(size_t i=0; i<pathObjects_.size(); ++i) {
      const HLTTauDQMPath *path = pathObjects_[i];
      const int lastFilter = path->filtersSize()-1;

      if(path->goodOfflineEvent(lastFilter, refCollection)) {
        all_events->Fill(i+0.5);
      }
      if(path->fired(triggerResults)) {
        triggerObjs.clear();
        matchedTriggerObjs.clear();
        matchedOfflineObjs.clear();
        path->getFilterObjects(triggerEvent, lastFilter, triggerObjs);
        if(path->offlineMatching(lastFilter, triggerObjs, refCollection, hltMatchDr_, matchedTriggerObjs, matchedOfflineObjs)) {
          accepted_events->Fill(i+0.5);
        }
      }
    }
  }
  else {
    for(size_t i=0; i<pathObjects_.size(); ++i) {
      const HLTTauDQMPath *path = pathObjects_[i];
      all_events->Fill(i+0.5);
      if(path->fired(triggerResults)) {
        accepted_events->Fill(i+0.5);
      }
    }
  }
}

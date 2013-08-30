#include "DQMOffline/Trigger/interface/HLTTauDQMPathPlotter2.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

HLTTauDQMPathPlotter2::HLTTauDQMPathPlotter2(const edm::ParameterSet& pset, bool doRefAnalysis, const std::string& dqmBaseFolder,
                                             const std::string& hltProcess, int ptbins, int etabins, int phibins,
                                             double l1MatchDr, double hltMatchDr):
  hltProcess_(hltProcess),
  ptbins_(ptbins),
  etabins_(etabins),
  phibins_(phibins),
  l1MatchDr_(l1MatchDr),
  hltMatchDr_(hltMatchDr),
  doRefAnalysis_(doRefAnalysis),
  hltPath_(hltProcess, doRefAnalysis_)
{
  dqmBaseFolder_ = dqmBaseFolder;

  // Parse configuration
  try {
    triggerTag_         = pset.getUntrackedParameter<std::string>("DQMFolder");
    hltPath_.initialize(pset);
  } catch(cms::Exception& e) {
    edm::LogInfo("HLTTauDQMOffline") << "HLTTauDQMPathPlotter2::HLTTauDQMPathPlotter2(): " << e.what();
    validity_ = false;
    return;
  }
  validity_ = true;
}

void HLTTauDQMPathPlotter2::beginRun(const HLTConfigProvider& HLTCP) {
  if(!validity_)
    return;

  // Identify the correct HLT path
  if(!HLTCP.inited()) {
    edm::LogInfo("HLTTauDQMOffline") << "HLTTauDQMPathPlotter2::beginRun(): HLTConfigProvider is not initialized!";
    validity_ = false;
    return;
  }

  // Search path candidates
  validity_ = hltPath_.beginRun(HLTCP);
  if(!validity_)
    return;

  // Book histograms
  if(store_) {
    store_->setCurrentFolder(triggerTag());
    store_->removeContents();

    hAcceptedEvents_ = store_->book1D("EventsPerFilter","Accepted Events per filter;;entries", hltPath_.filtersSize(), 0, hltPath_.filtersSize());
    for(size_t i=0; i<hltPath_.filtersSize(); ++i) {
      hAcceptedEvents_->setBinLabel(i+1, hltPath_.getFilterName(i));
    }

    hTrigTauEt_ = store_->book1D("TrigTauEt",   "#tau E_{t}", ptbins_,     0, 100);
    hTrigTauEta_ = store_->book1D("TrigTauEta", "#tau #eta",  etabins_, -2.5, 2.5);
    hTrigTauPhi_ = store_->book1D("TrigTauPhi", "#tau #phi",  phibins_, -3.2, 3.2);
  }
}


HLTTauDQMPathPlotter2::~HLTTauDQMPathPlotter2() {}

void HLTTauDQMPathPlotter2::analyze(const edm::TriggerResults& triggerResults, const trigger::TriggerEvent& triggerEvent, const std::map<int, LVColl>& refCollection) {

  std::vector<HLTTauDQMPath::Object> triggerObjs;
  std::vector<HLTTauDQMPath::Object> matchedTriggerObjs;
  LVColl matchedOfflineObjs;

  // Events per filter
  const int lastPassedFilter = hltPath_.lastPassedFilter(triggerResults);
  //std::cout << "Last passed filter " << lastPassedFilter << " " << (lastPassedFilter >= 0 ? hltPath_.getFilterName(lastPassedFilter) : "") << std::endl;
  if(doRefAnalysis_) {
    double matchDr = hltPath_.isFirstFilterL1Seed() ? l1MatchDr_ : hltMatchDr_;
    for(int i=0; i<=lastPassedFilter; ++i) {
      triggerObjs.clear();
      matchedTriggerObjs.clear();
      matchedOfflineObjs.clear();
      hltPath_.getFilterObjects(triggerEvent, i, triggerObjs);
      //std::cout << "Filter name " << hltPath_.getFilterName(i) << " nobjs " << triggerObjs.size() << std::endl;
      bool matched = hltPath_.offlineMatching(i, triggerObjs, refCollection, matchDr, matchedTriggerObjs, matchedOfflineObjs);
      //std::cout << "  offline matching: " << matched << std::endl;
      matchDr = hltMatchDr_;
      if(!matched)
        break;

      hAcceptedEvents_->Fill(i+0.5);
    }
  }
  else {
    for(int i=0; i<=lastPassedFilter; ++i) {
      hAcceptedEvents_->Fill(i+0.5);
    }
  }


  // Triggered tau kinematics
  if(hltPath_.fired(triggerResults)) {
    triggerObjs.clear();
    matchedTriggerObjs.clear();
    matchedOfflineObjs.clear();
    hltPath_.getFilterObjects(triggerEvent, lastPassedFilter, triggerObjs);
    if(doRefAnalysis_) {
      bool matched = hltPath_.offlineMatching(lastPassedFilter, triggerObjs, refCollection, hltMatchDr_, matchedTriggerObjs, matchedOfflineObjs);
      if(matched)
        triggerObjs.swap(matchedTriggerObjs);
      else
        triggerObjs.clear();
    }

    for(const HLTTauDQMPath::Object& obj: triggerObjs) {
      if(obj.id != trigger::TriggerTau)
        continue;
      hTrigTauEt_->Fill(obj.object.pt());
      hTrigTauEta_->Fill(obj.object.eta());
      hTrigTauPhi_->Fill(obj.object.phi());
    }
  }
}

#include "DQMOffline/Trigger/interface/HLTTauDQMPathPlotter2.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

HLTTauDQMPathPlotter2::HLTTauDQMPathPlotter2(const edm::ParameterSet& pset, bool doRefAnalysis, const std::string& dqmBaseFolder,
                                             const std::string& hltProcess, int ptbins, int etabins, int phibins,
                                             double hltMatchDr):
  hltProcess_(hltProcess),
  ptbins_(ptbins),
  etabins_(etabins),
  phibins_(phibins),
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

  // Events per filter
  const int lastPassedFilter = hltPath_.lastPassedFilter(triggerResults);
  if(doRefAnalysis_) {
    std::vector<HLTTauDQMPath::Object> objs;
    //std::cout << "Last passed filter " << lastPassedFilter << " " << (lastPassedFilter >= 0 ? hltPath_.getFilterName(lastPassedFilter) : "") << std::endl;
    for(int i=0; i<=lastPassedFilter; ++i) {
      //std::cout << "Filter name " << hltPath_.getFilterName(i) << std::endl;
      hltPath_.getFilterObjects(triggerEvent, i, objs);
      /*
      for(const HLTTauDQMPath::Object& obj: objs)
        //std::cout << "  object id " << obj.id << std::endl;
        */
      if(!hltPath_.offlineMatching(i, objs, refCollection, hltMatchDr_)) {
        //std::cout << "  offline matching: false" << std::endl;
        break;
      }
      //std::cout << "  offline matching: true" << std::endl;

      hAcceptedEvents_->Fill(i+0.5);
      objs.clear();
    }
  }
  else {
    for(int i=0; i<=lastPassedFilter; ++i) {
      hAcceptedEvents_->Fill(i+0.5);
    }
  }


  // Triggered tau kinematics
  if(hltPath_.fired(triggerResults)) {
    //std::cout << "Path " << pathName_ << std::endl;
    std::vector<HLTTauDQMPath::Object> objs;
    hltPath_.getFilterObjects(triggerEvent, lastPassedFilter, objs);
    for(const HLTTauDQMPath::Object& obj: objs) {
      if(obj.id != trigger::TriggerTau)
        continue;
      hTrigTauEt_->Fill(obj.object.pt());
      hTrigTauEta_->Fill(obj.object.eta());
      hTrigTauPhi_->Fill(obj.object.phi());
    }
  }
}

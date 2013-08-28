#include "DQMOffline/Trigger/interface/HLTTauDQMPathPlotter2.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

HLTTauDQMPathPlotter2::HLTTauDQMPathPlotter2(const edm::ParameterSet& pset, bool doRefAnalysis, const std::string& dqmBaseFolder,
                                             const std::string& hltProcess, int ptbins, int etabins, int phibins):
  hltProcess_(hltProcess),
  ptbins_(ptbins),
  etabins_(etabins),
  phibins_(phibins),
  doRefAnalysis_(doRefAnalysis),
  hltPath_(doRefAnalysis_)
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

    hAcceptedEvents_ = store_->book1D("EventsPerFilter","Accepted Events per Filter;;entries", hltPath_.filtersSize(), 0, hltPath_.filtersSize());
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
  int fillUntilBin = hltPath_.lastPassedFilter(triggerResults);
  for(int bin=0; bin<=fillUntilBin; ++bin) {
    hAcceptedEvents_->Fill(bin+0.5);
  }


  // Triggered tau kinematics
  if(hltPath_.fired(triggerResults)) {
    //std::cout << "Path " << pathName_ << std::endl;
    trigger::size_type filterIndex = triggerEvent.filterIndex(edm::InputTag(hltPath_.getLastFilterName(), "", hltProcess_));
    if(filterIndex != triggerEvent.sizeFilters()) {
      const trigger::Keys& keys = triggerEvent.filterKeys(filterIndex);
      const trigger::Vids& ids = triggerEvent.filterIds(filterIndex);
      const trigger::TriggerObjectCollection& triggerObjects = triggerEvent.getObjects();
      for(size_t i=0; i<keys.size(); ++i) {
        if(ids[i] == trigger::TriggerTau) {
          const trigger::TriggerObject& object = triggerObjects[keys[i]];
          hTrigTauEt_->Fill(object.pt());
          hTrigTauEta_->Fill(object.eta());
          hTrigTauPhi_->Fill(object.phi());
        }
        //std::cout << "Id " << object.id() << " pt " << object.pt() << " id2 " << ids[i] << std::endl;
      }
    }
    else {
      //std::cout << "No index for filter " << std::get<0>(filterIndices_.back()) << std::endl;
    }
  }
}

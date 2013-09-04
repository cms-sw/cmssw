#include "DQMOffline/Trigger/interface/HLTTauDQMPathPlotter.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

HLTTauDQMPathPlotter::HLTTauDQMPathPlotter(const edm::ParameterSet& pset, bool doRefAnalysis, const std::string& dqmBaseFolder,
                                             const std::string& hltProcess, int ptbins, int etabins, int phibins,
                                             double l1MatchDr, double hltMatchDr):
  HLTTauDQMPlotter(pset, dqmBaseFolder),
  ptbins_(ptbins),
  etabins_(etabins),
  phibins_(phibins),
  l1MatchDr_(l1MatchDr),
  hltMatchDr_(hltMatchDr),
  doRefAnalysis_(doRefAnalysis),
  hltPath_(hltProcess, doRefAnalysis_)
{
  if(!configValid_)
    return;

  // Parse configuration
  try {
    hltPath_.initialize(pset);
  } catch(cms::Exception& e) {
    edm::LogWarning("HLTTauDQMOffline") << "HLTTauDQMPathPlotter::HLTTauDQMPathPlotter(): " << e.what();
    configValid_ = false;
    return;
  }
  configValid_ = true;
}

void HLTTauDQMPathPlotter::beginRun(const HLTConfigProvider& HLTCP) {
  if(!configValid_)
    return;

  // Identify the correct HLT path
  if(!HLTCP.inited()) {
    edm::LogWarning("HLTTauDQMOffline") << "HLTTauDQMPathPlotter::beginRun(): HLTConfigProvider is not initialized!";
    runValid_ = false;
    return;
  }

  // Search path candidates
  runValid_ = hltPath_.beginRun(HLTCP);
  if(!runValid_)
    return;

  // Book histograms
  edm::Service<DQMStore> store;
  if(store.isAvailable()) {
    store->setCurrentFolder(triggerTag());
    store->removeContents();

    hAcceptedEvents_ = store->book1D("EventsPerFilter", "Accepted Events per filter;;entries", hltPath_.filtersSize(), 0, hltPath_.filtersSize());
    for(size_t i=0; i<hltPath_.filtersSize(); ++i) {
      hAcceptedEvents_->setBinLabel(i+1, hltPath_.getFilterName(i));
    }

    hTrigTauEt_ = store->book1D("TrigTauEt",   "#tau E_{t}", ptbins_,     0, 100);
    hTrigTauEta_ = store->book1D("TrigTauEta", "#tau #eta",  etabins_, -2.5, 2.5);
    hTrigTauPhi_ = store->book1D("TrigTauPhi", "#tau #phi",  phibins_, -3.2, 3.2);

    // Book di-object invariant mass histogram only for mu+tau, ele+tau, and di-tau paths
    hMass_ = nullptr;
    if(doRefAnalysis_) {
      const int lastFilter = hltPath_.filtersSize()-1;
      const int ntaus = hltPath_.getFilterNTaus(lastFilter);
      const int neles = hltPath_.getFilterNElectrons(lastFilter);
      const int nmus = hltPath_.getFilterNMuons(lastFilter);
      if(ntaus+neles+nmus == 2) {
        hMass_ = store->book1D("OfflineMass", "Invariant mass of offline "+dqmFolder_, 100, 0, 500);
      }
    }
    runValid_ = true;
  }
  else
    runValid_ = false;
}


HLTTauDQMPathPlotter::~HLTTauDQMPathPlotter() {}

void HLTTauDQMPathPlotter::analyze(const edm::TriggerResults& triggerResults, const trigger::TriggerEvent& triggerEvent, const HLTTauDQMOfflineObjects& refCollection) {

  std::vector<HLTTauDQMPath::Object> triggerObjs;
  std::vector<HLTTauDQMPath::Object> matchedTriggerObjs;
  HLTTauDQMOfflineObjects matchedOfflineObjs;

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


  if(hltPath_.fired(triggerResults)) {
    triggerObjs.clear();
    matchedTriggerObjs.clear();
    matchedOfflineObjs.clear();
    hltPath_.getFilterObjects(triggerEvent, lastPassedFilter, triggerObjs);
    if(doRefAnalysis_) {
      bool matched = hltPath_.offlineMatching(lastPassedFilter, triggerObjs, refCollection, hltMatchDr_, matchedTriggerObjs, matchedOfflineObjs);
      // Di-object invariant mass
      if(hMass_ && matched) {
        const int ntaus = hltPath_.getFilterNTaus(lastPassedFilter);
        if(ntaus == 2) {
          // Di-tau (matchedOfflineObjs are already sorted)
          hMass_->Fill( (matchedOfflineObjs.taus[0]+matchedOfflineObjs.taus[1]).M() );
        }
        // Electron+tau
        else if(ntaus == 1 && hltPath_.getFilterNElectrons(lastPassedFilter) == 1) {
          hMass_->Fill( (matchedOfflineObjs.taus[0]+matchedOfflineObjs.electrons[0]).M() );
        }
        // Muon+tau
        else if(ntaus == 1 && hltPath_.getFilterNMuons(lastPassedFilter) == 1) {
          hMass_->Fill( (matchedOfflineObjs.taus[0]+matchedOfflineObjs.muons[0]).M() );
        }
      }

      if(matched)
        triggerObjs.swap(matchedTriggerObjs);
      else
        triggerObjs.clear();
    }

    // Triggered tau kinematics
    for(const HLTTauDQMPath::Object& obj: triggerObjs) {
      if(obj.id != trigger::TriggerTau)
        continue;
      hTrigTauEt_->Fill(obj.object.pt());
      hTrigTauEta_->Fill(obj.object.eta());
      hTrigTauPhi_->Fill(obj.object.phi());
    }
  }
}

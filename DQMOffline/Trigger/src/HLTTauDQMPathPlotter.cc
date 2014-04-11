#include "DQMOffline/Trigger/interface/HLTTauDQMPathPlotter.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

namespace {
  std::string stripVersion(const std::string& pathName) {
    size_t versionStart = pathName.rfind("_v");
    if(versionStart == std::string::npos)
      return pathName;
    return pathName.substr(0, versionStart);
  }
}

HLTTauDQMPathPlotter::HLTTauDQMPathPlotter(const std::string& pathName, const HLTConfigProvider& HLTCP,
                                           bool doRefAnalysis, const std::string& dqmBaseFolder,
                                           const std::string& hltProcess, int ptbins, int etabins, int phibins,
                                           double ptmax, double highptmax,
                                           double l1MatchDr, double hltMatchDr):
  HLTTauDQMPlotter(stripVersion(pathName), dqmBaseFolder),
  ptbins_(ptbins),
  etabins_(etabins),
  phibins_(phibins),
  ptmax_(ptmax),
  highptmax_(highptmax),
  l1MatchDr_(l1MatchDr),
  hltMatchDr_(hltMatchDr),
  doRefAnalysis_(doRefAnalysis),
  hltPath_(pathName, hltProcess, doRefAnalysis_, HLTCP)
{
  configValid_ = configValid_ && hltPath_.isValid();
}

void HLTTauDQMPathPlotter::bookHistograms(DQMStore::IBooker &iBooker) {
  if(!isValid())
    return;

  // Book histograms
  iBooker.setCurrentFolder(triggerTag());

  hAcceptedEvents_ = iBooker.book1D("EventsPerFilter", "Accepted Events per filter;;entries", hltPath_.filtersSize(), 0, hltPath_.filtersSize());
  for(size_t i=0; i<hltPath_.filtersSize(); ++i) {
    hAcceptedEvents_->setBinLabel(i+1, hltPath_.getFilterName(i));
  }

  hTrigTauEt_ = iBooker.book1D("TrigTauEt",   "Triggered #tau p_{T};#tau p_{T};entries", ptbins_,     0, ptmax_);
  hTrigTauEta_ = iBooker.book1D("TrigTauEta", "Triggered #tau #eta;#tau #eta;entries",  etabins_, -2.5, 2.5);
  hTrigTauPhi_ = iBooker.book1D("TrigTauPhi", "Triggered #tau #phi;#tau #phi;entries",  phibins_, -3.2, 3.2);

  // Efficiency helpers
  if(doRefAnalysis_) {
    iBooker.setCurrentFolder(triggerTag()+"/helpers");
    if(hltPath_.hasL2Taus()) {
      hL2TrigTauEtEffNum_    = iBooker.book1D("L2TrigTauEtEffNum",    "L2 #tau p_{T} efficiency;Ref #tau p_{T};entries", ptbins_, 0, ptmax_);
      hL2TrigTauEtEffDenom_  = iBooker.book1D("L2TrigTauEtEffDenom",  "L2 #tau p_{T} denominator;Ref #tau p_{T};entries", ptbins_, 0, ptmax_);
      hL2TrigTauEtaEffNum_   = iBooker.book1D("L2TrigTauEtaEffNum",   "L2 #tau #eta efficiency;Ref #tau #eta;entries", etabins_, -2.5, 2.5);
      hL2TrigTauEtaEffDenom_ = iBooker.book1D("L2TrigTauEtaEffDenom", "L2 #tau #eta denominator;Ref #tau #eta;entries", etabins_, -2.5, 2.5);
      hL2TrigTauPhiEffNum_   = iBooker.book1D("L2TrigTauPhiEffNum",   "L2 #tau #phi efficiency;Ref #tau #phi;entries", phibins_, -3.2, 3.2);
      hL2TrigTauPhiEffDenom_ = iBooker.book1D("L2TrigTauPhiEffDenom", "L2 #tau #phi denominator;Ref #tau #phi;entries", phibins_, -3.2, 3.2);
      hL2TrigTauHighEtEffNum_   = iBooker.book1D("L2TrigTauHighEtEffNum",    "L2 #tau p_{T} efficiency (high p_{T})Ref #tau p_{T};entries", ptbins_, 0, highptmax_);
      hL2TrigTauHighEtEffDenom_ = iBooker.book1D("L2TrigTauHighEtEffDenom",  "L2 #tau p_{T} denominator (high p_{T})Ref #tau p_{T};entries", ptbins_, 0, highptmax_);
    }

    if(hltPath_.hasL3Taus()) {
      hL3TrigTauEtEffNum_    = iBooker.book1D("L3TrigTauEtEffNum",    "L3 #tau p_{T} efficiency;Ref #tau p_{T};entries", ptbins_, 0, ptmax_);
      hL3TrigTauEtEffDenom_  = iBooker.book1D("L3TrigTauEtEffDenom",  "L3 #tau p_{T} denominator;Ref #tau p_{T};entries", ptbins_, 0, ptmax_);
      hL3TrigTauEtaEffNum_   = iBooker.book1D("L3TrigTauEtaEffNum",   "L3 #tau #eta efficiency;Ref #tau #eta;entries", etabins_, -2.5, 2.5);
      hL3TrigTauEtaEffDenom_ = iBooker.book1D("L3TrigTauEtaEffDenom", "L3 #tau #eta denominator;Ref #tau #eta;entries", etabins_, -2.5, 2.5);
      hL3TrigTauPhiEffNum_   = iBooker.book1D("L3TrigTauPhiEffNum",   "L3 #tau #phi efficiency;Ref #tau #phi;entries", phibins_, -3.2, 3.2);
      hL3TrigTauPhiEffDenom_ = iBooker.book1D("L3TrigTauPhiEffDenom", "L3 #tau #phi denominator;Ref #tau #phi;entries", phibins_, -3.2, 3.2);
      hL3TrigTauHighEtEffNum_    = iBooker.book1D("L3TrigTauHighEtEffNum",    "L3 #tau p_{T} efficiency (high p_{T});Ref #tau p_{T};entries", ptbins_, 0, highptmax_);
      hL3TrigTauHighEtEffDenom_  = iBooker.book1D("L3TrigTauHighEtEffDenom",  "L3 #tau p_{T} denominator (high p_{T});Ref #tau p_{T};entries", ptbins_, 0, highptmax_);
    }
    iBooker.setCurrentFolder(triggerTag());
  }

  // Book di-object invariant mass histogram only for mu+tau, ele+tau, and di-tau paths
  hMass_ = nullptr;
  if(doRefAnalysis_) {
    const int lastFilter = hltPath_.filtersSize()-1;
    const int ntaus = hltPath_.getFilterNTaus(lastFilter);
    const int neles = hltPath_.getFilterNElectrons(lastFilter);
    const int nmus = hltPath_.getFilterNMuons(lastFilter);
    auto create = [&](const std::string& name) {
      this->hMass_ = iBooker.book1D("ReferenceMass", "Invariant mass of reference "+name+";Reference invariant mass;entries", 100, 0, 500);
    };

    LogDebug("HLTTauDQMOffline") << "Path " << hltPath_.getPathName() << " number of taus " << ntaus << " electrons " << neles << " muons " << nmus;

    if(ntaus == 2)
      create("di-tau");
    else if(ntaus == 1) {
      if(neles == 1)
        create("electron-tau");
      else if(nmus == 1)
        create("muon-tau");
    }
  }
}


HLTTauDQMPathPlotter::~HLTTauDQMPathPlotter() {}

void HLTTauDQMPathPlotter::analyze(const edm::TriggerResults& triggerResults, const trigger::TriggerEvent& triggerEvent, const HLTTauDQMOfflineObjects& refCollection) {

  std::vector<HLTTauDQMPath::Object> triggerObjs;
  std::vector<HLTTauDQMPath::Object> matchedTriggerObjs;
  HLTTauDQMOfflineObjects matchedOfflineObjs;

  // Events per filter
  const int lastPassedFilter = hltPath_.lastPassedFilter(triggerResults);
  int lastMatchedFilter = -1;
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
      lastMatchedFilter = i;
    }
  }
  else {
    for(int i=0; i<=lastPassedFilter; ++i) {
      hAcceptedEvents_->Fill(i+0.5);
    }
  }

  // Efficiency plots
  if(doRefAnalysis_ && lastMatchedFilter >= 0) {
    // L2 taus
    if(hltPath_.hasL2Taus()) {
      // Denominators
      if(static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastFilterBeforeL2TauIndex()) {
        for(const LV& tau: refCollection.taus) {
          hL2TrigTauEtEffDenom_->Fill(tau.pt());
          hL2TrigTauHighEtEffDenom_->Fill(tau.pt());
          hL2TrigTauEtaEffDenom_->Fill(tau.eta());
          hL2TrigTauPhiEffDenom_->Fill(tau.phi());
        }
      }

      // Numerators
      if(static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastL2TauFilterIndex()) {
        triggerObjs.clear();
        matchedTriggerObjs.clear();
        matchedOfflineObjs.clear();
        hltPath_.getFilterObjects(triggerEvent, hltPath_.getLastL2TauFilterIndex(), triggerObjs);
        bool matched = hltPath_.offlineMatching(hltPath_.getLastL2TauFilterIndex(), triggerObjs, refCollection, hltMatchDr_, matchedTriggerObjs, matchedOfflineObjs);
        if(matched) {
          for(const LV& tau: matchedOfflineObjs.taus) {
            hL2TrigTauEtEffNum_->Fill(tau.pt());
            hL2TrigTauHighEtEffNum_->Fill(tau.pt());
            hL2TrigTauEtaEffNum_->Fill(tau.eta());
            hL2TrigTauPhiEffNum_->Fill(tau.phi());
          }
        }
      }
    }

    // L3 taus
    if(hltPath_.hasL3Taus()) {
      // Denominators
      if(static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastFilterBeforeL3TauIndex()) {
        for(const LV& tau: refCollection.taus) {
          hL3TrigTauEtEffDenom_->Fill(tau.pt());
          hL3TrigTauHighEtEffDenom_->Fill(tau.pt());
          hL3TrigTauEtaEffDenom_->Fill(tau.eta());
          hL3TrigTauPhiEffDenom_->Fill(tau.phi());
        }
      }

      // Numerators
      if(static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastL3TauFilterIndex()) {
        triggerObjs.clear();
        matchedTriggerObjs.clear();
        matchedOfflineObjs.clear();
        hltPath_.getFilterObjects(triggerEvent, hltPath_.getLastL3TauFilterIndex(), triggerObjs);
        bool matched = hltPath_.offlineMatching(hltPath_.getLastL3TauFilterIndex(), triggerObjs, refCollection, hltMatchDr_, matchedTriggerObjs, matchedOfflineObjs);
        if(matched) {
          for(const LV& tau: matchedOfflineObjs.taus) {
            hL3TrigTauEtEffNum_->Fill(tau.pt());
            hL3TrigTauHighEtEffNum_->Fill(tau.pt());
            hL3TrigTauEtaEffNum_->Fill(tau.eta());
            hL3TrigTauPhiEffNum_->Fill(tau.phi());
          }
        }
      }
    }
  }

  if(hltPath_.fired(triggerResults)) {
    triggerObjs.clear();
    matchedTriggerObjs.clear();
    matchedOfflineObjs.clear();
    hltPath_.getFilterObjects(triggerEvent, lastPassedFilter, triggerObjs);
    if(doRefAnalysis_) {
      bool matched = hltPath_.offlineMatching(lastPassedFilter, triggerObjs, refCollection, hltMatchDr_, matchedTriggerObjs, matchedOfflineObjs);
      if(matched) {
        // Di-object invariant mass
        if(hMass_) {
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

#include "DQMOffline/Trigger/interface/HLTTauDQMPathPlotter.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

namespace {
  std::string stripVersion(const std::string& pathName) {
    size_t versionStart = pathName.rfind("_v");
    if (versionStart == std::string::npos)
      return pathName;
    return pathName.substr(0, versionStart);
  }
}  // namespace

HLTTauDQMPathPlotter::HLTTauDQMPathPlotter(const std::string& pathName,
                                           const HLTConfigProvider& HLTCP,
                                           bool doRefAnalysis,
                                           const std::string& dqmBaseFolder,
                                           const std::string& hltProcess,
                                           int ptbins,
                                           int etabins,
                                           int phibins,
                                           double ptmax,
                                           double highptmax,
                                           double l1MatchDr,
                                           double hltMatchDr)
    : HLTTauDQMPlotter(stripVersion(pathName), dqmBaseFolder),
      ptbins_(ptbins),
      etabins_(etabins),
      phibins_(phibins),
      ptmax_(ptmax),
      highptmax_(highptmax),
      l1MatchDr_(l1MatchDr),
      hltMatchDr_(hltMatchDr),
      doRefAnalysis_(doRefAnalysis),
      hltPath_(pathName, hltProcess, doRefAnalysis_, HLTCP) {
  configValid_ = configValid_ && hltPath_.isValid();
}

#include <algorithm>
//void HLTTauDQMPathPlotter::bookHistograms(DQMStore::IBooker& iBooker) {
void HLTTauDQMPathPlotter::bookHistograms(IWrapper & iWrapper, DQMStore::IBooker &iBooker) {
  if (!isValid())
    return;

  // Book histograms
  iBooker.setCurrentFolder(triggerTag());

  hAcceptedEvents_ = iWrapper.book1D(iBooker, 
   "EventsPerFilter", "Accepted Events per filter;;entries", hltPath_.filtersSize(), 0, hltPath_.filtersSize(), kEverything);
  for (size_t i = 0; i < hltPath_.filtersSize(); ++i) {
    if(hAcceptedEvents_) hAcceptedEvents_->setBinLabel(i + 1, hltPath_.getFilterName(i));
  }

  // Efficiency helpers
  if (doRefAnalysis_) {
    iBooker.setCurrentFolder(triggerTag() + "/helpers");
    if (hltPath_.hasL2Taus()) {
      hL2TrigTauEtEffNum_ = iWrapper.book1D(iBooker, "L2TrigTauEtEffNum", "L2 #tau p_{T} efficiency;Ref #tau p_{T};entries", ptbins_, 0, ptmax_, kVital);
      hL2TrigTauEtEffDenom_ = iWrapper.book1D(iBooker, 
       "L2TrigTauEtEffDenom", "L2 #tau p_{T} denominator;Ref #tau p_{T};Efficiency", ptbins_, 0, ptmax_, kVital);
      hL2TrigTauEtaEffNum_ =
	iWrapper.book1D(iBooker, "L2TrigTauEtaEffNum", "L2 #tau #eta efficiency;Ref #tau #eta;entries", etabins_, -2.5, 2.5, kEverything);
      hL2TrigTauEtaEffDenom_ = iWrapper.book1D(iBooker, 
					      "L2TrigTauEtaEffDenom", "L2 #tau #eta denominator;Ref #tau #eta;Efficiency", etabins_, -2.5, 2.5, kEverything);
      hL2TrigTauPhiEffNum_ =
	iWrapper.book1D(iBooker, "L2TrigTauPhiEffNum", "L2 #tau #phi efficiency;Ref #tau #phi;entries", phibins_, -3.2, 3.2, kEverything);
      hL2TrigTauPhiEffDenom_ = iWrapper.book1D(iBooker, 
          "L2TrigTauPhiEffDenom", "L2 #tau #phi denominator;Ref #tau #phi;Efficiency", phibins_, -3.2, 3.2);
      hL2TrigTauHighEtEffNum_ = iWrapper.book1D(iBooker, "L2TrigTauHighEtEffNum",
                                               "L2 #tau p_{T} efficiency (high p_{T});Ref #tau p_{T};entries",
                                               ptbins_,
                                               0,
                                               highptmax_, kVital);
      hL2TrigTauHighEtEffDenom_ = iWrapper.book1D(iBooker, "L2TrigTauHighEtEffDenom",
                                                 "L2 #tau p_{T} denominator (high p_{T});Ref #tau p_{T};Efficiency",
                                                 ptbins_,
                                                 0,
                                                 highptmax_, kVital);
    }

    if (hltPath_.hasL3Taus()) {
      hL3TrigTauEtEffNum_ =
	iWrapper.book1D(iBooker, "L3TrigTauEtEffNum", "L3 #tau p_{T} efficiency;Ref #tau p_{T};entries", ptbins_, 0, ptmax_, kEverything);
      hL3TrigTauEtEffDenom_ = iWrapper.book1D(iBooker, "L3TrigTauEtEffDenom", "L3 #tau p_{T} denominator;Ref #tau p_{T};Efficiency", ptbins_, 0, ptmax_, kVital);
      hL3TrigTauEtaEffNum_ =
	iWrapper.book1D(iBooker, "L3TrigTauEtaEffNum", "L3 #tau #eta efficiency;Ref #tau #eta;entries", etabins_, -2.5, 2.5, kVital);
      hL3TrigTauEtaEffDenom_ = iWrapper.book1D(iBooker, "L3TrigTauEtaEffDenom", "L3 #tau #eta denominator;Ref #tau #eta;Efficiency", etabins_, -2.5, 2.5, kEverything);
      hL3TrigTauPhiEffNum_ =
	iWrapper.book1D(iBooker, "L3TrigTauPhiEffNum", "L3 #tau #phi efficiency;Ref #tau #phi;entries", phibins_, -3.2, 3.2, kEverything);
      hL3TrigTauPhiEffDenom_ = iWrapper.book1D(iBooker, "L3TrigTauPhiEffDenom", "L3 #tau #phi denominator;Ref #tau #phi;Efficiency", phibins_, -3.2, 3.2, kEverything);
      hL3TrigTauHighEtEffNum_ = iWrapper.book1D(iBooker, "L3TrigTauHighEtEffNum",
                                               "L3 #tau p_{T} efficiency (high p_{T});Ref #tau p_{T};entries",
                                               ptbins_,
                                               0,
                                               highptmax_, kVital);
      hL3TrigTauHighEtEffDenom_ = iWrapper.book1D(iBooker, "L3TrigTauHighEtEffDenom",
                                                 "L3 #tau p_{T} denominator (high p_{T});Ref #tau p_{T};Efficiency",
                                                 ptbins_,
                                                 0,
                                                 highptmax_, kVital);
      hL3TrigTauEtaPhiEffNum_ = iWrapper.book2D(iBooker, "L3TrigTauEtaPhiEffNum", "L3 efficiency in eta-phi plane", etabins_, -2.5, 2.5, phibins_, -3.2, 3.2, kEverything);
      hL3TrigTauEtaPhiEffDenom_ = iWrapper.book2D(iBooker, "L3TrigTauEtaPhiEffDenom", "L3 denominator in eta-phi plane", etabins_, -2.5, 2.5, phibins_, -3.2, 3.2, kEverything);
      if(hL3TrigTauEtaPhiEffDenom_) hL3TrigTauEtaPhiEffDenom_->setOption("COL");
    }

    if (hltPath_.hasL2Electrons()) {
      hL2TrigElectronEtEffNum_ = iWrapper.book1D(iBooker, "L2TrigElectronEtEffNum", "L2 electron p_{T} efficiency;Ref electron p_{T};entries", ptbins_, 0, ptmax_, kVital);
      hL2TrigElectronEtEffDenom_ = iWrapper.book1D(iBooker, "L2TrigElectronEtEffDenom", "L2 electron p_{T} denominator;Ref electron p_{T};Efficiency", ptbins_, 0, ptmax_, kVital);
      hL2TrigElectronEtaEffNum_ = iWrapper.book1D(iBooker, "L2TrigElectronEtaEffNum", "L2 electron #eta efficiency;Ref electron #eta;entries", etabins_, -2.5, 2.5, kEverything);
      hL2TrigElectronEtaEffDenom_ = iWrapper.book1D(iBooker, "L2TrigElectronEtaEffDenom", "L2 electron #eta denominator;Ref electron #eta;Efficiency", etabins_, -2.5, 2.5, kEverything);
      hL2TrigElectronPhiEffNum_ = iWrapper.book1D(iBooker, "L2TrigElectronPhiEffNum", "L2 electron #phi efficiency;Ref electron #phi;entries", phibins_, -3.2, 3.2, kEverything);
      hL2TrigElectronPhiEffDenom_ = iWrapper.book1D(iBooker, "L2TrigElectronPhiEffDenom", "L2 electron #phi denominator;Ref electron #phi;Efficiency", phibins_, -3.2, 3.2, kEverything);
    }

    if (hltPath_.hasL3Electrons()) {
      hL3TrigElectronEtEffNum_ = iWrapper.book1D(iBooker, 
						"L3TrigElectronEtEffNum", "L3 electron p_{T} efficiency;Ref electron p_{T};entries", ptbins_, 0, ptmax_, kVital);
      hL3TrigElectronEtEffDenom_ = iWrapper.book1D(iBooker, 
						  "L3TrigElectronEtEffDenom", "L3 electron p_{T} denominator;Ref electron p_{T};Efficiency", ptbins_, 0, ptmax_, kVital);
      hL3TrigElectronEtaEffNum_ = iWrapper.book1D(iBooker, 
						 "L3TrigElectronEtaEffNum", "L3 electron #eta efficiency;Ref electron #eta;entries", etabins_, -2.5, 2.5, kEverything);
      hL3TrigElectronEtaEffDenom_ = iWrapper.book1D(iBooker, 
						   "L3TrigElectronEtaEffDenom", "L3 electron #eta denominator;Ref electron #eta;Efficiency", etabins_, -2.5, 2.5, kEverything);
      hL3TrigElectronPhiEffNum_ = iWrapper.book1D(iBooker, 
						 "L3TrigElectronPhiEffNum", "L3 electron #phi efficiency;Ref electron #phi;entries", phibins_, -3.2, 3.2, kEverything);
      hL3TrigElectronPhiEffDenom_ = iWrapper.book1D(iBooker, 
						   "L3TrigElectronPhiEffDenom", "L3 electron #phi denominator;Ref electron #phi;Efficiency", phibins_, -3.2, 3.2, kEverything);
    }

    if (hltPath_.hasL2Muons()) {
      hL2TrigMuonEtEffNum_ =
	iWrapper.book1D(iBooker, "L2TrigMuonEtEffNum", "L2 muon p_{T} efficiency;Ref muon p_{T};entries", ptbins_, 0, ptmax_, kVital);
      hL2TrigMuonEtEffDenom_ = iWrapper.book1D(iBooker, "L2TrigMuonEtEffDenom", "L2 muon p_{T} denominator;Ref muon p_{T};Efficiency", ptbins_, 0, ptmax_, kVital);
      hL2TrigMuonEtaEffNum_ =
          iWrapper.book1D(iBooker, "L2TrigMuonEtaEffNum", "L2 muon #eta efficiency;Ref muon #eta;entries", etabins_, -2.5, 2.5, kVital);
      hL2TrigMuonEtaEffDenom_ = iWrapper.book1D(iBooker, 
          "L2TrigMuonEtaEffDenom", "L2 muon #eta denominator;Ref muon #eta;Efficiency", etabins_, -2.5, 2.5, kEverything);
      hL2TrigMuonPhiEffNum_ =
          iWrapper.book1D(iBooker, "L2TrigMuonPhiEffNum", "L2 muon #phi efficiency;Ref muon #phi;entries", phibins_, -3.2, 3.2, kEverything);
      hL2TrigMuonPhiEffDenom_ = iWrapper.book1D(iBooker, 
          "L2TrigMuonPhiEffDenom", "L2 muon #phi denominator;Ref muon #phi;Efficiency", phibins_, -3.2, 3.2, kEverything);
    }

    if (hltPath_.hasL3Muons()) {
      hL3TrigMuonEtEffNum_ =
	iWrapper.book1D(iBooker, "L3TrigMuonEtEffNum", "L3 muon p_{T} efficiency;Ref muon p_{T};entries", ptbins_, 0, ptmax_, kVital);
      hL3TrigMuonEtEffDenom_ = iWrapper.book1D(iBooker, 
       "L3TrigMuonEtEffDenom", "L3 muon p_{T} denominator;Ref muon p_{T};Efficiency", ptbins_, 0, ptmax_, kVital);
      hL3TrigMuonEtaEffNum_ =
	iWrapper.book1D(iBooker, "L3TrigMuonEtaEffNum", "L3 muon #eta efficiency;Ref muon #eta;entries", etabins_, -2.5, 2.5, kEverything);
      hL3TrigMuonEtaEffDenom_ = iWrapper.book1D(iBooker, 
	"L3TrigMuonEtaEffDenom", "L3 muon #eta denominator;Ref muon #eta;Efficiency", etabins_, -2.5, 2.5, kEverything);
      hL3TrigMuonPhiEffNum_ =
	iWrapper.book1D(iBooker, "L3TrigMuonPhiEffNum", "L3 muon #phi efficiency;Ref muon #phi;entries", phibins_, -3.2, 3.2, kEverything);
      hL3TrigMuonPhiEffDenom_ = iWrapper.book1D(iBooker, 
	"L3TrigMuonPhiEffDenom", "L3 muon #phi denominator;Ref muon #phi;Efficiency", phibins_, -3.2, 3.2, kEverything);
    }

    if (hltPath_.hasL2CaloMET()) {
      hL2TrigMETEtEffNum_ =
	iWrapper.book1D(iBooker, "L2TrigMETEtEffNum", "L2 MET efficiency;Ref MET;entries", ptbins_, 0, ptmax_, kVital);
      hL2TrigMETEtEffDenom_ =
	iWrapper.book1D(iBooker, "L2TrigMETEtEffDenom", "L2 MET denominator;Ref MET;Efficiency", ptbins_, 0, ptmax_, kVital);
    }

    iBooker.setCurrentFolder(triggerTag());
  }

  // Book di-object invariant mass histogram only for mu+tau, ele+tau, and di-tau paths
  hMass_ = nullptr;
  if (doRefAnalysis_) {
    const int ntaus = hltPath_.getFilterNTaus(hltPath_.getLastL3TauFilterIndex());
    const int neles = hltPath_.getFilterNElectrons(hltPath_.getLastL3ElectronFilterIndex());
    const int nmus = hltPath_.getFilterNMuons(hltPath_.getLastL3MuonFilterIndex());

    int nmet = 0;
    int lastMatchedMETFilter = -1;
    for (size_t i = 0; i < hltPath_.filtersSize(); ++i) {
      if (hltPath_.getFilterName(i).find("hltMET") < hltPath_.getFilterName(i).length())
        lastMatchedMETFilter = i;
    }
    if (lastMatchedMETFilter >= 0)
      nmet = hltPath_.getFilterMET(lastMatchedMETFilter);
    auto create = [&](const std::string& name) {
      if (name == "tau-met") {
        this->hMass_ = iWrapper.book1D(iBooker, 
            "ReferenceMass", "Transverse mass of reference " + name + ";Reference transverse mass;entries", 100, 0, 500);
      } else {
        this->hMass_ = iWrapper.book1D(iBooker, 
            "ReferenceMass", "Invariant mass of reference " + name + ";Reference invariant mass;entries", 100, 0, 500);
      }
    };
    LogDebug("HLTTauDQMOffline") << "Path " << hltPath_.getPathName() << " number of taus " << ntaus << " electrons "
                                 << neles << " muons " << nmus;
    if (ntaus > 0) {
      hTrigTauEt_ = iWrapper.book1D(iBooker, "TrigTauEt", "Triggered #tau p_{T};#tau p_{T};entries", ptbins_, 0, ptmax_);
      hTrigTauEta_ = iWrapper.book1D(iBooker, "TrigTauEta", "Triggered #tau #eta;#tau #eta;entries", etabins_, -2.5, 2.5);
      hTrigTauPhi_ = iWrapper.book1D(iBooker, "TrigTauPhi", "Triggered #tau #phi;#tau #phi;entries", phibins_, -3.2, 3.2);
    }
    if (neles > 0) {
      hTrigElectronEt_ =
          iWrapper.book1D(iBooker, "TrigElectronEt", "Triggered electron p_{T};electron p_{T};entries", ptbins_, 0, ptmax_);
      hTrigElectronEta_ =
          iWrapper.book1D(iBooker, "TrigElectronEta", "Triggered electron #eta;electron #eta;entries", etabins_, -2.5, 2.5);
      hTrigElectronPhi_ =
          iWrapper.book1D(iBooker, "TrigElectronPhi", "Triggered electron #phi;electron #phi;entries", phibins_, -3.2, 3.2);
    }
    if (nmus > 0) {
      hTrigMuonEt_ = iWrapper.book1D(iBooker, "TrigMuonEt", "Triggered muon p_{T};muon p_{T};entries", ptbins_, 0, ptmax_);
      hTrigMuonEta_ = iWrapper.book1D(iBooker, "TrigMuonEta", "Triggered muon #eta;muon #eta;entries", etabins_, -2.5, 2.5);
      hTrigMuonPhi_ = iWrapper.book1D(iBooker, "TrigMuonPhi", "Triggered muon #phi;muon #phi;entries", phibins_, -3.2, 3.2);
    }
    if (nmet > 0) {
      hTrigMETEt_ = iWrapper.book1D(iBooker, "TrigMETEt", "Triggered MET E_{T};MET E_{T};entries", ptbins_, 0, ptmax_);
      hTrigMETPhi_ = iWrapper.book1D(iBooker, "TrigMETPhi", "Triggered MET #phi;MET #phi;entries", phibins_, -3.2, 3.2);
    }
    if (ntaus == 2 && neles == 0 && nmus == 0 && nmet == 0)
      create("di-tau");
    if (ntaus == 1 && neles == 1 && nmus == 0 && nmet == 0)
      create("electron-tau");
    if (ntaus == 1 && neles == 0 && nmus == 1 && nmet == 0)
      create("muon-tau");
    if (ntaus == 1 && neles == 0 && nmus == 0 && nmet == 1)
      create("tau-met");
  }
}

HLTTauDQMPathPlotter::~HLTTauDQMPathPlotter() = default;

void HLTTauDQMPathPlotter::analyze(const edm::TriggerResults& triggerResults,
                                   const trigger::TriggerEvent& triggerEvent,
                                   const HLTTauDQMOfflineObjects& refCollection) {
  std::vector<HLTTauDQMPath::Object> triggerObjs;
  std::vector<HLTTauDQMPath::Object> matchedTriggerObjs;
  HLTTauDQMOfflineObjects matchedOfflineObjs;

  // Events per filter
  const int lastPassedFilter = hltPath_.lastPassedFilter(triggerResults);
  int lastMatchedFilter = -1;
  int lastMatchedMETFilter = -1;
  int lastMatchedElectronFilter = -1;
  int lastMatchedMuonFilter = -1;
  int lastMatchedTauFilter = -1;
  int firstMatchedMETFilter = -1;

  if (doRefAnalysis_) {
    double matchDr = hltPath_.isFirstFilterL1Seed() ? l1MatchDr_ : hltMatchDr_;
    for (int i = 0; i <= lastPassedFilter; ++i) {
      triggerObjs.clear();
      matchedTriggerObjs.clear();
      matchedOfflineObjs.clear();
      hltPath_.getFilterObjects(triggerEvent, i, triggerObjs);
      //std::cout << "Filter name " << hltPath_.getFilterName(i) << " nobjs " << triggerObjs.size() << " " << "ref size " << refCollection.taus.size() << std::endl;
      bool matched =
          hltPath_.offlineMatching(i, triggerObjs, refCollection, matchDr, matchedTriggerObjs, matchedOfflineObjs);
      //std::cout << "  offline matching: " << matched << " " << matchedTriggerObjs.size() << std::endl;
      matchDr = hltMatchDr_;
      if (!matched)
        break;

      if(hAcceptedEvents_) hAcceptedEvents_->Fill(i + 0.5);
      lastMatchedFilter = i;
      if (hltPath_.getFilterName(i).find("hltMET") < hltPath_.getFilterName(i).length())
        lastMatchedMETFilter = i;
      if (hltPath_.getFilterType(i) == "HLTMuonL3PreFilter" || hltPath_.getFilterType(i) == "HLTMuonIsoFilter")
        lastMatchedMuonFilter = i;
      if (hltPath_.getFilterName(i).find("hltEle") < hltPath_.getFilterName(i).length())
        lastMatchedElectronFilter = i;
      if (hltPath_.getFilterName(i).find("hltPFTau") < hltPath_.getFilterName(i).length() ||
          hltPath_.getFilterName(i).find("hltHpsPFTau") < hltPath_.getFilterName(i).length() ||
          hltPath_.getFilterName(i).find("hltDoublePFTau") < hltPath_.getFilterName(i).length() ||
          hltPath_.getFilterName(i).find("hltHpsDoublePFTau") < hltPath_.getFilterName(i).length())
        lastMatchedTauFilter = i;
      if (firstMatchedMETFilter < 0 && hltPath_.getFilterName(i).find("hltMET") < hltPath_.getFilterName(i).length())
        firstMatchedMETFilter = i;
    }
  } else {
    for (int i = 0; i <= lastPassedFilter; ++i) {
      if(hAcceptedEvents_) hAcceptedEvents_->Fill(i + 0.5);
    }
  }

  // Efficiency plots
  if (doRefAnalysis_ && lastMatchedFilter >= 0) {
    // L2 taus
    if (hltPath_.hasL2Taus()) {
      // Denominators
      if (static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastFilterBeforeL2TauIndex()) {
        for (const LV& tau : refCollection.taus) {
          if(hL2TrigTauEtEffDenom_) hL2TrigTauEtEffDenom_->Fill(tau.pt());
          if(hL2TrigTauHighEtEffDenom_) hL2TrigTauHighEtEffDenom_->Fill(tau.pt());
          if(hL2TrigTauEtaEffDenom_) hL2TrigTauEtaEffDenom_->Fill(tau.eta());
          if(hL2TrigTauPhiEffDenom_) hL2TrigTauPhiEffDenom_->Fill(tau.phi());
        }
      }

      // Numerators
      if (static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastL2TauFilterIndex()) {
        triggerObjs.clear();
        matchedTriggerObjs.clear();
        matchedOfflineObjs.clear();
        hltPath_.getFilterObjects(triggerEvent, hltPath_.getLastL2TauFilterIndex(), triggerObjs);
        bool matched = hltPath_.offlineMatching(hltPath_.getLastL2TauFilterIndex(),
                                                triggerObjs,
                                                refCollection,
                                                hltMatchDr_,
                                                matchedTriggerObjs,
                                                matchedOfflineObjs);
        if (matched) {
          for (const LV& tau : matchedOfflineObjs.taus) {
            if(hL2TrigTauEtEffNum_) hL2TrigTauEtEffNum_->Fill(tau.pt());
            if(hL2TrigTauHighEtEffNum_) hL2TrigTauHighEtEffNum_->Fill(tau.pt());
            if(hL2TrigTauEtaEffNum_) hL2TrigTauEtaEffNum_->Fill(tau.eta());
            if(hL2TrigTauPhiEffNum_) hL2TrigTauPhiEffNum_->Fill(tau.phi());
          }
        }
      }
    }

    // L3 taus
    if (hltPath_.hasL3Taus()) {
      // Denominators
      if (static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastFilterBeforeL3TauIndex()) {
        for (const LV& tau : refCollection.taus) {
          if(hL3TrigTauEtEffDenom_) hL3TrigTauEtEffDenom_->Fill(tau.pt());
          if(hL3TrigTauHighEtEffDenom_) hL3TrigTauHighEtEffDenom_->Fill(tau.pt());
          if(hL3TrigTauEtaEffDenom_) hL3TrigTauEtaEffDenom_->Fill(tau.eta());
          if(hL3TrigTauPhiEffDenom_) hL3TrigTauPhiEffDenom_->Fill(tau.phi());
          if(hL3TrigTauEtaPhiEffDenom_) hL3TrigTauEtaPhiEffDenom_->Fill(tau.eta(), tau.phi());
        }
      }

      // Numerators
      if (static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastL3TauFilterIndex()) {
        triggerObjs.clear();
        matchedTriggerObjs.clear();
        matchedOfflineObjs.clear();
        hltPath_.getFilterObjects(triggerEvent, hltPath_.getLastL3TauFilterIndex(), triggerObjs);
        bool matched = hltPath_.offlineMatching(hltPath_.getLastL3TauFilterIndex(),
                                                triggerObjs,
                                                refCollection,
                                                hltMatchDr_,
                                                matchedTriggerObjs,
                                                matchedOfflineObjs);
        if (matched) {
          for (const LV& tau : matchedOfflineObjs.taus) {
            if(hL3TrigTauEtEffNum_) hL3TrigTauEtEffNum_->Fill(tau.pt());
            if(hL3TrigTauHighEtEffNum_) hL3TrigTauHighEtEffNum_->Fill(tau.pt());
            if(hL3TrigTauEtaEffNum_) hL3TrigTauEtaEffNum_->Fill(tau.eta());
            if(hL3TrigTauPhiEffNum_) hL3TrigTauPhiEffNum_->Fill(tau.phi());
            if(hL3TrigTauEtaPhiEffNum_) hL3TrigTauEtaPhiEffNum_->Fill(tau.eta(), tau.phi());
          }
        }
      }
    }

    // L2 Electrons
    if (hltPath_.hasL2Electrons()) {
      // Denominators
      if (static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastFilterBeforeL2ElectronIndex()) {
        for (const LV& electron : refCollection.electrons) {
          if(hL2TrigElectronEtEffDenom_) hL2TrigElectronEtEffDenom_->Fill(electron.pt());
          if(hL2TrigElectronEtaEffDenom_) hL2TrigElectronEtaEffDenom_->Fill(electron.eta());
          if(hL2TrigElectronPhiEffDenom_) hL2TrigElectronPhiEffDenom_->Fill(electron.phi());
        }
      }

      // Numerators
      if (static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastL2ElectronFilterIndex()) {
        triggerObjs.clear();
        matchedTriggerObjs.clear();
        matchedOfflineObjs.clear();
        hltPath_.getFilterObjects(triggerEvent, hltPath_.getLastL2ElectronFilterIndex(), triggerObjs);
        bool matched = hltPath_.offlineMatching(hltPath_.getLastL2ElectronFilterIndex(),
                                                triggerObjs,
                                                refCollection,
                                                hltMatchDr_,
                                                matchedTriggerObjs,
                                                matchedOfflineObjs);
        if (matched) {
          for (const LV& electron : matchedOfflineObjs.electrons) {
            if(hL2TrigElectronEtEffNum_) hL2TrigElectronEtEffNum_->Fill(electron.pt());
            if(hL2TrigElectronEtaEffNum_) hL2TrigElectronEtaEffNum_->Fill(electron.eta());
            if(hL2TrigElectronPhiEffNum_) hL2TrigElectronPhiEffNum_->Fill(electron.phi());
          }
        }
      }
    }

    // L3 electron
    if (hltPath_.hasL3Electrons()) {
      // Denominators
      if (static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastFilterBeforeL3ElectronIndex()) {
        for (const LV& electron : refCollection.electrons) {
          if(hL3TrigElectronEtEffDenom_) hL3TrigElectronEtEffDenom_->Fill(electron.pt());
          if(hL3TrigElectronEtaEffDenom_) hL3TrigElectronEtaEffDenom_->Fill(electron.eta());
          if(hL3TrigElectronPhiEffDenom_) hL3TrigElectronPhiEffDenom_->Fill(electron.phi());
        }
      }

      // Numerators
      if (static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastL3ElectronFilterIndex()) {
        triggerObjs.clear();
        matchedTriggerObjs.clear();
        matchedOfflineObjs.clear();
        hltPath_.getFilterObjects(triggerEvent, hltPath_.getLastL3ElectronFilterIndex(), triggerObjs);
        bool matched = hltPath_.offlineMatching(hltPath_.getLastL3ElectronFilterIndex(),
                                                triggerObjs,
                                                refCollection,
                                                hltMatchDr_,
                                                matchedTriggerObjs,
                                                matchedOfflineObjs);
        if (matched) {
          for (const LV& electron : matchedOfflineObjs.electrons) {
            if(hL3TrigElectronEtEffNum_) hL3TrigElectronEtEffNum_->Fill(electron.pt());
            if(hL3TrigElectronEtaEffNum_) hL3TrigElectronEtaEffNum_->Fill(electron.eta());
            if(hL3TrigElectronPhiEffNum_) hL3TrigElectronPhiEffNum_->Fill(electron.phi());
          }
        }
      }
    }

    // L2 Muons
    if (hltPath_.hasL2Muons()) {
      // Denominators
      if (static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastFilterBeforeL2MuonIndex()) {
        for (const LV& muon : refCollection.muons) {
          if(hL2TrigMuonEtEffDenom_) hL2TrigMuonEtEffDenom_->Fill(muon.pt());
          if(hL2TrigMuonEtaEffDenom_) hL2TrigMuonEtaEffDenom_->Fill(muon.eta());
          if(hL2TrigMuonPhiEffDenom_) hL2TrigMuonPhiEffDenom_->Fill(muon.phi());
        }
      }

      // Numerators
      if (static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastL2MuonFilterIndex()) {
        triggerObjs.clear();
        matchedTriggerObjs.clear();
        matchedOfflineObjs.clear();
        hltPath_.getFilterObjects(triggerEvent, hltPath_.getLastL2MuonFilterIndex(), triggerObjs);
        bool matched = hltPath_.offlineMatching(hltPath_.getLastL2MuonFilterIndex(),
                                                triggerObjs,
                                                refCollection,
                                                hltMatchDr_,
                                                matchedTriggerObjs,
                                                matchedOfflineObjs);
        if (matched) {
          for (const LV& muon : matchedOfflineObjs.muons) {
            if(hL2TrigMuonEtEffNum_) hL2TrigMuonEtEffNum_->Fill(muon.pt());
            if(hL2TrigMuonEtaEffNum_) hL2TrigMuonEtaEffNum_->Fill(muon.eta());
            if(hL2TrigMuonPhiEffNum_) hL2TrigMuonPhiEffNum_->Fill(muon.phi());
          }
        }
      }
    }

    // L3 muon
    if (hltPath_.hasL3Muons()) {
      // Denominators
      if (static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastFilterBeforeL3MuonIndex()) {
        for (const LV& muon : refCollection.muons) {
          if(hL3TrigMuonEtEffDenom_) hL3TrigMuonEtEffDenom_->Fill(muon.pt());
          if(hL3TrigMuonEtaEffDenom_) hL3TrigMuonEtaEffDenom_->Fill(muon.eta());
          if(hL3TrigMuonPhiEffDenom_) hL3TrigMuonPhiEffDenom_->Fill(muon.phi());
        }
      }

      // Numerators
      if (static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastL3MuonFilterIndex()) {
        triggerObjs.clear();
        matchedTriggerObjs.clear();
        matchedOfflineObjs.clear();
        hltPath_.getFilterObjects(triggerEvent, hltPath_.getLastL3MuonFilterIndex(), triggerObjs);
        bool matched = hltPath_.offlineMatching(hltPath_.getLastL3MuonFilterIndex(),
                                                triggerObjs,
                                                refCollection,
                                                hltMatchDr_,
                                                matchedTriggerObjs,
                                                matchedOfflineObjs);
        if (matched) {
          for (const LV& muon : matchedOfflineObjs.muons) {
            if(hL3TrigMuonEtEffNum_) hL3TrigMuonEtEffNum_->Fill(muon.pt());
            if(hL3TrigMuonEtaEffNum_) hL3TrigMuonEtaEffNum_->Fill(muon.eta());
            if(hL3TrigMuonPhiEffNum_) hL3TrigMuonPhiEffNum_->Fill(muon.phi());
          }
        }
      }
    }

    // L2 CaloMET
    if (hltPath_.hasL2CaloMET()) {
      // Denominators
      if (static_cast<size_t>(firstMatchedMETFilter) >= hltPath_.getFirstFilterBeforeL2CaloMETIndex()) {
        if(hL2TrigMETEtEffDenom_) hL2TrigMETEtEffDenom_->Fill(refCollection.met[0].pt());
      }

      // Numerators
      if (static_cast<size_t>(lastMatchedMETFilter) >= hltPath_.getLastL2CaloMETFilterIndex()) {
        triggerObjs.clear();
        matchedTriggerObjs.clear();
        matchedOfflineObjs.clear();
        hltPath_.getFilterObjects(triggerEvent, hltPath_.getLastL2CaloMETFilterIndex(), triggerObjs);
        bool matched = hltPath_.offlineMatching(hltPath_.getLastL2CaloMETFilterIndex(),
                                                triggerObjs,
                                                refCollection,
                                                hltMatchDr_,
                                                matchedTriggerObjs,
                                                matchedOfflineObjs);
        if (matched) {
          if(hL2TrigMETEtEffNum_) hL2TrigMETEtEffNum_->Fill(matchedOfflineObjs.met[0].pt());
        }
      }
    }
  }

  if (hltPath_.fired(triggerResults)) {
    triggerObjs.clear();
    matchedTriggerObjs.clear();
    matchedOfflineObjs.clear();

    if (lastMatchedMETFilter >= 0)
      hltPath_.getFilterObjects(triggerEvent, lastMatchedMETFilter, triggerObjs);
    if (lastMatchedMuonFilter >= 0)
      hltPath_.getFilterObjects(triggerEvent, lastMatchedMuonFilter, triggerObjs);
    if (lastMatchedElectronFilter >= 0)
      hltPath_.getFilterObjects(triggerEvent, lastMatchedElectronFilter, triggerObjs);

    if (lastMatchedTauFilter >= 0)
      hltPath_.getFilterObjects(triggerEvent, lastMatchedTauFilter, triggerObjs);

    if (doRefAnalysis_) {
      bool matched = hltPath_.offlineMatching(
          lastPassedFilter, triggerObjs, refCollection, hltMatchDr_, matchedTriggerObjs, matchedOfflineObjs);
      if (matched) {
        // Di-object invariant mass
        if (hMass_) {
          const int ntaus = hltPath_.getFilterNTaus(lastPassedFilter);
          if (ntaus == 2 && hltPath_.getFilterNElectrons(lastMatchedElectronFilter) == 0 &&
              hltPath_.getFilterNMuons(lastMatchedMuonFilter) == 0) {
            // Di-tau (matchedOfflineObjs are already sorted)
            if(hMass_) hMass_->Fill((matchedOfflineObjs.taus[0] + matchedOfflineObjs.taus[1]).M());
          }
          // Electron+tau
          else if (ntaus == 1 && hltPath_.getFilterNElectrons(lastPassedFilter) == 1) {
            if(hMass_) hMass_->Fill((matchedOfflineObjs.taus[0] + matchedOfflineObjs.electrons[0]).M());
          }
          // Muon+tau
          else if (ntaus == 1 && hltPath_.getFilterNMuons(lastPassedFilter) == 1) {
            if(hMass_) hMass_->Fill((matchedOfflineObjs.taus[0] + matchedOfflineObjs.muons[0]).M());
          }
          // Tau+MET
          if (hltPath_.getFilterNTaus(lastPassedFilter) == 1 && hltPath_.getFilterMET(lastMatchedMETFilter) == 1) {
            double taupt = matchedOfflineObjs.taus[0].Pt();
            double tauphi = matchedOfflineObjs.taus[0].Phi();
            double met = matchedOfflineObjs.met[0].Pt();
            double metphi = matchedOfflineObjs.met[0].Phi();
            double mT = sqrt(2 * taupt * met * (1 - cos(tauphi - metphi)));

            if(hMass_) hMass_->Fill(mT);
          }
        }
      }

      // Triggered object kinematics
      for (const HLTTauDQMPath::Object& obj : triggerObjs) {
        if (obj.id == trigger::TriggerTau) {
          if(hTrigTauEt_) hTrigTauEt_->Fill(obj.object.pt());
          if(hTrigTauEta_) hTrigTauEta_->Fill(obj.object.eta());
          if(hTrigTauPhi_) hTrigTauPhi_->Fill(obj.object.phi());
        }
        if (obj.id == trigger::TriggerElectron || obj.id == trigger::TriggerPhoton) {
          if(hTrigElectronEt_) hTrigElectronEt_->Fill(obj.object.pt());
          if(hTrigElectronEta_) hTrigElectronEta_->Fill(obj.object.eta());
          if(hTrigElectronPhi_) hTrigElectronPhi_->Fill(obj.object.phi());
        }
        if (obj.id == trigger::TriggerMuon) {
          if(hTrigMuonEt_) hTrigMuonEt_->Fill(obj.object.pt());
          if(hTrigMuonEta_) hTrigMuonEta_->Fill(obj.object.eta());
          if(hTrigMuonPhi_) hTrigMuonPhi_->Fill(obj.object.phi());
        }
        if (obj.id == trigger::TriggerMET) {
          if(hTrigMETEt_) hTrigMETEt_->Fill(obj.object.pt());
          if(hTrigMETPhi_) hTrigMETPhi_->Fill(obj.object.phi());
        }
      }
    }
  }
}

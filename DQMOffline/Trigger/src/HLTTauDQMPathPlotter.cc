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
void HLTTauDQMPathPlotter::bookHistograms(DQMStore::IBooker& iBooker) {
  if (!isValid())
    return;

  // Book histograms
  iBooker.setCurrentFolder(triggerTag());

  hAcceptedEvents_ = iBooker.book1D(
      "EventsPerFilter", "Accepted Events per filter;;entries", hltPath_.filtersSize(), 0, hltPath_.filtersSize());
  for (size_t i = 0; i < hltPath_.filtersSize(); ++i) {
    hAcceptedEvents_->setBinLabel(i + 1, hltPath_.getFilterName(i));
  }

  // Efficiency helpers
  if (doRefAnalysis_) {
    iBooker.setCurrentFolder(triggerTag() + "/helpers");
    if (hltPath_.hasL2Taus()) {
      hL2TrigTauEtEffNum_ =
          iBooker.book1D("L2TrigTauEtEffNum", "L2 #tau p_{T} efficiency;Ref #tau p_{T};entries", ptbins_, 0, ptmax_);
      hL2TrigTauEtEffDenom_ = iBooker.book1D(
          "L2TrigTauEtEffDenom", "L2 #tau p_{T} denominator;Ref #tau p_{T};Efficiency", ptbins_, 0, ptmax_);
      hL2TrigTauEtaEffNum_ =
          iBooker.book1D("L2TrigTauEtaEffNum", "L2 #tau #eta efficiency;Ref #tau #eta;entries", etabins_, -2.5, 2.5);
      hL2TrigTauEtaEffDenom_ = iBooker.book1D(
          "L2TrigTauEtaEffDenom", "L2 #tau #eta denominator;Ref #tau #eta;Efficiency", etabins_, -2.5, 2.5);
      hL2TrigTauPhiEffNum_ =
          iBooker.book1D("L2TrigTauPhiEffNum", "L2 #tau #phi efficiency;Ref #tau #phi;entries", phibins_, -3.2, 3.2);
      hL2TrigTauPhiEffDenom_ = iBooker.book1D(
          "L2TrigTauPhiEffDenom", "L2 #tau #phi denominator;Ref #tau #phi;Efficiency", phibins_, -3.2, 3.2);
      hL2TrigTauHighEtEffNum_ = iBooker.book1D("L2TrigTauHighEtEffNum",
                                               "L2 #tau p_{T} efficiency (high p_{T});Ref #tau p_{T};entries",
                                               ptbins_,
                                               0,
                                               highptmax_);
      hL2TrigTauHighEtEffDenom_ = iBooker.book1D("L2TrigTauHighEtEffDenom",
                                                 "L2 #tau p_{T} denominator (high p_{T});Ref #tau p_{T};Efficiency",
                                                 ptbins_,
                                                 0,
                                                 highptmax_);
    }

    if (hltPath_.hasL3Taus()) {
      hL3TrigTauEtEffNum_ =
          iBooker.book1D("L3TrigTauEtEffNum", "L3 #tau p_{T} efficiency;Ref #tau p_{T};entries", ptbins_, 0, ptmax_);
      hL3TrigTauEtEffDenom_ = iBooker.book1D(
          "L3TrigTauEtEffDenom", "L3 #tau p_{T} denominator;Ref #tau p_{T};Efficiency", ptbins_, 0, ptmax_);
      hL3TrigTauEtaEffNum_ =
          iBooker.book1D("L3TrigTauEtaEffNum", "L3 #tau #eta efficiency;Ref #tau #eta;entries", etabins_, -2.5, 2.5);
      hL3TrigTauEtaEffDenom_ = iBooker.book1D(
          "L3TrigTauEtaEffDenom", "L3 #tau #eta denominator;Ref #tau #eta;Efficiency", etabins_, -2.5, 2.5);
      hL3TrigTauPhiEffNum_ =
          iBooker.book1D("L3TrigTauPhiEffNum", "L3 #tau #phi efficiency;Ref #tau #phi;entries", phibins_, -3.2, 3.2);
      hL3TrigTauPhiEffDenom_ = iBooker.book1D(
          "L3TrigTauPhiEffDenom", "L3 #tau #phi denominator;Ref #tau #phi;Efficiency", phibins_, -3.2, 3.2);
      hL3TrigTauHighEtEffNum_ = iBooker.book1D("L3TrigTauHighEtEffNum",
                                               "L3 #tau p_{T} efficiency (high p_{T});Ref #tau p_{T};entries",
                                               ptbins_,
                                               0,
                                               highptmax_);
      hL3TrigTauHighEtEffDenom_ = iBooker.book1D("L3TrigTauHighEtEffDenom",
                                                 "L3 #tau p_{T} denominator (high p_{T});Ref #tau p_{T};Efficiency",
                                                 ptbins_,
                                                 0,
                                                 highptmax_);
      hL3TrigTauEtaPhiEffNum_ = iBooker.book2D(
          "L3TrigTauEtaPhiEffNum", "L3 efficiency in eta-phi plane", etabins_, -2.5, 2.5, phibins_, -3.2, 3.2);
      hL3TrigTauEtaPhiEffDenom_ = iBooker.book2D(
          "L3TrigTauEtaPhiEffDenom", "L3 denominator in eta-phi plane", etabins_, -2.5, 2.5, phibins_, -3.2, 3.2);
      hL3TrigTauEtaPhiEffDenom_->setOption("COL");
    }

    if (hltPath_.hasL2Electrons()) {
      hL2TrigElectronEtEffNum_ = iBooker.book1D(
          "L2TrigElectronEtEffNum", "L2 electron p_{T} efficiency;Ref electron p_{T};entries", ptbins_, 0, ptmax_);
      hL2TrigElectronEtEffDenom_ = iBooker.book1D(
          "L2TrigElectronEtEffDenom", "L2 electron p_{T} denominator;Ref electron p_{T};Efficiency", ptbins_, 0, ptmax_);
      hL2TrigElectronEtaEffNum_ = iBooker.book1D(
          "L2TrigElectronEtaEffNum", "L2 electron #eta efficiency;Ref electron #eta;entries", etabins_, -2.5, 2.5);
      hL2TrigElectronEtaEffDenom_ = iBooker.book1D(
          "L2TrigElectronEtaEffDenom", "L2 electron #eta denominator;Ref electron #eta;Efficiency", etabins_, -2.5, 2.5);
      hL2TrigElectronPhiEffNum_ = iBooker.book1D(
          "L2TrigElectronPhiEffNum", "L2 electron #phi efficiency;Ref electron #phi;entries", phibins_, -3.2, 3.2);
      hL2TrigElectronPhiEffDenom_ = iBooker.book1D(
          "L2TrigElectronPhiEffDenom", "L2 electron #phi denominator;Ref electron #phi;Efficiency", phibins_, -3.2, 3.2);
    }

    if (hltPath_.hasL3Electrons()) {
      hL3TrigElectronEtEffNum_ = iBooker.book1D(
          "L3TrigElectronEtEffNum", "L3 electron p_{T} efficiency;Ref electron p_{T};entries", ptbins_, 0, ptmax_);
      hL3TrigElectronEtEffDenom_ = iBooker.book1D(
          "L3TrigElectronEtEffDenom", "L3 electron p_{T} denominator;Ref electron p_{T};Efficiency", ptbins_, 0, ptmax_);
      hL3TrigElectronEtaEffNum_ = iBooker.book1D(
          "L3TrigElectronEtaEffNum", "L3 electron #eta efficiency;Ref electron #eta;entries", etabins_, -2.5, 2.5);
      hL3TrigElectronEtaEffDenom_ = iBooker.book1D(
          "L3TrigElectronEtaEffDenom", "L3 electron #eta denominator;Ref electron #eta;Efficiency", etabins_, -2.5, 2.5);
      hL3TrigElectronPhiEffNum_ = iBooker.book1D(
          "L3TrigElectronPhiEffNum", "L3 electron #phi efficiency;Ref electron #phi;entries", phibins_, -3.2, 3.2);
      hL3TrigElectronPhiEffDenom_ = iBooker.book1D(
          "L3TrigElectronPhiEffDenom", "L3 electron #phi denominator;Ref electron #phi;Efficiency", phibins_, -3.2, 3.2);
    }

    if (hltPath_.hasL2Muons()) {
      hL2TrigMuonEtEffNum_ =
          iBooker.book1D("L2TrigMuonEtEffNum", "L2 muon p_{T} efficiency;Ref muon p_{T};entries", ptbins_, 0, ptmax_);
      hL2TrigMuonEtEffDenom_ = iBooker.book1D(
          "L2TrigMuonEtEffDenom", "L2 muon p_{T} denominator;Ref muon p_{T};Efficiency", ptbins_, 0, ptmax_);
      hL2TrigMuonEtaEffNum_ =
          iBooker.book1D("L2TrigMuonEtaEffNum", "L2 muon #eta efficiency;Ref muon #eta;entries", etabins_, -2.5, 2.5);
      hL2TrigMuonEtaEffDenom_ = iBooker.book1D(
          "L2TrigMuonEtaEffDenom", "L2 muon #eta denominator;Ref muon #eta;Efficiency", etabins_, -2.5, 2.5);
      hL2TrigMuonPhiEffNum_ =
          iBooker.book1D("L2TrigMuonPhiEffNum", "L2 muon #phi efficiency;Ref muon #phi;entries", phibins_, -3.2, 3.2);
      hL2TrigMuonPhiEffDenom_ = iBooker.book1D(
          "L2TrigMuonPhiEffDenom", "L2 muon #phi denominator;Ref muon #phi;Efficiency", phibins_, -3.2, 3.2);
    }

    if (hltPath_.hasL3Muons()) {
      hL3TrigMuonEtEffNum_ =
          iBooker.book1D("L3TrigMuonEtEffNum", "L3 muon p_{T} efficiency;Ref muon p_{T};entries", ptbins_, 0, ptmax_);
      hL3TrigMuonEtEffDenom_ = iBooker.book1D(
          "L3TrigMuonEtEffDenom", "L3 muon p_{T} denominator;Ref muon p_{T};Efficiency", ptbins_, 0, ptmax_);
      hL3TrigMuonEtaEffNum_ =
          iBooker.book1D("L3TrigMuonEtaEffNum", "L3 muon #eta efficiency;Ref muon #eta;entries", etabins_, -2.5, 2.5);
      hL3TrigMuonEtaEffDenom_ = iBooker.book1D(
          "L3TrigMuonEtaEffDenom", "L3 muon #eta denominator;Ref muon #eta;Efficiency", etabins_, -2.5, 2.5);
      hL3TrigMuonPhiEffNum_ =
          iBooker.book1D("L3TrigMuonPhiEffNum", "L3 muon #phi efficiency;Ref muon #phi;entries", phibins_, -3.2, 3.2);
      hL3TrigMuonPhiEffDenom_ = iBooker.book1D(
          "L3TrigMuonPhiEffDenom", "L3 muon #phi denominator;Ref muon #phi;Efficiency", phibins_, -3.2, 3.2);
    }

    if (hltPath_.hasL2CaloMET()) {
      hL2TrigMETEtEffNum_ =
          iBooker.book1D("L2TrigMETEtEffNum", "L2 MET efficiency;Ref MET;entries", ptbins_, 0, ptmax_);
      hL2TrigMETEtEffDenom_ =
          iBooker.book1D("L2TrigMETEtEffDenom", "L2 MET denominator;Ref MET;Efficiency", ptbins_, 0, ptmax_);
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
        this->hMass_ = iBooker.book1D(
            "ReferenceMass", "Transverse mass of reference " + name + ";Reference transverse mass;entries", 100, 0, 500);
      } else {
        this->hMass_ = iBooker.book1D(
            "ReferenceMass", "Invariant mass of reference " + name + ";Reference invariant mass;entries", 100, 0, 500);
      }
    };
    LogDebug("HLTTauDQMOffline") << "Path " << hltPath_.getPathName() << " number of taus " << ntaus << " electrons "
                                 << neles << " muons " << nmus;
    if (ntaus > 0) {
      hTrigTauEt_ = iBooker.book1D("TrigTauEt", "Triggered #tau p_{T};#tau p_{T};entries", ptbins_, 0, ptmax_);
      hTrigTauEta_ = iBooker.book1D("TrigTauEta", "Triggered #tau #eta;#tau #eta;entries", etabins_, -2.5, 2.5);
      hTrigTauPhi_ = iBooker.book1D("TrigTauPhi", "Triggered #tau #phi;#tau #phi;entries", phibins_, -3.2, 3.2);
    }
    if (neles > 0) {
      hTrigElectronEt_ =
          iBooker.book1D("TrigElectronEt", "Triggered electron p_{T};electron p_{T};entries", ptbins_, 0, ptmax_);
      hTrigElectronEta_ =
          iBooker.book1D("TrigElectronEta", "Triggered electron #eta;electron #eta;entries", etabins_, -2.5, 2.5);
      hTrigElectronPhi_ =
          iBooker.book1D("TrigElectronPhi", "Triggered electron #phi;electron #phi;entries", phibins_, -3.2, 3.2);
    }
    if (nmus > 0) {
      hTrigMuonEt_ = iBooker.book1D("TrigMuonEt", "Triggered muon p_{T};muon p_{T};entries", ptbins_, 0, ptmax_);
      hTrigMuonEta_ = iBooker.book1D("TrigMuonEta", "Triggered muon #eta;muon #eta;entries", etabins_, -2.5, 2.5);
      hTrigMuonPhi_ = iBooker.book1D("TrigMuonPhi", "Triggered muon #phi;muon #phi;entries", phibins_, -3.2, 3.2);
    }
    if (nmet > 0) {
      hTrigMETEt_ = iBooker.book1D("TrigMETEt", "Triggered MET E_{T};MET E_{T};entries", ptbins_, 0, ptmax_);
      hTrigMETPhi_ = iBooker.book1D("TrigMETPhi", "Triggered MET #phi;MET #phi;entries", phibins_, -3.2, 3.2);
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

      hAcceptedEvents_->Fill(i + 0.5);
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
      hAcceptedEvents_->Fill(i + 0.5);
    }
  }

  // Efficiency plots
  if (doRefAnalysis_ && lastMatchedFilter >= 0) {
    // L2 taus
    if (hltPath_.hasL2Taus()) {
      // Denominators
      if (static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastFilterBeforeL2TauIndex()) {
        for (const LV& tau : refCollection.taus) {
          hL2TrigTauEtEffDenom_->Fill(tau.pt());
          hL2TrigTauHighEtEffDenom_->Fill(tau.pt());
          hL2TrigTauEtaEffDenom_->Fill(tau.eta());
          hL2TrigTauPhiEffDenom_->Fill(tau.phi());
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
            hL2TrigTauEtEffNum_->Fill(tau.pt());
            hL2TrigTauHighEtEffNum_->Fill(tau.pt());
            hL2TrigTauEtaEffNum_->Fill(tau.eta());
            hL2TrigTauPhiEffNum_->Fill(tau.phi());
          }
        }
      }
    }

    // L3 taus
    if (hltPath_.hasL3Taus()) {
      // Denominators
      if (static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastFilterBeforeL3TauIndex()) {
        for (const LV& tau : refCollection.taus) {
          hL3TrigTauEtEffDenom_->Fill(tau.pt());
          hL3TrigTauHighEtEffDenom_->Fill(tau.pt());
          hL3TrigTauEtaEffDenom_->Fill(tau.eta());
          hL3TrigTauPhiEffDenom_->Fill(tau.phi());
          hL3TrigTauEtaPhiEffDenom_->Fill(tau.eta(), tau.phi());
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
            hL3TrigTauEtEffNum_->Fill(tau.pt());
            hL3TrigTauHighEtEffNum_->Fill(tau.pt());
            hL3TrigTauEtaEffNum_->Fill(tau.eta());
            hL3TrigTauPhiEffNum_->Fill(tau.phi());
            hL3TrigTauEtaPhiEffNum_->Fill(tau.eta(), tau.phi());
          }
        }
      }
    }

    // L2 Electrons
    if (hltPath_.hasL2Electrons()) {
      // Denominators
      if (static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastFilterBeforeL2ElectronIndex()) {
        for (const LV& electron : refCollection.electrons) {
          hL2TrigElectronEtEffDenom_->Fill(electron.pt());
          hL2TrigElectronEtaEffDenom_->Fill(electron.eta());
          hL2TrigElectronPhiEffDenom_->Fill(electron.phi());
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
            hL2TrigElectronEtEffNum_->Fill(electron.pt());
            hL2TrigElectronEtaEffNum_->Fill(electron.eta());
            hL2TrigElectronPhiEffNum_->Fill(electron.phi());
          }
        }
      }
    }

    // L3 electron
    if (hltPath_.hasL3Electrons()) {
      // Denominators
      if (static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastFilterBeforeL3ElectronIndex()) {
        for (const LV& electron : refCollection.electrons) {
          hL3TrigElectronEtEffDenom_->Fill(electron.pt());
          hL3TrigElectronEtaEffDenom_->Fill(electron.eta());
          hL3TrigElectronPhiEffDenom_->Fill(electron.phi());
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
            hL3TrigElectronEtEffNum_->Fill(electron.pt());
            hL3TrigElectronEtaEffNum_->Fill(electron.eta());
            hL3TrigElectronPhiEffNum_->Fill(electron.phi());
          }
        }
      }
    }

    // L2 Muons
    if (hltPath_.hasL2Muons()) {
      // Denominators
      if (static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastFilterBeforeL2MuonIndex()) {
        for (const LV& muon : refCollection.muons) {
          hL2TrigMuonEtEffDenom_->Fill(muon.pt());
          hL2TrigMuonEtaEffDenom_->Fill(muon.eta());
          hL2TrigMuonPhiEffDenom_->Fill(muon.phi());
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
            hL2TrigMuonEtEffNum_->Fill(muon.pt());
            hL2TrigMuonEtaEffNum_->Fill(muon.eta());
            hL2TrigMuonPhiEffNum_->Fill(muon.phi());
          }
        }
      }
    }

    // L3 muon
    if (hltPath_.hasL3Muons()) {
      // Denominators
      if (static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastFilterBeforeL3MuonIndex()) {
        for (const LV& muon : refCollection.muons) {
          hL3TrigMuonEtEffDenom_->Fill(muon.pt());
          hL3TrigMuonEtaEffDenom_->Fill(muon.eta());
          hL3TrigMuonPhiEffDenom_->Fill(muon.phi());
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
            hL3TrigMuonEtEffNum_->Fill(muon.pt());
            hL3TrigMuonEtaEffNum_->Fill(muon.eta());
            hL3TrigMuonPhiEffNum_->Fill(muon.phi());
          }
        }
      }
    }

    // L2 CaloMET
    if (hltPath_.hasL2CaloMET()) {
      // Denominators
      if (static_cast<size_t>(firstMatchedMETFilter) >= hltPath_.getFirstFilterBeforeL2CaloMETIndex()) {
        hL2TrigMETEtEffDenom_->Fill(refCollection.met[0].pt());
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
          hL2TrigMETEtEffNum_->Fill(matchedOfflineObjs.met[0].pt());
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
            hMass_->Fill((matchedOfflineObjs.taus[0] + matchedOfflineObjs.taus[1]).M());
          }
          // Electron+tau
          else if (ntaus == 1 && hltPath_.getFilterNElectrons(lastPassedFilter) == 1) {
            hMass_->Fill((matchedOfflineObjs.taus[0] + matchedOfflineObjs.electrons[0]).M());
          }
          // Muon+tau
          else if (ntaus == 1 && hltPath_.getFilterNMuons(lastPassedFilter) == 1) {
            hMass_->Fill((matchedOfflineObjs.taus[0] + matchedOfflineObjs.muons[0]).M());
          }
          // Tau+MET
          if (hltPath_.getFilterNTaus(lastPassedFilter) == 1 && hltPath_.getFilterMET(lastMatchedMETFilter) == 1) {
            double taupt = matchedOfflineObjs.taus[0].Pt();
            double tauphi = matchedOfflineObjs.taus[0].Phi();
            double met = matchedOfflineObjs.met[0].Pt();
            double metphi = matchedOfflineObjs.met[0].Phi();
            double mT = sqrt(2 * taupt * met * (1 - cos(tauphi - metphi)));

            hMass_->Fill(mT);
          }
        }
      }

      // Triggered object kinematics
      for (const HLTTauDQMPath::Object& obj : triggerObjs) {
        if (obj.id == trigger::TriggerTau) {
          hTrigTauEt_->Fill(obj.object.pt());
          hTrigTauEta_->Fill(obj.object.eta());
          hTrigTauPhi_->Fill(obj.object.phi());
        }
        if (obj.id == trigger::TriggerElectron || obj.id == trigger::TriggerPhoton) {
          hTrigElectronEt_->Fill(obj.object.pt());
          hTrigElectronEta_->Fill(obj.object.eta());
          hTrigElectronPhi_->Fill(obj.object.phi());
        }
        if (obj.id == trigger::TriggerMuon) {
          hTrigMuonEt_->Fill(obj.object.pt());
          hTrigMuonEta_->Fill(obj.object.eta());
          hTrigMuonPhi_->Fill(obj.object.phi());
        }
        if (obj.id == trigger::TriggerMET) {
          hTrigMETEt_->Fill(obj.object.pt());
          hTrigMETPhi_->Fill(obj.object.phi());
        }
      }
    }
  }
}

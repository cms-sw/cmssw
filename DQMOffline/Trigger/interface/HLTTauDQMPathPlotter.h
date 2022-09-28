// -*- c++ -*-
#ifndef DQMOffline_Trigger_HLTTauDQMPathPlotter_h
#define DQMOffline_Trigger_HLTTauDQMPathPlotter_h

#include "DQMOffline/Trigger/interface/HLTTauDQMPlotter.h"
#include "DQMOffline/Trigger/interface/HLTTauDQMPath.h"
#include "DQMOffline/Trigger/interface/HistoWrapper.h"

namespace edm {
  class Event;
  class EventSetup;
  class TriggerResults;
}  // namespace edm

namespace trigger {
  class TriggerEvent;
}

class HLTConfigProvider;

class HLTTauDQMPathPlotter : private HLTTauDQMPlotter {
public:
  HLTTauDQMPathPlotter(const std::string &pathName,
                       const HLTConfigProvider &HLTCP,
                       bool doRefAnalysis,
                       const std::string &dqmBaseFolder,
                       const std::string &hltProcess,
                       int ptbins,
                       int etabins,
                       int phibins,
                       double ptmax,
                       double highptmax,
                       double l1MatchDr,
                       double hltMatchDr);
  ~HLTTauDQMPathPlotter();

  using HLTTauDQMPlotter::isValid;

  void bookHistograms(HistoWrapper &iWrapper, DQMStore::IBooker &iBooker);

  template <class T>
  void analyze(const edm::TriggerResults &triggerResults,
               const T &triggerEvent,
               const HLTTauDQMOfflineObjects &refCollection) {
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

        if (hAcceptedEvents_)
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
        if (hAcceptedEvents_)
          hAcceptedEvents_->Fill(i + 0.5);
      }
    }

    // Efficiency plots
    if (doRefAnalysis_ && lastMatchedFilter >= 0) {
      // L2 taus
      if (hltPath_.hasL2Taus()) {
        // Denominators
        if (static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastFilterBeforeL2TauIndex()) {
          for (const LV &tau : refCollection.taus) {
            if (hL2TrigTauEtEffDenom_)
              hL2TrigTauEtEffDenom_->Fill(tau.pt());
            if (hL2TrigTauHighEtEffDenom_)
              hL2TrigTauHighEtEffDenom_->Fill(tau.pt());
            if (hL2TrigTauEtaEffDenom_)
              hL2TrigTauEtaEffDenom_->Fill(tau.eta());
            if (hL2TrigTauPhiEffDenom_)
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
            for (const LV &tau : matchedOfflineObjs.taus) {
              if (hL2TrigTauEtEffNum_)
                hL2TrigTauEtEffNum_->Fill(tau.pt());
              if (hL2TrigTauHighEtEffNum_)
                hL2TrigTauHighEtEffNum_->Fill(tau.pt());
              if (hL2TrigTauEtaEffNum_)
                hL2TrigTauEtaEffNum_->Fill(tau.eta());
              if (hL2TrigTauPhiEffNum_)
                hL2TrigTauPhiEffNum_->Fill(tau.phi());
            }
          }
        }
      }

      // L3 taus
      if (hltPath_.hasL3Taus()) {
        // Denominators
        if (static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastFilterBeforeL3TauIndex()) {
          for (const LV &tau : refCollection.taus) {
            if (hL3TrigTauEtEffDenom_)
              hL3TrigTauEtEffDenom_->Fill(tau.pt());
            if (hL3TrigTauHighEtEffDenom_)
              hL3TrigTauHighEtEffDenom_->Fill(tau.pt());
            if (hL3TrigTauEtaEffDenom_)
              hL3TrigTauEtaEffDenom_->Fill(tau.eta());
            if (hL3TrigTauPhiEffDenom_)
              hL3TrigTauPhiEffDenom_->Fill(tau.phi());
            if (hL3TrigTauEtaPhiEffDenom_)
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
            for (const LV &tau : matchedOfflineObjs.taus) {
              if (hL3TrigTauEtEffNum_)
                hL3TrigTauEtEffNum_->Fill(tau.pt());
              if (hL3TrigTauHighEtEffNum_)
                hL3TrigTauHighEtEffNum_->Fill(tau.pt());
              if (hL3TrigTauEtaEffNum_)
                hL3TrigTauEtaEffNum_->Fill(tau.eta());
              if (hL3TrigTauPhiEffNum_)
                hL3TrigTauPhiEffNum_->Fill(tau.phi());
              if (hL3TrigTauEtaPhiEffNum_)
                hL3TrigTauEtaPhiEffNum_->Fill(tau.eta(), tau.phi());
            }
          }
        }
      }

      // L2 Electrons
      if (hltPath_.hasL2Electrons()) {
        // Denominators
        if (static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastFilterBeforeL2ElectronIndex()) {
          for (const LV &electron : refCollection.electrons) {
            if (hL2TrigElectronEtEffDenom_)
              hL2TrigElectronEtEffDenom_->Fill(electron.pt());
            if (hL2TrigElectronEtaEffDenom_)
              hL2TrigElectronEtaEffDenom_->Fill(electron.eta());
            if (hL2TrigElectronPhiEffDenom_)
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
            for (const LV &electron : matchedOfflineObjs.electrons) {
              if (hL2TrigElectronEtEffNum_)
                hL2TrigElectronEtEffNum_->Fill(electron.pt());
              if (hL2TrigElectronEtaEffNum_)
                hL2TrigElectronEtaEffNum_->Fill(electron.eta());
              if (hL2TrigElectronPhiEffNum_)
                hL2TrigElectronPhiEffNum_->Fill(electron.phi());
            }
          }
        }
      }

      // L3 electron
      if (hltPath_.hasL3Electrons()) {
        // Denominators
        if (static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastFilterBeforeL3ElectronIndex()) {
          for (const LV &electron : refCollection.electrons) {
            if (hL3TrigElectronEtEffDenom_)
              hL3TrigElectronEtEffDenom_->Fill(electron.pt());
            if (hL3TrigElectronEtaEffDenom_)
              hL3TrigElectronEtaEffDenom_->Fill(electron.eta());
            if (hL3TrigElectronPhiEffDenom_)
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
            for (const LV &electron : matchedOfflineObjs.electrons) {
              if (hL3TrigElectronEtEffNum_)
                hL3TrigElectronEtEffNum_->Fill(electron.pt());
              if (hL3TrigElectronEtaEffNum_)
                hL3TrigElectronEtaEffNum_->Fill(electron.eta());
              if (hL3TrigElectronPhiEffNum_)
                hL3TrigElectronPhiEffNum_->Fill(electron.phi());
            }
          }
        }
      }

      // L2 Muons
      if (hltPath_.hasL2Muons()) {
        // Denominators
        if (static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastFilterBeforeL2MuonIndex()) {
          for (const LV &muon : refCollection.muons) {
            if (hL2TrigMuonEtEffDenom_)
              hL2TrigMuonEtEffDenom_->Fill(muon.pt());
            if (hL2TrigMuonEtaEffDenom_)
              hL2TrigMuonEtaEffDenom_->Fill(muon.eta());
            if (hL2TrigMuonPhiEffDenom_)
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
            for (const LV &muon : matchedOfflineObjs.muons) {
              if (hL2TrigMuonEtEffNum_)
                hL2TrigMuonEtEffNum_->Fill(muon.pt());
              if (hL2TrigMuonEtaEffNum_)
                hL2TrigMuonEtaEffNum_->Fill(muon.eta());
              if (hL2TrigMuonPhiEffNum_)
                hL2TrigMuonPhiEffNum_->Fill(muon.phi());
            }
          }
        }
      }

      // L3 muon
      if (hltPath_.hasL3Muons()) {
        // Denominators
        if (static_cast<size_t>(lastMatchedFilter) >= hltPath_.getLastFilterBeforeL3MuonIndex()) {
          for (const LV &muon : refCollection.muons) {
            if (hL3TrigMuonEtEffDenom_)
              hL3TrigMuonEtEffDenom_->Fill(muon.pt());
            if (hL3TrigMuonEtaEffDenom_)
              hL3TrigMuonEtaEffDenom_->Fill(muon.eta());
            if (hL3TrigMuonPhiEffDenom_)
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
            for (const LV &muon : matchedOfflineObjs.muons) {
              if (hL3TrigMuonEtEffNum_)
                hL3TrigMuonEtEffNum_->Fill(muon.pt());
              if (hL3TrigMuonEtaEffNum_)
                hL3TrigMuonEtaEffNum_->Fill(muon.eta());
              if (hL3TrigMuonPhiEffNum_)
                hL3TrigMuonPhiEffNum_->Fill(muon.phi());
            }
          }
        }
      }

      // L2 CaloMET
      if (hltPath_.hasL2CaloMET()) {
        // Denominators
        if (static_cast<size_t>(firstMatchedMETFilter) >= hltPath_.getFirstFilterBeforeL2CaloMETIndex()) {
          if (hL2TrigMETEtEffDenom_)
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
            if (hL2TrigMETEtEffNum_)
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
              if (hMass_)
                hMass_->Fill((matchedOfflineObjs.taus[0] + matchedOfflineObjs.taus[1]).M());
            }
            // Electron+tau
            else if (ntaus == 1 && hltPath_.getFilterNElectrons(lastPassedFilter) == 1) {
              if (hMass_)
                hMass_->Fill((matchedOfflineObjs.taus[0] + matchedOfflineObjs.electrons[0]).M());
            }
            // Muon+tau
            else if (ntaus == 1 && hltPath_.getFilterNMuons(lastPassedFilter) == 1) {
              if (hMass_)
                hMass_->Fill((matchedOfflineObjs.taus[0] + matchedOfflineObjs.muons[0]).M());
            }
            // Tau+MET
            if (hltPath_.getFilterNTaus(lastPassedFilter) == 1 && hltPath_.getFilterMET(lastMatchedMETFilter) == 1) {
              double taupt = matchedOfflineObjs.taus[0].Pt();
              double tauphi = matchedOfflineObjs.taus[0].Phi();
              double met = matchedOfflineObjs.met[0].Pt();
              double metphi = matchedOfflineObjs.met[0].Phi();
              double mT = sqrt(2 * taupt * met * (1 - cos(tauphi - metphi)));

              if (hMass_)
                hMass_->Fill(mT);
            }
          }
        }

        // Triggered object kinematics
        for (const HLTTauDQMPath::Object &obj : triggerObjs) {
          if (obj.id == trigger::TriggerTau) {
            if (hTrigTauEt_)
              hTrigTauEt_->Fill(obj.object.pt());
            if (hTrigTauEta_)
              hTrigTauEta_->Fill(obj.object.eta());
            if (hTrigTauPhi_)
              hTrigTauPhi_->Fill(obj.object.phi());
          }
          if (obj.id == trigger::TriggerElectron || obj.id == trigger::TriggerPhoton) {
            if (hTrigElectronEt_)
              hTrigElectronEt_->Fill(obj.object.pt());
            if (hTrigElectronEta_)
              hTrigElectronEta_->Fill(obj.object.eta());
            if (hTrigElectronPhi_)
              hTrigElectronPhi_->Fill(obj.object.phi());
          }
          if (obj.id == trigger::TriggerMuon) {
            if (hTrigMuonEt_)
              hTrigMuonEt_->Fill(obj.object.pt());
            if (hTrigMuonEta_)
              hTrigMuonEta_->Fill(obj.object.eta());
            if (hTrigMuonPhi_)
              hTrigMuonPhi_->Fill(obj.object.phi());
          }
          if (obj.id == trigger::TriggerMET) {
            if (hTrigMETEt_)
              hTrigMETEt_->Fill(obj.object.pt());
            if (hTrigMETPhi_)
              hTrigMETPhi_->Fill(obj.object.phi());
          }
        }
      }
    }
  }

  const HLTTauDQMPath *getPathObject() const { return &hltPath_; }

  typedef std::tuple<std::string, size_t> FilterIndex;

private:
  const int ptbins_;
  const int etabins_;
  const int phibins_;
  const double ptmax_;
  const double highptmax_;
  const double l1MatchDr_;
  const double hltMatchDr_;
  const bool doRefAnalysis_;

  HLTTauDQMPath hltPath_;

  MonitorElement *hAcceptedEvents_;
  MonitorElement *hTrigTauEt_;
  MonitorElement *hTrigTauEta_;
  MonitorElement *hTrigTauPhi_;
  MonitorElement *hTrigMuonEt_;
  MonitorElement *hTrigMuonEta_;
  MonitorElement *hTrigMuonPhi_;
  MonitorElement *hTrigElectronEt_;
  MonitorElement *hTrigElectronEta_;
  MonitorElement *hTrigElectronPhi_;
  MonitorElement *hTrigMETEt_;
  MonitorElement *hTrigMETPhi_;
  MonitorElement *hMass_;

  MonitorElement *hL2TrigTauEtEffNum_;
  MonitorElement *hL2TrigTauEtEffDenom_;
  MonitorElement *hL2TrigTauHighEtEffNum_;
  MonitorElement *hL2TrigTauHighEtEffDenom_;
  MonitorElement *hL2TrigTauEtaEffNum_;
  MonitorElement *hL2TrigTauEtaEffDenom_;
  MonitorElement *hL2TrigTauPhiEffNum_;
  MonitorElement *hL2TrigTauPhiEffDenom_;

  MonitorElement *hL3TrigTauEtEffNum_;
  MonitorElement *hL3TrigTauEtEffDenom_;
  MonitorElement *hL3TrigTauHighEtEffNum_;
  MonitorElement *hL3TrigTauHighEtEffDenom_;
  MonitorElement *hL3TrigTauEtaEffNum_;
  MonitorElement *hL3TrigTauEtaEffDenom_;
  MonitorElement *hL3TrigTauPhiEffNum_;
  MonitorElement *hL3TrigTauPhiEffDenom_;
  MonitorElement *hL3TrigTauEtaPhiEffNum_;
  MonitorElement *hL3TrigTauEtaPhiEffDenom_;

  MonitorElement *hL2TrigElectronEtEffNum_;
  MonitorElement *hL2TrigElectronEtEffDenom_;
  MonitorElement *hL2TrigElectronEtaEffNum_;
  MonitorElement *hL2TrigElectronEtaEffDenom_;
  MonitorElement *hL2TrigElectronPhiEffNum_;
  MonitorElement *hL2TrigElectronPhiEffDenom_;

  MonitorElement *hL3TrigElectronEtEffNum_;
  MonitorElement *hL3TrigElectronEtEffDenom_;
  MonitorElement *hL3TrigElectronEtaEffNum_;
  MonitorElement *hL3TrigElectronEtaEffDenom_;
  MonitorElement *hL3TrigElectronPhiEffNum_;
  MonitorElement *hL3TrigElectronPhiEffDenom_;

  MonitorElement *hL2TrigMuonEtEffNum_;
  MonitorElement *hL2TrigMuonEtEffDenom_;
  MonitorElement *hL2TrigMuonEtaEffNum_;
  MonitorElement *hL2TrigMuonEtaEffDenom_;
  MonitorElement *hL2TrigMuonPhiEffNum_;
  MonitorElement *hL2TrigMuonPhiEffDenom_;

  MonitorElement *hL3TrigMuonEtEffNum_;
  MonitorElement *hL3TrigMuonEtEffDenom_;
  MonitorElement *hL3TrigMuonEtaEffNum_;
  MonitorElement *hL3TrigMuonEtaEffDenom_;
  MonitorElement *hL3TrigMuonPhiEffNum_;
  MonitorElement *hL3TrigMuonPhiEffDenom_;

  MonitorElement *hL2TrigMETEtEffNum_;
  MonitorElement *hL2TrigMETEtEffDenom_;
};

#endif

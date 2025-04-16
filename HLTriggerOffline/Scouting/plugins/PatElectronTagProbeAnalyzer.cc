// system includes
#include <string>
#include <vector>

#include "Math/Vector4D.h"
#include "Math/VectorUtil.h"

// user include files
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/PackedTriggerPrescales.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "DataFormats/Scouting/interface/Run3ScoutingElectron.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionData.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionEvaluator.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionParser.h"
#include "L1Trigger/L1TGlobal/interface/L1TGlobalUtil.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"


#include "ScoutingDQMUtils.h"

/////////////////////////
//  Class declaration  //
/////////////////////////

struct kProbeFilterHistos{
  dqm::reco::MonitorElement* hPt_Barrel_passBaseDST;
  dqm::reco::MonitorElement* hPt_Endcap_passBaseDST;
  dqm::reco::MonitorElement* hEta_passBaseDST;
  std::vector<dqm::reco::MonitorElement*> hPt_Barrel_passDST;
  std::vector<dqm::reco::MonitorElement*> hPt_Endcap_passDST;
  std::vector<dqm::reco::MonitorElement*> hEta_passDST;
  std::vector<dqm::reco::MonitorElement*> hPt_Barrel_fireTrigObj;
  std::vector<dqm::reco::MonitorElement*> hPt_Endcap_fireTrigObj;
  std::vector<dqm::reco::MonitorElement*> hEta_fireTrigObj;
  std::vector<dqm::reco::MonitorElement*> hPt_Barrel_fireL1;
  std::vector<dqm::reco::MonitorElement*> hPt_Endcap_fireL1;
  std::vector<dqm::reco::MonitorElement*> hEta_fireL1;
};

struct kProbeKinematicHistos {
  dqm::reco::MonitorElement* hPt_Barrel;
  dqm::reco::MonitorElement* hPt_Endcap;
  dqm::reco::MonitorElement* hEta;
  dqm::reco::MonitorElement* hEtavPhi;
  dqm::reco::MonitorElement* hPhi;
  dqm::reco::MonitorElement* hHoverE_Barrel;
  dqm::reco::MonitorElement* hHoverE_Endcap;
  dqm::reco::MonitorElement* hOoEMOoP_Barrel;
  dqm::reco::MonitorElement* hOoEMOoP_Endcap;
  dqm::reco::MonitorElement* hdPhiIn_Barrel;
  dqm::reco::MonitorElement* hdPhiIn_Endcap;
  dqm::reco::MonitorElement* hdEtaIn_Barrel;
  dqm::reco::MonitorElement* hdEtaIn_Endcap;
  dqm::reco::MonitorElement* hSigmaIetaIeta_Barrel;
  dqm::reco::MonitorElement* hSigmaIetaIeta_Endcap;
  dqm::reco::MonitorElement* hMissingHits_Barrel;
  dqm::reco::MonitorElement* hMissingHits_Endcap;
  dqm::reco::MonitorElement* hTrackfbrem_Barrel;
  dqm::reco::MonitorElement* hTrackfbrem_Endcap;
  dqm::reco::MonitorElement* hTrack_pt_Barrel;
  dqm::reco::MonitorElement* hTrack_pt_Endcap;
  dqm::reco::MonitorElement* hTrack_pMode_Barrel;
  dqm::reco::MonitorElement* hTrack_pMode_Endcap;
  dqm::reco::MonitorElement* hTrack_etaMode_Barrel;
  dqm::reco::MonitorElement* hTrack_etaMode_Endcap;
  dqm::reco::MonitorElement* hTrack_phiMode_Barrel;
  dqm::reco::MonitorElement* hTrack_phiMode_Endcap;
  dqm::reco::MonitorElement* hTrack_qoverpModeError_Barrel;
  dqm::reco::MonitorElement* hTrack_qoverpModeError_Endcap;
  dqm::reco::MonitorElement* hRelEcalIsolation_Barrel;
  dqm::reco::MonitorElement* hRelEcalIsolation_Endcap;
  dqm::reco::MonitorElement* hRelHcalIsolation_Barrel;
  dqm::reco::MonitorElement* hRelHcalIsolation_Endcap;
  dqm::reco::MonitorElement* hRelTrackIsolation_Barrel;
  dqm::reco::MonitorElement* hRelTrackIsolation_Endcap;
  dqm::reco::MonitorElement* hPtvsInvMass_Barrel;
  dqm::reco::MonitorElement* hPtvsInvMass_Endcap;
  dqm::reco::MonitorElement* hEtavsInvMass;
  dqm::reco::MonitorElement* hInvMass;
  kProbeFilterHistos leading_electron;
  kProbeFilterHistos subleading_electron;
};

struct kTagProbeResonance {
  kProbeKinematicHistos resonanceZ;
  kProbeKinematicHistos resonanceJ;
  kProbeKinematicHistos resonanceY;
  kProbeKinematicHistos resonanceAll;

};


struct kTagProbeHistos {
  kTagProbeResonance patElectron;
  kTagProbeResonance sctElectron;
};

class PatElectronTagProbeAnalyzer : public DQMGlobalEDAnalyzer<kTagProbeHistos> {
public:
  explicit PatElectronTagProbeAnalyzer(const edm::ParameterSet& conf);
  ~PatElectronTagProbeAnalyzer() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  // Constants
  static constexpr double TandP_Z_minMass = 80;      // Lower bound for Tag And Probe at the Z peak
  static constexpr double TandP_Z_maxMass = 100;     // Higher bound for Tag And Probe at the Z peak
  static constexpr double TandP_ups_minMass = 9.0;   // Lower bound for Tag And Probe at the Upsilon peak
  static constexpr double TandP_ups_maxMass = 12.6;  // Higher bound for Tag And Probe at the Upsilon peak
  static constexpr double TandP_jpsi_minMass = 2.8;  // Lower bound for Tag And Probe at the JPsi peak
  static constexpr double TandP_jpsi_maxMass = 3.8;  // Higher bound for Tag And Probe at the JPsi peak

private:
  void dqmAnalyze(const edm::Event& e, const edm::EventSetup& c, kTagProbeHistos const&) const override;

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, kTagProbeHistos&) const override;

  void bookHistograms_resonance(
      DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, kProbeKinematicHistos&, const std::string&) const;

  void fillHistograms_resonance(const kProbeKinematicHistos& histos,
                                const pat::Electron& el,
                                const std::vector<bool> trigger_result,
                                const trigger::TriggerObjectCollection* legObjects,
                                const trigger::TriggerObjectCollection* l1_legObjects,
                                const std::vector<bool> l1_result,
                                const bool pass_baseDST,
                                const float inv_mass,
                                const int pt_order) const;

  void fillHistograms_resonance_sct(const kProbeKinematicHistos& histos,
                                    const Run3ScoutingElectron& el,
                                    const int gsfTrackIndex,
                                    const std::vector<bool> trigger_result,
                                    const trigger::TriggerObjectCollection* legObjects,
                                    const trigger::TriggerObjectCollection* l1_legObjects,
                                    const std::vector<bool> l1_result,
                                    const bool pass_baseDST,
                                    const float inv_mass,
                                    const int pt_order) const;

  bool scoutingElectron_passHLT(const float el_eta,
                                const float el_phi,
                                const trigger::TriggerObjectCollection& legObjects) const;

  bool patElectron_passHLT(const pat::Electron& el, const trigger::TriggerObjectCollection& legObjects) const;

  // --------------------- member data  ----------------------
  const std::string outputInternalPath_;

  const std::vector<std::string> vBaseTriggerSelection_;
  const std::vector<std::string> vtriggerSelection_;
  const std::vector<std::string> filterToMatch_;
  const std::vector<std::string> l1filterToMatch_;
  const std::vector<unsigned int> l1filterIndex_;

  edm::EDGetToken algToken_;
  std::shared_ptr<l1t::L1TGlobalUtil> l1GtUtils_;
  std::vector<std::string> l1Seeds_;

  const edm::EDGetToken triggerResultsToken_;
  const edm::EDGetTokenT<pat::TriggerObjectStandAloneCollection> triggerObjects_;
  const edm::EDGetTokenT<edm::View<pat::Electron>> electronCollection_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingElectron>> scoutingElectronCollection_;
  const edm::EDGetTokenT<edm::ValueMap<bool>> eleIdMapTightToken_;
};

using namespace ROOT;

PatElectronTagProbeAnalyzer::PatElectronTagProbeAnalyzer(const edm::ParameterSet& iConfig)
    : outputInternalPath_(iConfig.getParameter<std::string>("OutputInternalPath")),
      vBaseTriggerSelection_{iConfig.getParameter<std::vector<std::string>>("BaseTriggerSelection")},
      vtriggerSelection_{iConfig.getParameter<std::vector<std::string>>("triggerSelection")},
      filterToMatch_{iConfig.getParameter<std::vector<std::string>>("finalfilterSelection")},
      l1filterToMatch_{iConfig.getParameter<std::vector<std::string>>("l1filterSelection")},
      l1filterIndex_{iConfig.getParameter<std::vector<unsigned int>>("l1filterSelectionIndex")},
      algToken_{consumes<BXVector<GlobalAlgBlk>>(iConfig.getParameter<edm::InputTag>("AlgInputTag"))},
      triggerResultsToken_(
          consumes<edm::TriggerResults>(iConfig.getParameter<edm::InputTag>("TriggerResultTag"))),
      triggerObjects_(
          consumes<pat::TriggerObjectStandAloneCollection>(iConfig.getParameter<edm::InputTag>("TriggerObjects"))),
      electronCollection_(
          consumes<edm::View<pat::Electron>>(iConfig.getParameter<edm::InputTag>("ElectronCollection"))),
      scoutingElectronCollection_(consumes<std::vector<Run3ScoutingElectron>>(
          iConfig.getParameter<edm::InputTag>("ScoutingElectronCollection"))),
      eleIdMapTightToken_(consumes<edm::ValueMap<bool>>(iConfig.getParameter<edm::InputTag>("eleIdMapTight"))) {

          l1GtUtils_ = std::make_shared<l1t::L1TGlobalUtil>(iConfig, consumesCollector(), l1t::UseEventSetupIn::RunAndEvent);
          l1Seeds_   = iConfig.getParameter<std::vector<std::string>>("L1Seeds");
}

void PatElectronTagProbeAnalyzer::dqmAnalyze(edm::Event const& iEvent,
                                             edm::EventSetup const& iSetup,
                                             kTagProbeHistos const& histos) const {

  // Check if pat electron collection exist.
  edm::Handle<edm::View<pat::Electron>> patEls;
  iEvent.getByToken(electronCollection_, patEls);
  if (patEls.failedToGet()) {
    edm::LogWarning("ScoutingMonitoring") << "pat::Electron collection not found.";
    return;
  }

  // Check if scouting electron collection exist.
  edm::Handle<std::vector<Run3ScoutingElectron>> sctEls;
  iEvent.getByToken(scoutingElectronCollection_, sctEls);
  if (sctEls.failedToGet()) {
    edm::LogWarning("ScoutingMonitoring") << "Run3ScoutingElectron collection not found.";
    return;
  }
 
  // Load pat Electron ID.
  edm::Handle<edm::ValueMap<bool>> tight_ele_id_decisions;
  iEvent.getByToken(eleIdMapTightToken_, tight_ele_id_decisions);

  edm::LogInfo("ScoutingMonitoring") << "Process pat::Electrons: " << patEls->size();
  edm::LogInfo("ScoutingMonitoring") << "Process Run3ScoutingElectrons: " << sctEls->size();

  // Trigger
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(triggerResultsToken_, triggerResults);

  // Trigger Object
  edm::Handle<pat::TriggerObjectStandAloneCollection> triggerObjects;
  iEvent.getByToken(triggerObjects_, triggerObjects);


  // Trigger result
  if (triggerResults.failedToGet()){
    edm::LogWarning("ScoutingEGammaCollectionMonitoring") << "Trgger Results not found.";
    return;
  }
  int nTriggers = triggerResults->size();
  std::vector<bool> vtrigger_result(vtriggerSelection_.size(), false);
  bool passBaseDST = false;
  const edm::TriggerNames& triggerNames = iEvent.triggerNames(*triggerResults);
  for (int i_Trig = 0; i_Trig < nTriggers; i_Trig++){
    if (triggerResults.product()->accept(i_Trig)){
        TString TrigPath = triggerNames.triggerName(i_Trig);
        for(unsigned int i_selectTrig = 0; i_selectTrig < vtriggerSelection_.size(); i_selectTrig++){
            if(TrigPath.Index(vtriggerSelection_.at(i_selectTrig)) >=0){
              vtrigger_result[i_selectTrig] = true;
            }
        }

        for (unsigned int i_BaseTrig = 0; i_BaseTrig < vBaseTriggerSelection_.size(); i_BaseTrig++){
           if(TrigPath.Index(vBaseTriggerSelection_.at(i_BaseTrig)) >=0){
             passBaseDST = true;
           }
        }
    }
  }

  // Trigger Object Matching
  size_t numberOfFilters = filterToMatch_.size();
  trigger::TriggerObjectCollection* legObjects = new trigger::TriggerObjectCollection[numberOfFilters];
  for (size_t iteFilter = 0; iteFilter < filterToMatch_.size(); iteFilter++) {
    std::string filterTag = filterToMatch_.at(iteFilter);
    for (pat::TriggerObjectStandAlone obj : *triggerObjects) {
      obj.unpackNamesAndLabels(iEvent, *triggerResults);
      if (obj.hasFilterLabel(filterTag)) {
        legObjects[iteFilter].push_back(obj);
      }
    }
  }

  // L1 Object Matching
  size_t numberOfl1Filters = l1filterToMatch_.size();
  trigger::TriggerObjectCollection* l1_legObjects = new trigger::TriggerObjectCollection[numberOfl1Filters];
  for (size_t iteFilter = 0; iteFilter < l1filterToMatch_.size(); iteFilter++) {
    std::string l1filterTag = l1filterToMatch_.at(iteFilter);
    for (pat::TriggerObjectStandAlone obj : *triggerObjects) {
      obj.unpackNamesAndLabels(iEvent, *triggerResults);
      if (obj.hasFilterLabel(l1filterTag)) {
        l1_legObjects[iteFilter].push_back(obj);
      }
    }
  }


  // L1Seeds
  l1GtUtils_->retrieveL1(iEvent, iSetup, algToken_);
  std::vector<bool> l1_result(l1Seeds_.size(), false);
  for (unsigned int i_l1seed = 0; i_l1seed < l1Seeds_.size(); i_l1seed++){
      const auto& l1seed(l1Seeds_.at(i_l1seed));
      bool l1htbit = false;
      double prescale = -1;
      l1GtUtils_->getFinalDecisionByName(l1seed, l1htbit);
      l1GtUtils_->getPrescaleByName(l1seed, prescale);
      if (l1htbit == 1) l1_result[i_l1seed] = true;
  }

  // sct electron gsfTrack finding
  std::vector<int> sctElectron_gsfTrackIndex;
  for (const auto& sct_el : *sctEls) {
    size_t gsfTrkIdx = 9999;
    bool foundGoodGsfTrkIdx = scoutingDQMUtils::scoutingElectronGsfTrackIdx(sct_el, gsfTrkIdx);
    if (foundGoodGsfTrkIdx)
      sctElectron_gsfTrackIndex.push_back(gsfTrkIdx);
    else
      sctElectron_gsfTrackIndex.push_back(-1);
  }

  // Pt ordered pat electron and sct electron collection

  std::vector<std::pair<size_t, pat::Electron>> indexed_patElectrons;
  for (size_t i = 0; i < patEls->size(); i++){
    indexed_patElectrons.emplace_back(i, (*patEls)[i]);
  }

  std::sort(indexed_patElectrons.begin(), indexed_patElectrons.end(),
       [](const auto& a, const auto& b){
           return a.second.pt() > b.second.pt();
       });

  std::vector<std::pair<size_t, Run3ScoutingElectron>> indexed_sctElectrons;
  for (size_t i = 0; i < sctEls->size(); i++){
    indexed_sctElectrons.emplace_back(i, (*sctEls)[i]);
  }
  std::sort(indexed_sctElectrons.begin(), indexed_sctElectrons.end(),
       [](const auto& a, const auto& b){
           return a.second.pt() > b.second.pt();
       });


  // Tag electron: pat ele collection

  for (size_t pat_local_index=0; pat_local_index < indexed_patElectrons.size(); pat_local_index++) {
    const auto pat_index = indexed_patElectrons[pat_local_index].first; 
    const auto pat_el    = indexed_patElectrons[pat_local_index].second;

    edm::Ref<edm::View<pat::Electron>> electronRef(patEls, pat_index);
    if (!((*tight_ele_id_decisions)[electronRef]))
      continue;
    ROOT::Math::PtEtaPhiMVector tag_pat_el(pat_el.pt(), pat_el.eta(), pat_el.phi(), pat_el.mass());

    // Probe electron: from pat electron
    int second_pat_pt_order = -1;
    for (size_t second_pat_local_index=0; second_pat_local_index < indexed_patElectrons.size(); second_pat_local_index++) {
      const auto second_pat_index = indexed_patElectrons[second_pat_local_index].first;
      const auto pat_el_second    = indexed_patElectrons[second_pat_local_index].second;
      edm::Ref<edm::View<pat::Electron>> second_electronRef(patEls, second_pat_index);
      if (!((*tight_ele_id_decisions)[second_electronRef]))
        continue;
      second_pat_pt_order += 1;
      if (pat_index == second_pat_index)
        continue;
      ROOT::Math::PtEtaPhiMVector probe_pat_el(
          pat_el_second.pt(), pat_el_second.eta(), pat_el_second.phi(), pat_el_second.mass());
      float invMass = (tag_pat_el + probe_pat_el).mass();

      // Z mass windows
      if ((TandP_Z_minMass < invMass) && (invMass < TandP_Z_maxMass)) {
        fillHistograms_resonance(histos.patElectron.resonanceZ, pat_el_second, vtrigger_result, legObjects, l1_legObjects, l1_result, passBaseDST, invMass, second_pat_pt_order);
        fillHistograms_resonance(histos.patElectron.resonanceAll, pat_el_second, vtrigger_result, legObjects, l1_legObjects, l1_result, passBaseDST, invMass, second_pat_pt_order);
      }

      // jpsi mass windows
      if ((TandP_jpsi_minMass < invMass) && (invMass < TandP_jpsi_maxMass)) {
        fillHistograms_resonance(histos.patElectron.resonanceJ,
                                 pat_el_second,
                                 vtrigger_result, legObjects,l1_legObjects, l1_result,
                                 passBaseDST,
                                 invMass,
                                 second_pat_pt_order);  // J/Psi mass: 3.3 +/- 0.2 GeV
        fillHistograms_resonance(histos.patElectron.resonanceAll, pat_el_second, vtrigger_result, legObjects, l1_legObjects, l1_result, passBaseDST, invMass, second_pat_pt_order);
      }

      // ups mass windows
      if ((TandP_ups_minMass < invMass) && (invMass < TandP_ups_maxMass)) {
        fillHistograms_resonance(histos.patElectron.resonanceY,
                                 pat_el_second,
                                 vtrigger_result, legObjects,l1_legObjects, l1_result,
                                 passBaseDST,
                                 invMass,
                                 second_pat_pt_order);  // Y mass: 9.8 +/- 0.4 GeV & 10.6 +/- 1 GeV
        fillHistograms_resonance(histos.patElectron.resonanceAll, pat_el_second, vtrigger_result, legObjects,l1_legObjects, l1_result, passBaseDST, invMass, second_pat_pt_order);
      }

    }

    int sct_pt_order = -1;
    for (size_t sct_local_index = 0; sct_local_index < indexed_sctElectrons.size(); ++sct_local_index){
      const auto sct_el_index = indexed_sctElectrons[sct_local_index].first;
      const auto sct_el_second = indexed_sctElectrons[sct_local_index].second;
      int gsfTrackIndex = sctElectron_gsfTrackIndex[sct_el_index];
      if (gsfTrackIndex < 0)
        continue;

      ROOT::Math::PtEtaPhiMVector sctEl0(
          sct_el_second.pt(), sct_el_second.eta(), sct_el_second.phi(), sct_el_second.m());
      ROOT::Math::PtEtaPhiMVector probe_sct_el(scoutingDQMUtils::computePtFromEnergyMassEta(
                                                   sctEl0.energy(), 0.0005, sct_el_second.trketaMode()[gsfTrackIndex]),
                                               sct_el_second.trketaMode()[gsfTrackIndex],
                                               sct_el_second.trkphiMode()[gsfTrackIndex],
                                               0.0005);

      if (!scoutingDQMUtils::scoutingElectronID(sct_el_second))
        continue;
      sct_pt_order += 1;

      if (ROOT::Math::VectorUtil::DeltaR(probe_sct_el, tag_pat_el) < 0.1)
        continue;


      float invMass = (tag_pat_el + probe_sct_el).mass();
      if ((TandP_Z_minMass < invMass) && (invMass < TandP_Z_maxMass)) {
        fillHistograms_resonance_sct(histos.sctElectron.resonanceZ, sct_el_second, gsfTrackIndex, vtrigger_result, legObjects, l1_legObjects, l1_result, passBaseDST, invMass, sct_pt_order);
        fillHistograms_resonance_sct(histos.sctElectron.resonanceAll, sct_el_second, gsfTrackIndex, vtrigger_result, legObjects, l1_legObjects, l1_result, passBaseDST, invMass, sct_pt_order);
      }
      if ((TandP_jpsi_minMass < invMass) && (invMass < TandP_jpsi_maxMass)) {
        fillHistograms_resonance_sct(histos.sctElectron.resonanceJ,
                                     sct_el_second,
                                     gsfTrackIndex, 
                                     vtrigger_result, legObjects, l1_legObjects, l1_result,
                                     passBaseDST,
                                     invMass,
                                     sct_pt_order);  // J/Psi mass: 3.3 +/- 0.2 GeV
        fillHistograms_resonance_sct(histos.sctElectron.resonanceAll, sct_el_second, gsfTrackIndex, vtrigger_result, legObjects, l1_legObjects, l1_result, passBaseDST, invMass, sct_pt_order);
      }
      if ((TandP_ups_minMass < invMass) && (invMass < TandP_ups_maxMass)) {
        fillHistograms_resonance_sct(histos.sctElectron.resonanceY,
                                     sct_el_second,
                                     gsfTrackIndex, 
                                     vtrigger_result, legObjects, l1_legObjects, l1_result,
                                     passBaseDST,
                                     invMass,
                                     sct_pt_order);  // Y mass: 9.8 +/- 0.4 GeV & 10.6 +/- 1 GeV
        fillHistograms_resonance_sct(histos.sctElectron.resonanceAll, sct_el_second, gsfTrackIndex, vtrigger_result, legObjects,l1_legObjects, l1_result, passBaseDST, invMass, sct_pt_order);
      }
    }
  }
}

void PatElectronTagProbeAnalyzer::fillHistograms_resonance(const kProbeKinematicHistos& histos,
                                                           const pat::Electron& el,
                                                           const std::vector<bool> trigger_result,
                                                           const trigger::TriggerObjectCollection* legObjects,
                                                           const trigger::TriggerObjectCollection* l1_legObjects,
                                                           const std::vector<bool> l1_result,
                                                           const bool pass_baseDST,
                                                           const float inv_mass,
                                                           const int pt_order) const {
  histos.hEta->Fill(el.eta());
  histos.hPhi->Fill(el.phi());
  histos.hInvMass->Fill(inv_mass);
  histos.hEtavsInvMass->Fill(el.eta(), inv_mass);

  if (el.isEB()) {
    histos.hPt_Barrel->Fill(el.pt());
    histos.hHoverE_Barrel->Fill(el.hadronicOverEm());
    histos.hOoEMOoP_Barrel->Fill((1.0 / el.ecalEnergy() - el.eSuperClusterOverP() / el.ecalEnergy()));
    histos.hdPhiIn_Barrel->Fill(std::abs(el.deltaPhiSuperClusterTrackAtVtx()));
    histos.hdEtaIn_Barrel->Fill(std::abs(el.deltaEtaSuperClusterTrackAtVtx()));
    histos.hSigmaIetaIeta_Barrel->Fill(el.sigmaIetaIeta());
    histos.hMissingHits_Barrel->Fill(
        el.gsfTrack()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS));
    histos.hTrackfbrem_Barrel->Fill(el.fbrem());
    histos.hRelEcalIsolation_Barrel->Fill(el.ecalIso() / el.pt());
    histos.hRelHcalIsolation_Barrel->Fill(el.hcalIso() / el.pt());
    histos.hRelTrackIsolation_Barrel->Fill(el.trackIso() / el.pt());
    histos.hPtvsInvMass_Barrel->Fill(el.pt(), inv_mass);

    if(pass_baseDST){

      if (pt_order == 0){
        histos.leading_electron.hPt_Barrel_passBaseDST->Fill(el.pt());
        histos.leading_electron.hEta_passBaseDST->Fill(el.eta());
        for(unsigned int iTrig = 0; iTrig < vtriggerSelection_.size(); iTrig++){
            if(trigger_result[iTrig]){
               histos.leading_electron.hPt_Barrel_passDST[iTrig]->Fill(el.pt());
               histos.leading_electron.hEta_passDST[iTrig]->Fill(el.eta());
            }
            if(patElectron_passHLT(el, legObjects[iTrig])){
               histos.leading_electron.hPt_Barrel_fireTrigObj[iTrig]->Fill(el.pt());
               histos.leading_electron.hEta_fireTrigObj[iTrig]->Fill(el.eta());
            }
        }

        for(unsigned int il1seed = 0; il1seed < l1Seeds_.size(); il1seed++){
            if(l1_result[il1seed]){
               unsigned int ifilter = 0;
               while ((ifilter < l1filterIndex_.size()) && (il1seed >= l1filterIndex_[ifilter])) ifilter++;
               if(!(patElectron_passHLT(el, l1_legObjects[ifilter]))) continue;
               // check if electron fire the l1 seed leg
               histos.leading_electron.hPt_Barrel_fireL1[il1seed]->Fill(el.pt());
               histos.leading_electron.hEta_fireL1[il1seed]->Fill(el.eta());
            }
        }
      }
      else if (pt_order == 1){
        histos.subleading_electron.hPt_Barrel_passBaseDST->Fill(el.pt());
        histos.subleading_electron.hEta_passBaseDST->Fill(el.eta());
        for(unsigned int iTrig = 0; iTrig < vtriggerSelection_.size(); iTrig++){
            if(trigger_result[iTrig]){
               histos.subleading_electron.hPt_Barrel_passDST[iTrig]->Fill(el.pt());
               histos.subleading_electron.hEta_passDST[iTrig]->Fill(el.eta());
            }
            if(patElectron_passHLT(el, legObjects[iTrig])){
               histos.subleading_electron.hPt_Barrel_fireTrigObj[iTrig]->Fill(el.pt());
               histos.subleading_electron.hEta_fireTrigObj[iTrig]->Fill(el.eta());
            }
        }

        for(unsigned int il1seed = 0; il1seed < l1Seeds_.size(); il1seed++){
            if(l1_result[il1seed]){
               unsigned int ifilter = 0;
               while ((ifilter < l1filterIndex_.size()) && (il1seed >= l1filterIndex_[ifilter])) ifilter++;
               if(!(patElectron_passHLT(el, l1_legObjects[ifilter]))) continue;
               histos.subleading_electron.hPt_Barrel_fireL1[il1seed]->Fill(el.pt());
               histos.subleading_electron.hEta_fireL1[il1seed]->Fill(el.eta());
            }
        }
      }
    }
  } else {
    histos.hPt_Endcap->Fill(el.pt());
    histos.hHoverE_Endcap->Fill(el.hadronicOverEm());
    histos.hOoEMOoP_Endcap->Fill((1.0 / el.ecalEnergy() - el.eSuperClusterOverP() / el.ecalEnergy()));
    histos.hdPhiIn_Endcap->Fill(std::abs(el.deltaPhiSuperClusterTrackAtVtx()));
    histos.hdEtaIn_Endcap->Fill(std::abs(el.deltaEtaSuperClusterTrackAtVtx()));
    histos.hSigmaIetaIeta_Endcap->Fill(el.sigmaIetaIeta());
    histos.hMissingHits_Endcap->Fill(
        el.gsfTrack()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS));
    histos.hTrackfbrem_Endcap->Fill(el.fbrem());
    histos.hRelEcalIsolation_Endcap->Fill(el.ecalIso() / el.pt());
    histos.hRelHcalIsolation_Endcap->Fill(el.hcalIso() / el.pt());
    histos.hRelTrackIsolation_Endcap->Fill(el.trackIso() / el.pt());
    histos.hPtvsInvMass_Endcap->Fill(el.pt(), inv_mass);

    if(pass_baseDST){
      if (pt_order == 0){
        histos.leading_electron.hPt_Endcap_passBaseDST->Fill(el.pt());
        histos.leading_electron.hEta_passBaseDST->Fill(el.eta());
        for(unsigned int iTrig = 0; iTrig < vtriggerSelection_.size(); iTrig++){
            if(trigger_result[iTrig]){
               histos.leading_electron.hPt_Endcap_passDST[iTrig]->Fill(el.pt());
               histos.leading_electron.hEta_passDST[iTrig]->Fill(el.eta());
            }
            if(patElectron_passHLT(el, legObjects[iTrig])){
               histos.leading_electron.hPt_Endcap_fireTrigObj[iTrig]->Fill(el.pt());
               histos.leading_electron.hEta_fireTrigObj[iTrig]->Fill(el.eta());
            }
        }

        for(unsigned int il1seed = 0; il1seed < l1Seeds_.size(); il1seed++){
            if(l1_result[il1seed]){
               unsigned int ifilter = 0;
               while ((ifilter < l1filterIndex_.size()) && (il1seed >= l1filterIndex_[ifilter])) ifilter++;
               if(!(patElectron_passHLT(el, l1_legObjects[ifilter]))) continue;
               histos.leading_electron.hPt_Endcap_fireL1[il1seed]->Fill(el.pt());
               histos.leading_electron.hEta_fireL1[il1seed]->Fill(el.eta());
            }
        }
      }
      else if (pt_order == 1){
        histos.subleading_electron.hPt_Endcap_passBaseDST->Fill(el.pt());
        histos.subleading_electron.hEta_passBaseDST->Fill(el.eta());
        for(unsigned int iTrig = 0; iTrig < vtriggerSelection_.size(); iTrig++){
            if(trigger_result[iTrig]){
               histos.subleading_electron.hPt_Endcap_passDST[iTrig]->Fill(el.pt());
               histos.subleading_electron.hEta_passDST[iTrig]->Fill(el.eta());
            }
            if(patElectron_passHLT(el, legObjects[iTrig])){
               histos.subleading_electron.hPt_Endcap_fireTrigObj[iTrig]->Fill(el.pt());
               histos.subleading_electron.hEta_fireTrigObj[iTrig]->Fill(el.eta());
            }
        }

        for(unsigned int il1seed = 0; il1seed < l1Seeds_.size(); il1seed++){
            if(l1_result[il1seed]){
               unsigned int ifilter = 0;
               while ((ifilter < l1filterIndex_.size()) && (il1seed >= l1filterIndex_[ifilter])) ifilter++;
               if(!(patElectron_passHLT(el, l1_legObjects[ifilter]))) continue;
               histos.subleading_electron.hPt_Endcap_fireL1[il1seed]->Fill(el.pt());
               histos.subleading_electron.hEta_fireL1[il1seed]->Fill(el.eta());
            }
        }
      }
    }
  }
}

void PatElectronTagProbeAnalyzer::fillHistograms_resonance_sct(const kProbeKinematicHistos& histos,
                                                               const Run3ScoutingElectron& el,
                                                               const int gsfTrackIndex,
                                                               const std::vector<bool> trigger_result,
                                                               const trigger::TriggerObjectCollection* legObjects,
                                                               const trigger::TriggerObjectCollection* l1_legObjects,
                                                               const std::vector<bool> l1_result,
                                                               const bool pass_baseDST,
                                                               const float inv_mass,
                                                               const int pt_order) const {
  histos.hEta->Fill(el.eta());
  histos.hPhi->Fill(el.phi());
  histos.hInvMass->Fill(inv_mass);
  histos.hEtavsInvMass->Fill(el.eta(), inv_mass);

  if (std::abs(el.eta()) < 1.5) {
    histos.hPt_Barrel->Fill(el.pt());
    histos.hHoverE_Barrel->Fill(el.hOverE());
    histos.hOoEMOoP_Barrel->Fill(el.ooEMOop());
    histos.hdPhiIn_Barrel->Fill(std::abs(el.dPhiIn()));
    histos.hdEtaIn_Barrel->Fill(std::abs(el.dEtaIn()));
    histos.hSigmaIetaIeta_Barrel->Fill(el.sigmaIetaIeta());
    histos.hMissingHits_Barrel->Fill(el.missingHits());
    histos.hTrackfbrem_Barrel->Fill(el.trackfbrem());
    histos.hRelEcalIsolation_Barrel->Fill(el.ecalIso() / el.pt());
    histos.hRelHcalIsolation_Barrel->Fill(el.hcalIso() / el.pt());
    histos.hRelTrackIsolation_Barrel->Fill(el.trackIso() / el.pt());
    histos.hPtvsInvMass_Barrel->Fill(el.pt(), inv_mass);
    for (const auto& trk : el.trkpt()) {
      histos.hTrack_pt_Barrel->Fill(trk);
    }
    for (const auto& trk : el.trkpMode()) {
      histos.hTrack_pMode_Barrel->Fill(trk);
    }
    for (const auto& trk : el.trketaMode()) {
      histos.hTrack_etaMode_Barrel->Fill(trk);
    }
    for (const auto& trk : el.trkphiMode()) {
      histos.hTrack_phiMode_Barrel->Fill(trk);
    }
    for (const auto& trk : el.trkqoverpModeError()) {
      histos.hTrack_qoverpModeError_Barrel->Fill(trk);
    }

    if(pass_baseDST){

      if (pt_order == 0){
        histos.leading_electron.hPt_Barrel_passBaseDST->Fill(el.pt());
        histos.leading_electron.hEta_passBaseDST->Fill(el.eta());

        for(unsigned int iTrig = 0; iTrig < vtriggerSelection_.size(); iTrig++){
          if(trigger_result[iTrig]){
             histos.leading_electron.hPt_Barrel_passDST[iTrig]->Fill(el.pt());
             histos.leading_electron.hEta_passDST[iTrig]->Fill(el.eta());
          }
          if(scoutingElectron_passHLT(el.trketaMode()[gsfTrackIndex], el.trkphiMode()[gsfTrackIndex], legObjects[iTrig])){
             histos.leading_electron.hPt_Barrel_fireTrigObj[iTrig]->Fill(el.pt());
             histos.leading_electron.hEta_fireTrigObj[iTrig]->Fill(el.eta());
          }
        }

        for(unsigned int il1seed = 0; il1seed < l1Seeds_.size(); il1seed++){
          if(l1_result[il1seed]){
             unsigned int ifilter = 0;
             while ((ifilter < l1filterIndex_.size()) && (il1seed >= l1filterIndex_[ifilter])) ifilter++;
             if(!(scoutingElectron_passHLT(el.trketaMode()[gsfTrackIndex], el.trkphiMode()[gsfTrackIndex], l1_legObjects[ifilter]))) continue;

             histos.leading_electron.hPt_Barrel_fireL1[il1seed]->Fill(el.pt());
             histos.leading_electron.hEta_fireL1[il1seed]->Fill(el.eta());
          }
        }
      }
      else if (pt_order == 1){
        histos.subleading_electron.hPt_Barrel_passBaseDST->Fill(el.pt());
        histos.subleading_electron.hEta_passBaseDST->Fill(el.eta());

        for(unsigned int iTrig = 0; iTrig < vtriggerSelection_.size(); iTrig++){
          if(trigger_result[iTrig]){
             histos.subleading_electron.hPt_Barrel_passDST[iTrig]->Fill(el.pt());
             histos.subleading_electron.hEta_passDST[iTrig]->Fill(el.eta());
          }
          if(scoutingElectron_passHLT(el.trketaMode()[gsfTrackIndex], el.trkphiMode()[gsfTrackIndex], legObjects[iTrig])){
             histos.subleading_electron.hPt_Barrel_fireTrigObj[iTrig]->Fill(el.pt());
             histos.subleading_electron.hEta_fireTrigObj[iTrig]->Fill(el.eta());
          }
        }

        for(unsigned int il1seed = 0; il1seed < l1Seeds_.size(); il1seed++){
          if(l1_result[il1seed]){
             unsigned int ifilter = 0;
             while ((ifilter < l1filterIndex_.size()) && (il1seed >= l1filterIndex_[ifilter])) ifilter++;
             if(!(scoutingElectron_passHLT(el.trketaMode()[gsfTrackIndex], el.trkphiMode()[gsfTrackIndex], l1_legObjects[ifilter]))) continue;

             histos.subleading_electron.hPt_Barrel_fireL1[il1seed]->Fill(el.pt());
             histos.subleading_electron.hEta_fireL1[il1seed]->Fill(el.eta());
          }
        }
      }

    }
  } else {
    histos.hPt_Endcap->Fill(el.pt());
    histos.hHoverE_Endcap->Fill(el.hOverE());
    histos.hOoEMOoP_Endcap->Fill(el.ooEMOop());
    histos.hdPhiIn_Endcap->Fill(std::abs(el.dPhiIn()));
    histos.hdEtaIn_Endcap->Fill(std::abs(el.dEtaIn()));
    histos.hSigmaIetaIeta_Endcap->Fill(el.sigmaIetaIeta());
    histos.hMissingHits_Endcap->Fill(el.missingHits());
    histos.hTrackfbrem_Endcap->Fill(el.trackfbrem());
    histos.hRelEcalIsolation_Endcap->Fill(el.ecalIso() / el.pt());
    histos.hRelHcalIsolation_Endcap->Fill(el.hcalIso() / el.pt());
    histos.hRelTrackIsolation_Endcap->Fill(el.trackIso() / el.pt());
    histos.hPtvsInvMass_Endcap->Fill(el.pt(), inv_mass);
    for (const auto& trk : el.trkpt()) {
      histos.hTrack_pt_Endcap->Fill(trk);
    }
    for (const auto& trk : el.trkpMode()) {
      histos.hTrack_pMode_Endcap->Fill(trk);
    }
    for (const auto& trk : el.trketaMode()) {
      histos.hTrack_etaMode_Endcap->Fill(trk);
    }
    for (const auto& trk : el.trkphiMode()) {
      histos.hTrack_phiMode_Endcap->Fill(trk);
    }
    for (const auto& trk : el.trkqoverpModeError()) {
      histos.hTrack_qoverpModeError_Endcap->Fill(trk);
    }

    if(pass_baseDST){

      if (pt_order == 0){
        histos.leading_electron.hPt_Endcap_passBaseDST->Fill(el.pt());
        histos.leading_electron.hEta_passBaseDST->Fill(el.eta());

        for(unsigned int iTrig = 0; iTrig < vtriggerSelection_.size(); iTrig++){
          if(trigger_result[iTrig]){
             histos.leading_electron.hPt_Endcap_passDST[iTrig]->Fill(el.pt());
             histos.leading_electron.hEta_passDST[iTrig]->Fill(el.eta());
          }
          if(scoutingElectron_passHLT(el.trketaMode()[gsfTrackIndex], el.trkphiMode()[gsfTrackIndex], legObjects[iTrig])){
             histos.leading_electron.hPt_Endcap_fireTrigObj[iTrig]->Fill(el.pt());
             histos.leading_electron.hEta_fireTrigObj[iTrig]->Fill(el.eta());
          }
        }

        for(unsigned int il1seed = 0; il1seed < l1Seeds_.size(); il1seed++){
          if(l1_result[il1seed]){
             unsigned int ifilter = 0;
             while ((ifilter < l1filterIndex_.size()) && (il1seed >= l1filterIndex_[ifilter])) ifilter++;
             if(!(scoutingElectron_passHLT(el.trketaMode()[gsfTrackIndex], el.trkphiMode()[gsfTrackIndex], l1_legObjects[ifilter]))) continue;

             histos.leading_electron.hPt_Endcap_fireL1[il1seed]->Fill(el.pt());
             histos.leading_electron.hEta_fireL1[il1seed]->Fill(el.eta());
          }
        }
      }
      else if (pt_order == 1){
        histos.subleading_electron.hPt_Endcap_passBaseDST->Fill(el.pt());
        histos.subleading_electron.hEta_passBaseDST->Fill(el.eta());

        for(unsigned int iTrig = 0; iTrig < vtriggerSelection_.size(); iTrig++){
          if(trigger_result[iTrig]){
             histos.subleading_electron.hPt_Endcap_passDST[iTrig]->Fill(el.pt());
             histos.subleading_electron.hEta_passDST[iTrig]->Fill(el.eta());
          }
          if(scoutingElectron_passHLT(el.trketaMode()[gsfTrackIndex], el.trkphiMode()[gsfTrackIndex], legObjects[iTrig])){
             histos.subleading_electron.hPt_Endcap_fireTrigObj[iTrig]->Fill(el.pt());
             histos.subleading_electron.hEta_fireTrigObj[iTrig]->Fill(el.eta());
          }
        }

        for(unsigned int il1seed = 0; il1seed < l1Seeds_.size(); il1seed++){
          if(l1_result[il1seed]){
             unsigned int ifilter = 0;
             while ((ifilter < l1filterIndex_.size()) && (il1seed >= l1filterIndex_[ifilter])) ifilter++;
             if(!(scoutingElectron_passHLT(el.trketaMode()[gsfTrackIndex], el.trkphiMode()[gsfTrackIndex], l1_legObjects[ifilter]))) continue;

             histos.subleading_electron.hPt_Endcap_fireL1[il1seed]->Fill(el.pt());
             histos.subleading_electron.hEta_fireL1[il1seed]->Fill(el.eta());
          }
        }
      }
    }
  }
}
void PatElectronTagProbeAnalyzer::bookHistograms(DQMStore::IBooker& ibook,
                                                 edm::Run const& run,
                                                 edm::EventSetup const& iSetup,
                                                 kTagProbeHistos& histos) const {
  ibook.setCurrentFolder(outputInternalPath_);

  bookHistograms_resonance(ibook, run, iSetup, histos.patElectron.resonanceZ, "resonanceZ_Tag_pat_Probe_patElectron");
  bookHistograms_resonance(ibook, run, iSetup, histos.patElectron.resonanceJ, "resonanceJ_Tag_pat_Probe_patElectron");
  bookHistograms_resonance(ibook, run, iSetup, histos.patElectron.resonanceY, "resonanceY_Tag_pat_Probe_patElectron");
  bookHistograms_resonance(ibook, run, iSetup, histos.patElectron.resonanceAll, "resonanceAll_Tag_pat_Probe_patElectron");

  bookHistograms_resonance(ibook, run, iSetup, histos.sctElectron.resonanceZ, "resonanceZ_Tag_pat_Probe_sctElectron");
  bookHistograms_resonance(ibook, run, iSetup, histos.sctElectron.resonanceJ, "resonanceJ_Tag_pat_Probe_sctElectron");
  bookHistograms_resonance(ibook, run, iSetup, histos.sctElectron.resonanceY, "resonanceY_Tag_pat_Probe_sctElectron");
  bookHistograms_resonance(ibook, run, iSetup, histos.sctElectron.resonanceAll, "resonanceAll_Tag_pat_Probe_sctElectron");
}

void PatElectronTagProbeAnalyzer::bookHistograms_resonance(DQMStore::IBooker& ibook,
                                                           edm::Run const& run,
                                                           edm::EventSetup const& iSetup,
                                                           kProbeKinematicHistos& histos,
                                                           const std::string& name) const {
  ibook.setCurrentFolder(outputInternalPath_);
  histos.hPt_Barrel = ibook.book1D(name + "_Pt_Barrel", name + "_Pt_Barrel", 40, 0, 200);
  histos.hPt_Endcap = ibook.book1D(name + "_Pt_Endcap", name + "_Pt_Endcap", 40, 0, 200);

  histos.hEta = ibook.book1D(name + "_Eta", name + "_Eta", 20, -5.0, 5.0);
  histos.hPhi = ibook.book1D(name + "_Phi", name + "_Phi", 10, -3.3, 3.3);

  histos.hHoverE_Barrel = ibook.book1D(name + "_HoverE_Barrel", name + "_HoverE_Barrel", 200, 0, 0.5);
  histos.hHoverE_Endcap = ibook.book1D(name + "_HoverE_Endcap", name + "_HoverE_Endcap", 200, 0, 0.5);

  histos.hOoEMOoP_Barrel = ibook.book1D(name + "_OoEMOoP_Barrel", name + "_OoEMOoP_Barrel", 250, -0.2, 0.2);
  histos.hOoEMOoP_Endcap = ibook.book1D(name + "_OoEMOoP_Endcap", name + "_OoEMOoP_Endcap", 250, -0.2, 0.2);

  histos.hdPhiIn_Barrel = ibook.book1D(name + "_dPhiIn_Barrel", name + "_dPhiIn_Barrel", 100, 0, 0.1);
  histos.hdPhiIn_Endcap = ibook.book1D(name + "_dPhiIn_Endcap", name + "_dPhiIn_Endcap", 100, 0, 0.1);

  histos.hdEtaIn_Barrel = ibook.book1D(name + "_dEtaIn_Barrel", name + "_dEtaIn_Barrel", 100, 0, 0.1);
  histos.hdEtaIn_Endcap = ibook.book1D(name + "_dEtaIn_Endcap", name + "_dEtaIn_Endcap", 100, 0, 0.1);

  histos.hSigmaIetaIeta_Barrel =
      ibook.book1D(name + "_SigmaIetaIeta_Barrel", name + "_SigmaIetaIeta_Barrel", 500, 0, 0.05);
  histos.hSigmaIetaIeta_Endcap =
      ibook.book1D(name + "_SigmaIetaIeta_Endcap", name + "_SigmaIetaIeta_Endcap", 500, 0, 0.05);

  histos.hMissingHits_Barrel = ibook.book1D(name + "_MissingHits_Barrel", name + "_MissingHits_Barrel", 21, -0.5, 20.5);
  histos.hMissingHits_Endcap = ibook.book1D(name + "_MissingHits_Endcap", name + "_MissingHits_Endcap", 21, -0.5, 20.5);

  histos.hTrackfbrem_Barrel = ibook.book1D(name + "_Trackfbrem_Barrel", name + "_Trackfbrem_Barrel", 100, 0, 1.0);
  histos.hTrackfbrem_Endcap = ibook.book1D(name + "_Trackfbrem_Endcap", name + "_Trackfbrem_Endcap", 100, 0, 1.0);

  histos.hTrack_pt_Barrel = ibook.book1D(name + "_Track_pt_Barrel", name + "_Track_pt_Barrel", 200, 0, 100.0);
  histos.hTrack_pt_Endcap = ibook.book1D(name + "_Track_pt_Endcap", name + "_Track_pt_Endcap", 200, 0, 100.0);

  histos.hTrack_pMode_Barrel = ibook.book1D(name + "_Track_pMode_Barrel", name + "_Track_pMode_Barrel", 50, -0.5, 49.5);
  histos.hTrack_pMode_Endcap = ibook.book1D(name + "_Track_pMode_Endcap", name + "_Track_pMode_Endcap", 50, -0.5, 49.5);

  histos.hTrack_etaMode_Barrel =
      ibook.book1D(name + "_Track_etaMode_Barrel", name + "_Track_etaMode_Barrel", 26, -6.5, 6.5);
  histos.hTrack_etaMode_Endcap =
      ibook.book1D(name + "_Track_etaMode_Endcap", name + "_Track_etaMode_Endcap", 26, -6.5, 6.5);

  histos.hTrack_phiMode_Barrel =
      ibook.book1D(name + "_Track_phiMode_Barrel", name + "_Track_phiMode_Barrel", 18, -4.5, 4.5);
  histos.hTrack_phiMode_Endcap =
      ibook.book1D(name + "_Track_phiMode_Endcap", name + "_Track_phiMode_Endcap", 18, -4.5, 4.5);

  histos.hTrack_qoverpModeError_Barrel =
      ibook.book1D(name + "_Track_qoverpModeError_Barrel", name + "_Track_qoverpModeError_Barrel", 36, -4.5, 4.5);
  histos.hTrack_qoverpModeError_Endcap =
      ibook.book1D(name + "_Track_qoverpModeError_Endcap", name + "_Track_qoverpModeError_Endcap", 36, -4.5, 4.5);

  histos.hRelEcalIsolation_Barrel =
      ibook.book1D(name + "_RelEcalIsolation_Barrel", name + "_RelEcalIsolation_Barrel", 100, 0, 1.0);
  histos.hRelEcalIsolation_Endcap =
      ibook.book1D(name + "_RelEcalIsolation_Endcap", name + "_RelEcalIsolation_Endcap", 100, 0, 1.0);

  histos.hRelHcalIsolation_Barrel =
      ibook.book1D(name + "_RelHcalIsolation_Barrel", name + "_RelHcalIsolation_Barrel", 100, 0, 1.0);
  histos.hRelHcalIsolation_Endcap =
      ibook.book1D(name + "_RelHcalIsolation_Endcap", name + "_RelHcalIsolation_Endcap", 100, 0, 1.0);

  histos.hRelTrackIsolation_Barrel =
      ibook.book1D(name + "_RelTrackIsolation_Barrel", name + "_RelTrackIsolation_Barrel", 100, 0, 1.0);
  histos.hRelTrackIsolation_Endcap =
      ibook.book1D(name + "_RelTrackIsolation_Endcap", name + "_RelTrackIsolation_Endcap", 100, 0, 1.0);

  histos.hPtvsInvMass_Barrel =
      ibook.book2D(name + "_PtvsInvMass_Barrel", name + "_PtvsInvMass_Barrel", 5, 0, 100, 10, 80, 100);
  histos.hPtvsInvMass_Endcap =
      ibook.book2D(name + "_PtvsInvMass_Endcap", name + "_PtvsInvMass_Endcap", 5, 0, 100, 10, 80, 100);

  histos.hEtavsInvMass =
      ibook.book2D(name + "_EtavsInvMass", name + "_EtavsInvMass", 10, -2.5, 2.5, 10, TandP_Z_minMass, TandP_Z_maxMass);

  histos.hInvMass = ibook.book1D(name + "_Invariant_Mass", name + "_Invariant_Mass", 800, 0, 200);


  // Leading Electron
  histos.leading_electron.hPt_Barrel_passBaseDST =
      ibook.book1D(name + "_leading_Pt_Barrel_passBaseDST", name + "_leading_Pt_Barrel_passBaseDST",  40, 0, 200);
  histos.leading_electron.hPt_Endcap_passBaseDST = 
      ibook.book1D(name + "_leading_Pt_Endcap_passBaseDST", name + "_leading_Pt_Endcap_passBaseDST",  40, 0, 200);
  histos.leading_electron.hEta_passBaseDST = 
      ibook.book1D(name + "_leading_Eta_passBaseDST", name + "_leading_Eta_passBaseDST",  20, -5.0, 5.0);

  // Sub-Leading Electron
  histos.subleading_electron.hPt_Barrel_passBaseDST =
      ibook.book1D(name + "_subleading_Pt_Barrel_passBaseDST", name + "_subleading_Pt_Barrel_passBaseDST",  40, 0, 200);
  histos.subleading_electron.hPt_Endcap_passBaseDST = 
      ibook.book1D(name + "_subleading_Pt_Endcap_passBaseDST", name + "_subleading_Pt_Endcap_passBaseDST",  40, 0, 200);
  histos.subleading_electron.hEta_passBaseDST = 
      ibook.book1D(name + "_subleading_Eta_passBaseDST", name + "_subleading_Eta_passBaseDST",  20, -5.0, 5.0);




  for (auto const &vt : vtriggerSelection_){
      std::string cleaned_vt = vt;
      cleaned_vt.erase(std::remove(cleaned_vt.begin(), cleaned_vt.end(), '*'), cleaned_vt.end());

      // Leading Electron
      histos.leading_electron.hPt_Barrel_passDST.push_back(
          ibook.book1D(name + "_leading_Pt_Barrel_pass" + cleaned_vt, name + "_leading_Pt_Barrel_pass" + cleaned_vt,  40, 0, 200)
      );
      histos.leading_electron.hPt_Endcap_passDST.push_back(
          ibook.book1D(name + "_leading_Pt_Endcap_pass" + cleaned_vt, name + "_leading_Pt_Endcap_pass" + cleaned_vt,  40, 0, 200)
      );
      histos.leading_electron.hEta_passDST.push_back(
          ibook.book1D(name + "_leading_Eta_pass" + cleaned_vt, name + "_leading_Eta_pass" + cleaned_vt,  20, -5.0, 5.0)
      );

      histos.leading_electron.hPt_Barrel_fireTrigObj.push_back(
          ibook.book1D(name + "_leading_Pt_Barrel_pass" + cleaned_vt + "_fireTrigObj", name + "_leading_Pt_Barrel_pass" + cleaned_vt + "_fireTrigObj",  40, 0, 200)
      );
      histos.leading_electron.hPt_Endcap_fireTrigObj.push_back(
          ibook.book1D(name + "_leading_Pt_Endcap_pass" + cleaned_vt + "_fireTrigObj", name + "_leading_Pt_Endcap_pass" + cleaned_vt + "_fireTrigObj",  40, 0, 200)
      );
      histos.leading_electron.hEta_fireTrigObj.push_back(
          ibook.book1D(name + "_leading_Eta_pass" + cleaned_vt + "_fireTrigObj", name + "_leading_Eta_pass" + cleaned_vt + "_fireTrigObj",  20, -5.0, 5.0)
      );

      // SubLeading Electron
      histos.subleading_electron.hPt_Barrel_passDST.push_back(
          ibook.book1D(name + "_subleading_Pt_Barrel_pass" + cleaned_vt, name + "_subleading_Pt_Barrel_pass" + cleaned_vt,  40, 0, 200)
      );
      histos.subleading_electron.hPt_Endcap_passDST.push_back(
          ibook.book1D(name + "_subleading_Pt_Endcap_pass" + cleaned_vt, name + "_subleading_Pt_Endcap_pass" + cleaned_vt,  40, 0, 200)
      );
      histos.subleading_electron.hEta_passDST.push_back(
          ibook.book1D(name + "_subleading_Eta_pass" + cleaned_vt, name + "_subleading_Eta_pass" + cleaned_vt,  20, -5.0, 5.0)
      );

      histos.subleading_electron.hPt_Barrel_fireTrigObj.push_back(
          ibook.book1D(name + "_subleading_Pt_Barrel_pass" + cleaned_vt + "_fireTrigObj", name + "_subleading_Pt_Barrel_pass" + cleaned_vt + "_fireTrigObj",  40, 0, 200)
      );
      histos.subleading_electron.hPt_Endcap_fireTrigObj.push_back(
          ibook.book1D(name + "_subleading_Pt_Endcap_pass" + cleaned_vt + "_fireTrigObj", name + "_subleading_Pt_Endcap_pass" + cleaned_vt + "_fireTrigObj",  40, 0, 200)
      );
      histos.subleading_electron.hEta_fireTrigObj.push_back(
          ibook.book1D(name + "_subleading_Eta_pass" + cleaned_vt + "_fireTrigObj", name + "_subleading_Eta_pass" + cleaned_vt + "_fireTrigObj",  20, -5.0, 5.0)
      );

  }

  for (auto const &l1seed : l1Seeds_){

     // leading Electron
     histos.leading_electron.hPt_Barrel_fireL1.push_back(
          ibook.book1D(name + "_leading_Pt_Barrel_pass" + l1seed, name + "_leading_Pt_Barrel_pass" + l1seed,  40, 0, 200)
      );
      histos.leading_electron.hPt_Endcap_fireL1.push_back(
          ibook.book1D(name + "_leading_Pt_Endcap_pass" + l1seed, name + "_leading_Pt_Endcap_pass" + l1seed,  40, 0, 200)
      );
      histos.leading_electron.hEta_fireL1.push_back(
          ibook.book1D(name + "_leading_Eta_pass" + l1seed, name + "_leading_Eta_pass" + l1seed,  20, -5.0, 5.0)
      );

      // Subleading Electron
      histos.subleading_electron.hPt_Barrel_fireL1.push_back(

          ibook.book1D(name + "_subleading_Pt_Barrel_pass" + l1seed, name + "_subleading_Pt_Barrel_pass" + l1seed,  40, 0, 200)
      );
      histos.subleading_electron.hPt_Endcap_fireL1.push_back(
          ibook.book1D(name + "_subleading_Pt_Endcap_pass" + l1seed, name + "_subleading_Pt_Endcap_pass" + l1seed,  40, 0, 200)
      );
      histos.subleading_electron.hEta_fireL1.push_back(
          ibook.book1D(name + "_subleading_Eta_pass" + l1seed, name + "_subleading_Eta_pass" + l1seed,  20, -5.0, 5.0)
      );

  }

}

void PatElectronTagProbeAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("OutputInternalPath", "MY_FOLDER");
  desc.add<std::vector<std::string>>("BaseTriggerSelection", {});
  desc.add<std::vector<std::string>>("triggerSelection", {});
  desc.add<std::vector<std::string>>("finalfilterSelection", {});
  desc.add<std::vector<std::string>>("l1filterSelection", {});
  desc.add<std::vector<unsigned int>>("l1filterSelectionIndex", {});
  desc.add<edm::InputTag>("AlgInputTag", edm::InputTag("gtStage2Digis"));
  desc.add<std::vector<std::string>>("L1Seeds", {});
  desc.add<edm::InputTag>("l1tAlgBlkInputTag", edm::InputTag("gtStage2Digis"));
  desc.add<edm::InputTag>("l1tExtBlkInputTag", edm::InputTag("gtStage2Digis"));
  desc.add<bool>("ReadPrescalesFromFile", false);
  desc.add<edm::InputTag>("TriggerResultTag", edm::InputTag("TriggerResults", "", "HLT"));
  desc.add<edm::InputTag>("TriggerObjects", edm::InputTag("slimmedPatTrigger"));
  desc.add<edm::InputTag>("ElectronCollection", edm::InputTag("slimmedElectrons"));
  desc.add<edm::InputTag>("ScoutingElectronCollection", edm::InputTag("Run3ScoutingElectrons"));
  desc.add<edm::InputTag>("eleIdMapTight",
                          edm::InputTag("egmGsfElectronIDs:cutBasedElectronID-RunIIIWinter22-V1-tight"));
  descriptions.addWithDefaultLabel(desc);
}

bool PatElectronTagProbeAnalyzer::scoutingElectron_passHLT(const float el_eta,
                                                           const float el_phi,
                                                           const trigger::TriggerObjectCollection& legObjects) const {
  bool foundTheLeg = false;
  for (unsigned int i = 0; i < legObjects.size(); i++) {
    float delR2 = deltaR2(legObjects.at(i).eta(), legObjects.at(i).phi(), el_eta, el_phi);
    if (delR2 < 0.01) {
      foundTheLeg = true;
      break;
    }
  }
  return foundTheLeg;
}

bool PatElectronTagProbeAnalyzer::patElectron_passHLT(const pat::Electron& el,
                                                      const trigger::TriggerObjectCollection& legObjects) const {
  bool foundTheLeg = false;
  for (unsigned int i = 0; i < legObjects.size(); i++) {
    float delR2 = deltaR2(legObjects.at(i).eta(), legObjects.at(i).phi(), el.eta(), el.phi());
    if (delR2 < 0.01) {
      foundTheLeg = true;
      break;
    }
  }
  return foundTheLeg;
}

DEFINE_FWK_MODULE(PatElectronTagProbeAnalyzer);

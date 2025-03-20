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
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "ScoutingDQMUtils.h"

/////////////////////////
//  Class declaration  //
/////////////////////////

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
};

struct kTagProbeHistos {
  kProbeKinematicHistos resonanceZ_patElectron;
  kProbeKinematicHistos resonanceJ_patElectron;
  kProbeKinematicHistos resonanceY_patElectron;
  kProbeKinematicHistos resonanceAll_patElectron;
  kProbeKinematicHistos resonanceZ_sctElectron;
  kProbeKinematicHistos resonanceJ_sctElectron;
  kProbeKinematicHistos resonanceY_sctElectron;
  kProbeKinematicHistos resonanceAll_sctElectron;

  kProbeKinematicHistos resonanceZ_patElectron_passDoubleEG_DST;
  kProbeKinematicHistos resonanceJ_patElectron_passDoubleEG_DST;
  kProbeKinematicHistos resonanceY_patElectron_passDoubleEG_DST;
  kProbeKinematicHistos resonanceAll_patElectron_passDoubleEG_DST;
  kProbeKinematicHistos resonanceZ_sctElectron_passDoubleEG_DST;
  kProbeKinematicHistos resonanceJ_sctElectron_passDoubleEG_DST;
  kProbeKinematicHistos resonanceY_sctElectron_passDoubleEG_DST;
  kProbeKinematicHistos resonanceAll_sctElectron_passDoubleEG_DST;

  kProbeKinematicHistos resonanceZ_patElectron_passDoubleEG_DST_fireTrigObj;
  kProbeKinematicHistos resonanceJ_patElectron_passDoubleEG_DST_fireTrigObj;
  kProbeKinematicHistos resonanceY_patElectron_passDoubleEG_DST_fireTrigObj;
  kProbeKinematicHistos resonanceAll_patElectron_passDoubleEG_DST_fireTrigObj;
  kProbeKinematicHistos resonanceZ_sctElectron_passDoubleEG_DST_fireTrigObj;
  kProbeKinematicHistos resonanceJ_sctElectron_passDoubleEG_DST_fireTrigObj;
  kProbeKinematicHistos resonanceY_sctElectron_passDoubleEG_DST_fireTrigObj;
  kProbeKinematicHistos resonanceAll_sctElectron_passDoubleEG_DST_fireTrigObj;

  kProbeKinematicHistos resonanceZ_patElectron_passSinglePhoton_DST;
  kProbeKinematicHistos resonanceJ_patElectron_passSinglePhoton_DST;
  kProbeKinematicHistos resonanceY_patElectron_passSinglePhoton_DST;
  kProbeKinematicHistos resonanceAll_patElectron_passSinglePhoton_DST;
  kProbeKinematicHistos resonanceZ_sctElectron_passSinglePhoton_DST;
  kProbeKinematicHistos resonanceJ_sctElectron_passSinglePhoton_DST;
  kProbeKinematicHistos resonanceY_sctElectron_passSinglePhoton_DST;
  kProbeKinematicHistos resonanceAll_sctElectron_passSinglePhoton_DST;

  kProbeKinematicHistos resonanceZ_patElectron_passSinglePhoton_DST_fireTrigObj;
  kProbeKinematicHistos resonanceJ_patElectron_passSinglePhoton_DST_fireTrigObj;
  kProbeKinematicHistos resonanceY_patElectron_passSinglePhoton_DST_fireTrigObj;
  kProbeKinematicHistos resonanceAll_patElectron_passSinglePhoton_DST_fireTrigObj;
  kProbeKinematicHistos resonanceZ_sctElectron_passSinglePhoton_DST_fireTrigObj;
  kProbeKinematicHistos resonanceJ_sctElectron_passSinglePhoton_DST_fireTrigObj;
  kProbeKinematicHistos resonanceY_sctElectron_passSinglePhoton_DST_fireTrigObj;
  kProbeKinematicHistos resonanceAll_sctElectron_passSinglePhoton_DST_fireTrigObj;
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
                                const float inv_mass) const;

  void fillHistograms_resonance_sct(const kProbeKinematicHistos& histos,
                                    const Run3ScoutingElectron& el,
                                    const float inv_mass) const;

  bool scoutingElectron_passHLT(const float el_eta,
                                const float el_phi,
                                const trigger::TriggerObjectCollection& legObjects) const;

  bool patElectron_passHLT(const pat::Electron& el, const trigger::TriggerObjectCollection& legObjects) const;

  // --------------------- member data  ----------------------
  const std::string outputInternalPath_;
  const edm::EDGetToken triggerResultsToken_;
  const edm::EDGetTokenT<pat::TriggerObjectStandAloneCollection> triggerObjects_;
  const edm::EDGetTokenT<edm::View<pat::Electron>> electronCollection_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingElectron>> scoutingElectronCollection_;
  const edm::EDGetTokenT<edm::ValueMap<bool>> eleIdMapTightToken_;
};

using namespace ROOT;

PatElectronTagProbeAnalyzer::PatElectronTagProbeAnalyzer(const edm::ParameterSet& iConfig)
    : outputInternalPath_(iConfig.getParameter<std::string>("OutputInternalPath")),
      triggerResultsToken_(consumes<edm::TriggerResults>(iConfig.getParameter<edm::InputTag>("TriggerResultTag"))),
      triggerObjects_(
          consumes<pat::TriggerObjectStandAloneCollection>(iConfig.getParameter<edm::InputTag>("TriggerObjects"))),
      electronCollection_(
          consumes<edm::View<pat::Electron>>(iConfig.getParameter<edm::InputTag>("ElectronCollection"))),
      scoutingElectronCollection_(consumes<std::vector<Run3ScoutingElectron>>(
          iConfig.getParameter<edm::InputTag>("ScoutingElectronCollection"))),
      eleIdMapTightToken_(consumes<edm::ValueMap<bool>>(iConfig.getParameter<edm::InputTag>("eleIdMapTight"))) {}

void PatElectronTagProbeAnalyzer::dqmAnalyze(edm::Event const& iEvent,
                                             edm::EventSetup const& iSetup,
                                             kTagProbeHistos const& histos) const {
  edm::Handle<edm::View<pat::Electron>> patEls;
  iEvent.getByToken(electronCollection_, patEls);
  if (patEls.failedToGet()) {
    edm::LogWarning("ScoutingMonitoring") << "pat::Electron collection not found.";
    return;
  }

  edm::Handle<std::vector<Run3ScoutingElectron>> sctEls;
  iEvent.getByToken(scoutingElectronCollection_, sctEls);
  if (sctEls.failedToGet()) {
    edm::LogWarning("ScoutingMonitoring") << "Run3ScoutingElectron collection not found.";
    return;
  }

  edm::Handle<edm::ValueMap<bool>> tight_ele_id_decisions;
  iEvent.getByToken(eleIdMapTightToken_, tight_ele_id_decisions);

  edm::LogInfo("ScoutingMonitoring") << "Process pat::Electrons: " << patEls->size();
  edm::LogInfo("ScoutingMonitoring") << "Process Run3ScoutingElectrons: " << sctEls->size();

  // Trigger
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(triggerResultsToken_, triggerResults);
  const edm::TriggerNames& triggerNames = iEvent.triggerNames(*triggerResults);
  bool fire_singlePhoton_DST = scoutingDQMUtils::hasPatternInHLTPath(triggerNames, "DST_PFScouting_SinglePhotonEB");
  bool fire_doubleEG_DST = scoutingDQMUtils::hasPatternInHLTPath(triggerNames, "DST_PFScouting_DoubleEG");

  // Trigger Object
  edm::Handle<pat::TriggerObjectStandAloneCollection> triggerObjects;
  iEvent.getByToken(triggerObjects_, triggerObjects);

  std::vector<std::string> filterToMatch = {"hltDoubleEG11CaloIdLHEFilter", "hltEG30EBTightIDTightIsoTrackIsoFilter"};
  size_t numberOfFilters = filterToMatch.size();
  trigger::TriggerObjectCollection* legObjects = new trigger::TriggerObjectCollection[numberOfFilters];
  for (size_t iteFilter = 0; iteFilter < filterToMatch.size(); iteFilter++) {
    std::string filterTag = filterToMatch.at(iteFilter);
    for (pat::TriggerObjectStandAlone obj : *triggerObjects) {
      obj.unpackNamesAndLabels(iEvent, *triggerResults);
      if (obj.hasFilterLabel(filterTag)) {
        legObjects[iteFilter].push_back(obj);
      }
    }
  }

  std::vector<int> sctElectron_gsfTrackIndex;
  for (const auto& sct_el : *sctEls) {
    size_t gsfTrkIdx = 9999;
    bool foundGoodGsfTrkIdx = scoutingDQMUtils::scoutingElectronGsfTrackIdx(sct_el, gsfTrkIdx);
    if (foundGoodGsfTrkIdx)
      sctElectron_gsfTrackIndex.push_back(gsfTrkIdx);
    else
      sctElectron_gsfTrackIndex.push_back(-1);
  }

  // for (const auto& pat_el : *patEls){
  for (size_t i = 0; i < patEls->size(); ++i) {
    const auto pat_el = patEls->ptrAt(i);
    if (!((*tight_ele_id_decisions)[pat_el]))
      continue;

    ROOT::Math::PtEtaPhiMVector tag_pat_el(pat_el->pt(), pat_el->eta(), pat_el->phi(), pat_el->mass());
    for (size_t j = 0; j < patEls->size(); ++j) {
      //    for (const auto& pat_el_second : *patEls){
      const auto pat_el_second = patEls->ptrAt(j);
      if (i == j)
        continue;
      if (!((*tight_ele_id_decisions)[pat_el_second]))
        continue;
      ROOT::Math::PtEtaPhiMVector probe_pat_el(
          pat_el_second->pt(), pat_el_second->eta(), pat_el_second->phi(), pat_el_second->mass());
      float invMass = (tag_pat_el + probe_pat_el).mass();
      if ((TandP_Z_minMass < invMass) && (invMass < TandP_Z_maxMass)) {
        fillHistograms_resonance(histos.resonanceZ_patElectron, *pat_el_second, invMass);
        fillHistograms_resonance(histos.resonanceAll_patElectron, *pat_el_second, invMass);
      }
      if ((TandP_jpsi_minMass < invMass) && (invMass < TandP_jpsi_maxMass)) {
        fillHistograms_resonance(histos.resonanceJ_patElectron,
                                 *pat_el_second,
                                 invMass);  // J/Psi mass: 3.3 +/- 0.2 GeV
        fillHistograms_resonance(histos.resonanceAll_patElectron, *pat_el_second, invMass);
      }
      if ((TandP_ups_minMass < invMass) && (invMass < TandP_ups_maxMass)) {
        fillHistograms_resonance(histos.resonanceY_patElectron,
                                 *pat_el_second,
                                 invMass);  // Y mass: 9.8 +/- 0.4 GeV & 10.6 +/- 1 GeV
        fillHistograms_resonance(histos.resonanceAll_patElectron, *pat_el_second, invMass);
      }

      if (fire_singlePhoton_DST) {
        if ((TandP_Z_minMass < invMass) && (invMass < TandP_Z_maxMass)) {
          fillHistograms_resonance(histos.resonanceZ_patElectron_passSinglePhoton_DST, *pat_el_second, invMass);
          fillHistograms_resonance(histos.resonanceAll_patElectron_passSinglePhoton_DST, *pat_el_second, invMass);
        }
        if ((TandP_jpsi_minMass < invMass) && (invMass < TandP_jpsi_maxMass)) {
          fillHistograms_resonance(histos.resonanceJ_patElectron_passSinglePhoton_DST,
                                   *pat_el_second,
                                   invMass);  // J/Psi mass: 3.3 +/- 0.2 GeV
          fillHistograms_resonance(histos.resonanceAll_patElectron_passSinglePhoton_DST, *pat_el_second, invMass);
        }
        if ((TandP_ups_minMass < invMass) && (invMass < TandP_ups_maxMass)) {
          fillHistograms_resonance(histos.resonanceY_patElectron_passSinglePhoton_DST,
                                   *pat_el_second,
                                   invMass);  // Y mass: 9.8 +/- 0.4 GeV & 10.6 +/- 1 GeV
          fillHistograms_resonance(histos.resonanceAll_patElectron_passSinglePhoton_DST, *pat_el_second, invMass);
        }
      }

      if (patElectron_passHLT(*pat_el_second, legObjects[1])) {
        if ((TandP_Z_minMass < invMass) && (invMass < TandP_Z_maxMass)) {
          fillHistograms_resonance(
              histos.resonanceZ_patElectron_passSinglePhoton_DST_fireTrigObj, *pat_el_second, invMass);
          fillHistograms_resonance(
              histos.resonanceAll_patElectron_passSinglePhoton_DST_fireTrigObj, *pat_el_second, invMass);
        }
        if ((TandP_jpsi_minMass < invMass) && (invMass < TandP_jpsi_maxMass)) {
          fillHistograms_resonance(histos.resonanceJ_patElectron_passSinglePhoton_DST_fireTrigObj,
                                   *pat_el_second,
                                   invMass);  // J/Psi mass: 3.3 +/- 0.2 GeV
          fillHistograms_resonance(
              histos.resonanceAll_patElectron_passSinglePhoton_DST_fireTrigObj, *pat_el_second, invMass);
        }
        if ((TandP_ups_minMass < invMass) && (invMass < TandP_ups_maxMass)) {
          fillHistograms_resonance(histos.resonanceY_patElectron_passSinglePhoton_DST_fireTrigObj,
                                   *pat_el_second,
                                   invMass);  // Y mass: 9.8 +/- 0.4 GeV & 10.6 +/- 1 GeV
          fillHistograms_resonance(
              histos.resonanceAll_patElectron_passSinglePhoton_DST_fireTrigObj, *pat_el_second, invMass);
        }
      }

      if (fire_doubleEG_DST) {
        if ((TandP_Z_minMass < invMass) && (invMass < TandP_Z_maxMass)) {
          fillHistograms_resonance(histos.resonanceZ_patElectron_passDoubleEG_DST, *pat_el_second, invMass);
          fillHistograms_resonance(histos.resonanceAll_patElectron_passDoubleEG_DST, *pat_el_second, invMass);
        }
        if ((TandP_jpsi_minMass < invMass) && (invMass < TandP_jpsi_maxMass)) {
          fillHistograms_resonance(histos.resonanceJ_patElectron_passDoubleEG_DST,
                                   *pat_el_second,
                                   invMass);  // J/Psi mass: 3.3 +/- 0.2 GeV
          fillHistograms_resonance(histos.resonanceAll_patElectron_passDoubleEG_DST, *pat_el_second, invMass);
        }
        if ((TandP_ups_minMass < invMass) && (invMass < TandP_ups_maxMass)) {
          fillHistograms_resonance(histos.resonanceY_patElectron_passDoubleEG_DST,
                                   *pat_el_second,
                                   invMass);  // Y mass: 9.8 +/- 0.4 GeV & 10.6 +/- 1 GeV
          fillHistograms_resonance(histos.resonanceAll_patElectron_passDoubleEG_DST, *pat_el_second, invMass);
        }
      }

      if (patElectron_passHLT(*pat_el_second, legObjects[0])) {
        if ((TandP_Z_minMass < invMass) && (invMass < TandP_Z_maxMass)) {
          fillHistograms_resonance(histos.resonanceZ_patElectron_passDoubleEG_DST_fireTrigObj, *pat_el_second, invMass);
          fillHistograms_resonance(
              histos.resonanceAll_patElectron_passDoubleEG_DST_fireTrigObj, *pat_el_second, invMass);
        }
        if ((TandP_jpsi_minMass < invMass) && (invMass < TandP_jpsi_maxMass)) {
          fillHistograms_resonance(histos.resonanceJ_patElectron_passDoubleEG_DST_fireTrigObj,
                                   *pat_el_second,
                                   invMass);  // J/Psi mass: 3.3 +/- 0.2 GeV
          fillHistograms_resonance(
              histos.resonanceAll_patElectron_passDoubleEG_DST_fireTrigObj, *pat_el_second, invMass);
        }
        if ((TandP_ups_minMass < invMass) && (invMass < TandP_ups_maxMass)) {
          fillHistograms_resonance(histos.resonanceY_patElectron_passDoubleEG_DST_fireTrigObj,
                                   *pat_el_second,
                                   invMass);  // Y mass: 9.8 +/- 0.4 GeV & 10.6 +/- 1 GeV
          fillHistograms_resonance(
              histos.resonanceAll_patElectron_passDoubleEG_DST_fireTrigObj, *pat_el_second, invMass);
        }
      }
    }

    int sct_el_index = 0;
    for (const auto& sct_el_second : *sctEls) {
      int gsfTrackIndex = sctElectron_gsfTrackIndex[sct_el_index];
      sct_el_index += 1;
      if (gsfTrackIndex < 0)
        continue;

      ROOT::Math::PtEtaPhiMVector sctEl0(
          sct_el_second.pt(), sct_el_second.eta(), sct_el_second.phi(), sct_el_second.m());
      ROOT::Math::PtEtaPhiMVector probe_sct_el(scoutingDQMUtils::computePtFromEnergyMassEta(
                                                   sctEl0.energy(), 0.0005, sct_el_second.trketaMode()[gsfTrackIndex]),
                                               sct_el_second.trketaMode()[gsfTrackIndex],
                                               sct_el_second.trkphiMode()[gsfTrackIndex],
                                               0.0005);

      if (ROOT::Math::VectorUtil::DeltaR(probe_sct_el, tag_pat_el) < 0.1)
        continue;
      if (!scoutingDQMUtils::scoutingElectronID(sct_el_second))
        continue;

      float invMass = (tag_pat_el + probe_sct_el).mass();
      if ((TandP_Z_minMass < invMass) && (invMass < TandP_Z_maxMass)) {
        fillHistograms_resonance_sct(histos.resonanceZ_sctElectron, sct_el_second, invMass);
        fillHistograms_resonance_sct(histos.resonanceAll_sctElectron, sct_el_second, invMass);
      }
      if ((TandP_jpsi_minMass < invMass) && (invMass < TandP_jpsi_maxMass)) {
        fillHistograms_resonance_sct(histos.resonanceJ_sctElectron,
                                     sct_el_second,
                                     invMass);  // J/Psi mass: 3.3 +/- 0.2 GeV
        fillHistograms_resonance_sct(histos.resonanceAll_sctElectron, sct_el_second, invMass);
      }
      if ((TandP_ups_minMass < invMass) && (invMass < TandP_ups_maxMass)) {
        fillHistograms_resonance_sct(histos.resonanceY_sctElectron,
                                     sct_el_second,
                                     invMass);  // Y mass: 9.8 +/- 0.4 GeV & 10.6 +/- 1 GeV
        fillHistograms_resonance_sct(histos.resonanceAll_sctElectron, sct_el_second, invMass);
      }

      if (fire_singlePhoton_DST) {
        if ((TandP_Z_minMass < invMass) && (invMass < TandP_Z_maxMass)) {
          fillHistograms_resonance_sct(histos.resonanceZ_sctElectron_passSinglePhoton_DST, sct_el_second, invMass);
          fillHistograms_resonance_sct(histos.resonanceAll_sctElectron_passSinglePhoton_DST, sct_el_second, invMass);
        }
        if ((TandP_jpsi_minMass < invMass) && (invMass < TandP_jpsi_maxMass)) {
          fillHistograms_resonance_sct(histos.resonanceJ_sctElectron_passSinglePhoton_DST,
                                       sct_el_second,
                                       invMass);  // J/Psi mass: 3.3 +/- 0.2 GeV
          fillHistograms_resonance_sct(histos.resonanceAll_sctElectron_passSinglePhoton_DST, sct_el_second, invMass);
        }
        if ((TandP_ups_minMass < invMass) && (invMass < TandP_ups_maxMass)) {
          fillHistograms_resonance_sct(histos.resonanceY_sctElectron_passSinglePhoton_DST,
                                       sct_el_second,
                                       invMass);  // Y mass: 9.8 +/- 0.4 GeV & 10.6 +/- 1 GeV
          fillHistograms_resonance_sct(histos.resonanceAll_sctElectron_passSinglePhoton_DST, sct_el_second, invMass);
        }
      }

      if (scoutingElectron_passHLT(
              sct_el_second.trketaMode()[gsfTrackIndex], sct_el_second.trkphiMode()[gsfTrackIndex], legObjects[1])) {
        if ((TandP_Z_minMass < invMass) && (invMass < TandP_Z_maxMass)) {
          fillHistograms_resonance_sct(
              histos.resonanceZ_sctElectron_passSinglePhoton_DST_fireTrigObj, sct_el_second, invMass);
          fillHistograms_resonance_sct(
              histos.resonanceAll_sctElectron_passSinglePhoton_DST_fireTrigObj, sct_el_second, invMass);
        }
        if ((TandP_jpsi_minMass < invMass) && (invMass < TandP_jpsi_maxMass)) {
          fillHistograms_resonance_sct(histos.resonanceJ_sctElectron_passSinglePhoton_DST_fireTrigObj,
                                       sct_el_second,
                                       invMass);  // J/Psi mass: 3.3 +/- 0.2 GeV
          fillHistograms_resonance_sct(
              histos.resonanceAll_sctElectron_passSinglePhoton_DST_fireTrigObj, sct_el_second, invMass);
        }
        if ((TandP_ups_minMass < invMass) && (invMass < TandP_ups_maxMass)) {
          fillHistograms_resonance_sct(histos.resonanceY_sctElectron_passSinglePhoton_DST_fireTrigObj,
                                       sct_el_second,
                                       invMass);  // Y mass: 9.8 +/- 0.4 GeV & 10.6 +/- 1 GeV
          fillHistograms_resonance_sct(
              histos.resonanceAll_sctElectron_passSinglePhoton_DST_fireTrigObj, sct_el_second, invMass);
        }
      }

      if (fire_doubleEG_DST) {
        if ((TandP_Z_minMass < invMass) && (invMass < TandP_Z_maxMass)) {
          fillHistograms_resonance_sct(histos.resonanceZ_sctElectron_passDoubleEG_DST, sct_el_second, invMass);
          fillHistograms_resonance_sct(histos.resonanceAll_sctElectron_passDoubleEG_DST, sct_el_second, invMass);
        }
        if ((TandP_jpsi_minMass < invMass) && (invMass < TandP_jpsi_maxMass)) {
          fillHistograms_resonance_sct(histos.resonanceJ_sctElectron_passDoubleEG_DST,
                                       sct_el_second,
                                       invMass);  // J/Psi mass: 3.3 +/- 0.2 GeV
          fillHistograms_resonance_sct(histos.resonanceAll_sctElectron_passDoubleEG_DST, sct_el_second, invMass);
        }
        if ((TandP_ups_minMass < invMass) && (invMass < TandP_ups_maxMass)) {
          fillHistograms_resonance_sct(histos.resonanceY_sctElectron_passDoubleEG_DST,
                                       sct_el_second,
                                       invMass);  // Y mass: 9.8 +/- 0.4 GeV & 10.6 +/- 1 GeV
          fillHistograms_resonance_sct(histos.resonanceAll_sctElectron_passDoubleEG_DST, sct_el_second, invMass);
        }
      }

      if (scoutingElectron_passHLT(
              sct_el_second.trketaMode()[gsfTrackIndex], sct_el_second.trkphiMode()[gsfTrackIndex], legObjects[0])) {
        if ((TandP_Z_minMass < invMass) && (invMass < TandP_Z_maxMass)) {
          fillHistograms_resonance_sct(
              histos.resonanceZ_sctElectron_passDoubleEG_DST_fireTrigObj, sct_el_second, invMass);
          fillHistograms_resonance_sct(
              histos.resonanceAll_sctElectron_passDoubleEG_DST_fireTrigObj, sct_el_second, invMass);
        }
        if ((TandP_jpsi_minMass < invMass) && (invMass < TandP_jpsi_maxMass)) {
          fillHistograms_resonance_sct(histos.resonanceJ_sctElectron_passDoubleEG_DST_fireTrigObj,
                                       sct_el_second,
                                       invMass);  // J/Psi mass: 3.3 +/- 0.2 GeV
          fillHistograms_resonance_sct(
              histos.resonanceAll_sctElectron_passDoubleEG_DST_fireTrigObj, sct_el_second, invMass);
        }
        if ((TandP_ups_minMass < invMass) && (invMass < TandP_ups_maxMass)) {
          fillHistograms_resonance_sct(histos.resonanceY_sctElectron_passDoubleEG_DST_fireTrigObj,
                                       sct_el_second,
                                       invMass);  // Y mass: 9.8 +/- 0.4 GeV & 10.6 +/- 1 GeV
          fillHistograms_resonance_sct(
              histos.resonanceAll_sctElectron_passDoubleEG_DST_fireTrigObj, sct_el_second, invMass);
        }
      }
    }
  }
}

void PatElectronTagProbeAnalyzer::fillHistograms_resonance(const kProbeKinematicHistos& histos,
                                                           const pat::Electron& el,
                                                           const float inv_mass) const {
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
  }
}

void PatElectronTagProbeAnalyzer::fillHistograms_resonance_sct(const kProbeKinematicHistos& histos,
                                                               const Run3ScoutingElectron& el,
                                                               const float inv_mass) const {
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
  }
}
void PatElectronTagProbeAnalyzer::bookHistograms(DQMStore::IBooker& ibook,
                                                 edm::Run const& run,
                                                 edm::EventSetup const& iSetup,
                                                 kTagProbeHistos& histos) const {
  ibook.setCurrentFolder(outputInternalPath_);

  bookHistograms_resonance(ibook, run, iSetup, histos.resonanceZ_patElectron, "resonanceZ_Tag_pat_Probe_patElectron");
  bookHistograms_resonance(ibook, run, iSetup, histos.resonanceJ_patElectron, "resonanceJ_Tag_pat_Probe_patElectron");
  bookHistograms_resonance(ibook, run, iSetup, histos.resonanceY_patElectron, "resonanceY_Tag_pat_Probe_patElectron");
  bookHistograms_resonance(
      ibook, run, iSetup, histos.resonanceAll_patElectron, "resonanceAll_Tag_pat_Probe_patElectron");

  bookHistograms_resonance(ibook, run, iSetup, histos.resonanceZ_sctElectron, "resonanceZ_Tag_pat_Probe_sctElectron");
  bookHistograms_resonance(ibook, run, iSetup, histos.resonanceJ_sctElectron, "resonanceJ_Tag_pat_Probe_sctElectron");
  bookHistograms_resonance(ibook, run, iSetup, histos.resonanceY_sctElectron, "resonanceY_Tag_pat_Probe_sctElectron");
  bookHistograms_resonance(
      ibook, run, iSetup, histos.resonanceAll_sctElectron, "resonanceAll_Tag_pat_Probe_sctElectron");

  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceZ_patElectron_passSinglePhoton_DST,
                           "resonanceZ_Tag_pat_Probe_patElectron_passSinglePhoton_DST");
  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceJ_patElectron_passSinglePhoton_DST,
                           "resonanceJ_Tag_pat_Probe_patElectron_passSinglePhoton_DST");
  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceY_patElectron_passSinglePhoton_DST,
                           "resonanceY_Tag_pat_Probe_patElectron_passSinglePhoton_DST");
  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceAll_patElectron_passSinglePhoton_DST,
                           "resonanceAll_Tag_pat_Probe_patElectron_passSinglePhoton_DST");

  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceZ_sctElectron_passSinglePhoton_DST,
                           "resonanceZ_Tag_pat_Probe_sctElectron_passSinglePhoton_DST");
  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceJ_sctElectron_passSinglePhoton_DST,
                           "resonanceJ_Tag_pat_Probe_sctElectron_passSinglePhoton_DST");
  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceY_sctElectron_passSinglePhoton_DST,
                           "resonanceY_Tag_pat_Probe_sctElectron_passSinglePhoton_DST");
  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceAll_sctElectron_passSinglePhoton_DST,
                           "resonanceAll_Tag_pat_Probe_sctElectron_passSinglePhoton_DST");

  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceZ_patElectron_passSinglePhoton_DST_fireTrigObj,
                           "resonanceZ_Tag_pat_Probe_patElectron_passSinglePhoton_DST_fireTrigObj");
  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceJ_patElectron_passSinglePhoton_DST_fireTrigObj,
                           "resonanceJ_Tag_pat_Probe_patElectron_passSinglePhoton_DST_fireTrigObj");
  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceY_patElectron_passSinglePhoton_DST_fireTrigObj,
                           "resonanceY_Tag_pat_Probe_patElectron_passSinglePhoton_DST_fireTrigObj");
  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceAll_patElectron_passSinglePhoton_DST_fireTrigObj,
                           "resonanceAll_Tag_pat_Probe_patElectron_passSinglePhoton_DST_"
                           "fireTrigObj");

  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceZ_sctElectron_passSinglePhoton_DST_fireTrigObj,
                           "resonanceZ_Tag_pat_Probe_sctElectron_passSinglePhoton_DST_fireTrigObj");
  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceJ_sctElectron_passSinglePhoton_DST_fireTrigObj,
                           "resonanceJ_Tag_pat_Probe_sctElectron_passSinglePhoton_DST_fireTrigObj");
  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceY_sctElectron_passSinglePhoton_DST_fireTrigObj,
                           "resonanceY_Tag_pat_Probe_sctElectron_passSinglePhoton_DST_fireTrigObj");
  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceAll_sctElectron_passSinglePhoton_DST_fireTrigObj,
                           "resonanceAll_Tag_pat_Probe_sctElectron_passSinglePhoton_DST_"
                           "fireTrigObj");

  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceZ_patElectron_passDoubleEG_DST,
                           "resonanceZ_Tag_pat_Probe_patElectron_passDoubleEG_DST");
  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceJ_patElectron_passDoubleEG_DST,
                           "resonanceJ_Tag_pat_Probe_patElectron_passDoubleEG_DST");
  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceY_patElectron_passDoubleEG_DST,
                           "resonanceY_Tag_pat_Probe_patElectron_passDoubleEG_DST");
  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceAll_patElectron_passDoubleEG_DST,
                           "resonanceAll_Tag_pat_Probe_patElectron_passDoubleEG_DST");

  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceZ_sctElectron_passDoubleEG_DST,
                           "resonanceZ_Tag_pat_Probe_sctElectron_passDoubleEG_DST");
  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceJ_sctElectron_passDoubleEG_DST,
                           "resonanceJ_Tag_pat_Probe_sctElectron_passDoubleEG_DST");
  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceY_sctElectron_passDoubleEG_DST,
                           "resonanceY_Tag_pat_Probe_sctElectron_passDoubleEG_DST");
  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceAll_sctElectron_passDoubleEG_DST,
                           "resonanceAll_Tag_pat_Probe_sctElectron_passDoubleEG_DST");

  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceZ_patElectron_passDoubleEG_DST_fireTrigObj,
                           "resonanceZ_Tag_pat_Probe_patElectron_passDoubleEG_DST_fireTrigObj");
  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceJ_patElectron_passDoubleEG_DST_fireTrigObj,
                           "resonanceJ_Tag_pat_Probe_patElectron_passDoubleEG_DST_fireTrigObj");
  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceY_patElectron_passDoubleEG_DST_fireTrigObj,
                           "resonanceY_Tag_pat_Probe_patElectron_passDoubleEG_DST_fireTrigObj");
  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceAll_patElectron_passDoubleEG_DST_fireTrigObj,
                           "resonanceAll_Tag_pat_Probe_patElectron_passDoubleEG_DST_fireTrigObj");

  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceZ_sctElectron_passDoubleEG_DST_fireTrigObj,
                           "resonanceZ_Tag_pat_Probe_sctElectron_passDoubleEG_DST_fireTrigObj");
  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceJ_sctElectron_passDoubleEG_DST_fireTrigObj,
                           "resonanceJ_Tag_pat_Probe_sctElectron_passDoubleEG_DST_fireTrigObj");
  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceY_sctElectron_passDoubleEG_DST_fireTrigObj,
                           "resonanceY_Tag_pat_Probe_sctElectron_passDoubleEG_DST_fireTrigObj");
  bookHistograms_resonance(ibook,
                           run,
                           iSetup,
                           histos.resonanceAll_sctElectron_passDoubleEG_DST_fireTrigObj,
                           "resonanceAll_Tag_pat_Probe_sctElectron_passDoubleEG_DST_fireTrigObj");
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
}

void PatElectronTagProbeAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("OutputInternalPath", "MY_FOLDER");
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

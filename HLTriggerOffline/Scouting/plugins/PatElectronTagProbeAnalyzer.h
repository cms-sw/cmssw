#ifndef DQMOffline_Scouting_PatElectronTagProbeAnalyzer_h
#define DQMOffline_Scouting_PatElectronTagProbeAnalyzer_h

#include <string>
#include <vector>

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
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

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
  kProbeKinematicHistos
      resonanceAll_patElectron_passSinglePhoton_DST_fireTrigObj;
  kProbeKinematicHistos resonanceZ_sctElectron_passSinglePhoton_DST_fireTrigObj;
  kProbeKinematicHistos resonanceJ_sctElectron_passSinglePhoton_DST_fireTrigObj;
  kProbeKinematicHistos resonanceY_sctElectron_passSinglePhoton_DST_fireTrigObj;
  kProbeKinematicHistos
      resonanceAll_sctElectron_passSinglePhoton_DST_fireTrigObj;
};

class PatElectronTagProbeAnalyzer
    : public DQMGlobalEDAnalyzer<kTagProbeHistos> {
 public:
  explicit PatElectronTagProbeAnalyzer(const edm::ParameterSet& conf);
  ~PatElectronTagProbeAnalyzer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  // Constants
  static constexpr double TandP_Z_minMass =
      80;  // Lower bound for Tag And Probe at the Z peak
  static constexpr double TandP_Z_maxMass =
      100;  // Higher bound for Tag And Probe at the Z peak
  static constexpr double TandP_ups_minMass =
      9.0;  // Lower bound for Tag And Probe at the Upsilon peak
  static constexpr double TandP_ups_maxMass =
      12.6;  // Higher bound for Tag And Probe at the Upsilon peak
  static constexpr double TandP_jpsi_minMass =
      2.8;  // Lower bound for Tag And Probe at the JPsi peak
  static constexpr double TandP_jpsi_maxMass =
      3.8;  // Higher bound for Tag And Probe at the JPsi peak

 private:
  void dqmAnalyze(const edm::Event& e, const edm::EventSetup& c,
                  kTagProbeHistos const&) const override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&,
                      edm::EventSetup const&, kTagProbeHistos&) const override;
  void bookHistograms_resonance(DQMStore::IBooker&, edm::Run const&,
                                edm::EventSetup const&, kProbeKinematicHistos&,
                                const std::string&) const;
  void fillHistograms_resonance(const kProbeKinematicHistos histos,
                                const pat::Electron el,
                                const float inv_mass) const;
  void fillHistograms_resonance_sct(const kProbeKinematicHistos histos,
                                    const Run3ScoutingElectron el,
                                    const float inv_mass) const;
  double computePtFromEnergyMassEta(double energy, double mass,
                                    double eta) const;
  bool scoutingElectronID(const Run3ScoutingElectron el) const;
  bool scoutingElectronGsfTrackID(const Run3ScoutingElectron el,
                                  size_t trackIdx) const;
  bool scoutingElectronGsfTrackIdx(const Run3ScoutingElectron el,
                                   size_t& trackIdx) const;
  bool hasPatternInHLTPath(const edm::TriggerNames& triggerNames,
                           const std::string& pattern) const;
  bool scoutingElectron_passHLT(
      float el_eta, float el_phi, TString filter,
      trigger::TriggerObjectCollection legObjects) const;
  bool patElectron_passHLT(const pat::Electron el, TString filter,
                           trigger::TriggerObjectCollection legObjects) const;
  // --------------------- member data  ----------------------
  std::string outputInternalPath_;
  edm::EDGetToken triggerResultsToken_;
  const edm::EDGetTokenT<pat::TriggerObjectStandAloneCollection>
      triggerObjects_;
  const edm::EDGetTokenT<edm::View<pat::Electron>> electronCollection_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingElectron>>
      scoutingElectronCollection_;
  const edm::EDGetTokenT<edm::ValueMap<bool>> eleIdMapTightToken_;
};

#endif

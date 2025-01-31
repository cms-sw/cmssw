#ifndef DQMOffline_Scouting_ScoutingElectronTagProbeAnalyzer_h
#define DQMOffline_Scouting_ScoutingElectronTagProbeAnalyzer_h

#include <string>
#include <vector>

// user include files
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/Scouting/interface/Run3ScoutingElectron.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// HLT related header files
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/PatCandidates/interface/PackedTriggerPrescales.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"

// L1 related header files
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Muon.h"

/////////////////////////
//  Class declaration  //
/////////////////////////

struct kSctProbeKinematicHistos {
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
  dqm::reco::MonitorElement* hInvMass;
  dqm::reco::MonitorElement* hPt_Barrel_passID;
  dqm::reco::MonitorElement* hPt_Endcap_passID;
  dqm::reco::MonitorElement* hPt_Barrel_passDSTsingleEG;
  dqm::reco::MonitorElement* hPt_Endcap_passDSTsingleEG;
  dqm::reco::MonitorElement* hPt_Barrel_passDSTdoubleEG;
  dqm::reco::MonitorElement* hPt_Endcap_passDSTdoubleEG;
};

struct kSctTagProbeHistos {
  kSctProbeKinematicHistos resonanceZ;
  kSctProbeKinematicHistos resonanceJ;
  kSctProbeKinematicHistos resonanceY;
  kSctProbeKinematicHistos resonanceAll;
};

class ScoutingElectronTagProbeAnalyzer : public DQMGlobalEDAnalyzer<kSctTagProbeHistos> {
public:
  explicit ScoutingElectronTagProbeAnalyzer(const edm::ParameterSet& conf);
  ~ScoutingElectronTagProbeAnalyzer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void dqmAnalyze(const edm::Event& e, const edm::EventSetup& c, kSctTagProbeHistos const&) const override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, kSctTagProbeHistos&) const override;
  void bookHistograms_resonance(
      DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, kSctProbeKinematicHistos&, const std::string&) const;
  void fillHistograms_resonance(const kSctProbeKinematicHistos histos,
                                const Run3ScoutingElectron el,
                                const float inv_mass,
                                const trigger::TriggerObjectCollection* legObjectsCollection) const;
  bool scoutingElectronID(const Run3ScoutingElectron el) const;
  bool scoutingElectron_passHLT(const Run3ScoutingElectron el,
                                TString filter,
                                trigger::TriggerObjectCollection legObjects) const;
  // --------------------- member data  ----------------------
  std::string outputInternalPath_;

  edm::EDGetToken triggerResultsToken_;
  edm::EDGetToken triggerSummaryToken_;
  edm::EDGetTokenT<pat::TriggerObjectStandAloneCollection> triggerObjects_;
  std::vector<std::string> filterToMatch_;

  edm::EDGetTokenT<std::vector<pat::Electron>> electronCollection_;
  edm::EDGetTokenT<std::vector<Run3ScoutingElectron>> scoutingElectronCollection_;
};

#endif

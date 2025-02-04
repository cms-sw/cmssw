/*
Scouting EGamma DQM core implementation.

 Description: ScoutingEGammaCollectionMonitoring is developed to enable us to
monitor the comparison between pat::Object and Run3Scouting<Object>.

 Implementation:
     * Current runs on top of MINIAOD dataformat of the
ScoutingEGammaCollectionMonitoring dataset.
     * Implemented only for electrons as of now.

Authors: Ting-Hsiang Hsu, Abanti Ranadhir Sahasransu
*/

#include "ScoutingElectronTagProbeAnalyzer.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ScoutingElectronTagProbeAnalyzer::ScoutingElectronTagProbeAnalyzer(const edm::ParameterSet& iConfig)
    : outputInternalPath_(iConfig.getParameter<std::string>("OutputInternalPath")),
      triggerResultsToken_(consumes<edm::TriggerResults>(iConfig.getParameter<edm::InputTag>("TriggerResultTag"))),
      triggerObjects_(
          consumes<pat::TriggerObjectStandAloneCollection>(iConfig.getParameter<edm::InputTag>("TriggerObjects"))),
      filterToMatch_(iConfig.getParameter<std::vector<std::string>>("FilterToMatch")),
      electronCollection_(
          consumes<std::vector<pat::Electron>>(iConfig.getParameter<edm::InputTag>("ElectronCollection"))),
      scoutingElectronCollection_(consumes<std::vector<Run3ScoutingElectron>>(
          iConfig.getParameter<edm::InputTag>("ScoutingElectronCollection"))) {}

ScoutingElectronTagProbeAnalyzer::~ScoutingElectronTagProbeAnalyzer() {}

void ScoutingElectronTagProbeAnalyzer::dqmAnalyze(edm::Event const& iEvent,
                                                  edm::EventSetup const& iSetup,
                                                  kSctTagProbeHistos const& histos) const {
  edm::Handle<std::vector<pat::Electron>> patEls;
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

  // Trigger
  edm::Handle<edm::TriggerResults> triggerResults;
  edm::Handle<pat::TriggerObjectStandAloneCollection> triggerObjects;
  iEvent.getByToken(triggerResultsToken_, triggerResults);
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

  edm::LogInfo("ScoutingMonitoring") << "Process pat::Electrons: " << patEls->size();
  edm::LogInfo("ScoutingMonitoring") << "Process Run3ScoutingElectrons: " << sctEls->size();

  for (const auto& sct_el : *sctEls) {
    if (!scoutingElectronID(sct_el))
      continue;
    edm::LogInfo("ScoutingElectronTagProbeAnalyzer") << "Process Run3ScoutingElectrons: " << sct_el.sigmaIetaIeta();

    math::PtEtaPhiMLorentzVector tag_sct_el(sct_el.pt(), sct_el.eta(), sct_el.phi(), sct_el.m());
    for (const auto& sct_el_second : *sctEls) {
      if (&sct_el_second == &sct_el)
        continue;
      math::PtEtaPhiMLorentzVector probe_sct_el(
          sct_el_second.pt(), sct_el_second.eta(), sct_el_second.phi(), sct_el_second.m());
      float invMass = (tag_sct_el + probe_sct_el).mass();
      edm::LogInfo("ScoutingMonitoring") << "Inv Mass: " << invMass;
      if ((80 < invMass) && (invMass < 100)) {
        fillHistograms_resonance(histos.resonanceZ, sct_el_second, invMass, legObjects);
        fillHistograms_resonance(histos.resonanceAll, sct_el_second, invMass, legObjects);
      }
      if ((2.8 < invMass) && (invMass < 3.8)) {
        fillHistograms_resonance(histos.resonanceJ, sct_el_second, invMass,
                                 legObjects);  // J/Psi mass: 3.3 +/- 0.2 GeV
        fillHistograms_resonance(histos.resonanceAll, sct_el_second, invMass, legObjects);
      }
      if ((9.0 < invMass) && (invMass < 12.6)) {
        fillHistograms_resonance(histos.resonanceY,
                                 sct_el_second,
                                 invMass,
                                 legObjects);  // Y mass: 9.8 +/- 0.4 GeV & 10.6 +/- 1 GeV
        fillHistograms_resonance(histos.resonanceAll, sct_el_second, invMass, legObjects);
      }
    }
  }
}

bool ScoutingElectronTagProbeAnalyzer::scoutingElectron_passHLT(const Run3ScoutingElectron el,
                                                                TString filter,
                                                                trigger::TriggerObjectCollection legObjects) const {
  bool foundTheLeg = false;
  for (unsigned int i = 0; i < legObjects.size(); i++) {
    float delR = deltaR(legObjects.at(i).eta(), legObjects.at(i).phi(), el.eta(), el.phi());
    if (delR < 0.1) {
      foundTheLeg = true;
      break;
    }
  }
  return foundTheLeg;
}

bool ScoutingElectronTagProbeAnalyzer::scoutingElectronID(const Run3ScoutingElectron el) const {
  math::PtEtaPhiMLorentzVector particle(el.pt(), el.eta(), el.phi(), 0.0005);
  double particle_energy = particle.energy();
  bool isEB = (fabs(el.eta()) < 1.479);
  if (isEB) {
    if (el.sigmaIetaIeta() > 0.015)
      return false;
    if (el.hOverE() > 0.2)
      return false;
    if (fabs(el.dEtaIn()) > 0.008)
      return false;
    if (fabs(el.dPhiIn()) > 0.06)
      return false;
    if (el.ecalIso() / particle_energy > 0.25)
      return false;
    return true;

  } else {
    if (el.sigmaIetaIeta() > 0.045)
      return false;
    if (el.hOverE() > 0.2)
      return false;
    if (fabs(el.dEtaIn()) > 0.012)
      return false;
    if (fabs(el.dPhiIn()) > 0.06)
      return false;
    if (el.ecalIso() / particle_energy > 0.1)
      return false;
    return true;
  }
}

void ScoutingElectronTagProbeAnalyzer::fillHistograms_resonance(
    const kSctProbeKinematicHistos histos,
    const Run3ScoutingElectron el,
    const float inv_mass,
    const trigger::TriggerObjectCollection* legObjectsCollection) const {
  histos.hEta->Fill(el.eta());
  histos.hPhi->Fill(el.phi());
  histos.hInvMass->Fill(inv_mass);

  if (fabs(el.eta()) < 1.5) {
    histos.hPt_Barrel->Fill(el.pt());
    if (scoutingElectronID(el))
      histos.hPt_Barrel_passID->Fill(el.pt());
    if (scoutingElectron_passHLT(el, "hltDoubleEG16EG12CaloIdLHEFilter", legObjectsCollection[0]))
      histos.hPt_Barrel_passDSTdoubleEG->Fill(el.pt());
    if (scoutingElectron_passHLT(el, "hltSingleEG30CaloIdLHEFilter", legObjectsCollection[1]))
      histos.hPt_Barrel_passDSTsingleEG->Fill(el.pt());
    histos.hHoverE_Barrel->Fill(el.hOverE());
    histos.hOoEMOoP_Barrel->Fill(el.ooEMOop());
    histos.hdPhiIn_Barrel->Fill(fabs(el.dPhiIn()));
    histos.hdEtaIn_Barrel->Fill(fabs(el.dEtaIn()));
    histos.hSigmaIetaIeta_Barrel->Fill(el.sigmaIetaIeta());
    histos.hMissingHits_Barrel->Fill(el.missingHits());
    histos.hTrackfbrem_Barrel->Fill(el.trackfbrem());
    histos.hRelEcalIsolation_Barrel->Fill(el.ecalIso() / el.pt());
    histos.hRelHcalIsolation_Barrel->Fill(el.hcalIso() / el.pt());
    histos.hRelTrackIsolation_Barrel->Fill(el.trackIso() / el.pt());
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
    if (scoutingElectronID(el))
      histos.hPt_Endcap_passID->Fill(el.pt());
    if (scoutingElectron_passHLT(el, "hltDoubleEG16EG12CaloIdLHEFilter", legObjectsCollection[0]))
      histos.hPt_Endcap_passDSTdoubleEG->Fill(el.pt());
    if (scoutingElectron_passHLT(el, "hltSingleEG30CaloIdLHEFilter", legObjectsCollection[1]))
      histos.hPt_Endcap_passDSTsingleEG->Fill(el.pt());
    histos.hHoverE_Endcap->Fill(el.hOverE());
    histos.hOoEMOoP_Endcap->Fill(el.ooEMOop());
    histos.hdPhiIn_Endcap->Fill(fabs(el.dPhiIn()));
    histos.hdEtaIn_Endcap->Fill(fabs(el.dEtaIn()));
    histos.hSigmaIetaIeta_Endcap->Fill(el.sigmaIetaIeta());
    histos.hMissingHits_Endcap->Fill(el.missingHits());
    histos.hTrackfbrem_Endcap->Fill(el.trackfbrem());
    histos.hRelEcalIsolation_Endcap->Fill(el.ecalIso() / el.pt());
    histos.hRelHcalIsolation_Endcap->Fill(el.hcalIso() / el.pt());
    histos.hRelTrackIsolation_Endcap->Fill(el.trackIso() / el.pt());
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

void ScoutingElectronTagProbeAnalyzer::bookHistograms(DQMStore::IBooker& ibook,
                                                      edm::Run const& run,
                                                      edm::EventSetup const& iSetup,
                                                      kSctTagProbeHistos& histos) const {
  ibook.setCurrentFolder(outputInternalPath_);
  bookHistograms_resonance(ibook, run, iSetup, histos.resonanceZ, "resonanceZ");
  bookHistograms_resonance(ibook, run, iSetup, histos.resonanceJ, "resonanceJ");
  bookHistograms_resonance(ibook, run, iSetup, histos.resonanceY, "resonanceY");
  bookHistograms_resonance(ibook, run, iSetup, histos.resonanceAll, "resonanceAll");
}

void ScoutingElectronTagProbeAnalyzer::bookHistograms_resonance(DQMStore::IBooker& ibook,
                                                                edm::Run const& run,
                                                                edm::EventSetup const& iSetup,
                                                                kSctProbeKinematicHistos& histos,
                                                                const std::string& name) const {
  ibook.setCurrentFolder(outputInternalPath_);
  histos.hPt_Barrel =
      ibook.book1D(name + "_Probe_sctElectron_Pt_Barrel", name + "_Probe_sctElectron_Pt_Barrel", 500, 0, 500);
  histos.hPt_Endcap =
      ibook.book1D(name + "_Probe_sctElectron_Pt_Endcap", name + "_Probe_sctElectron_Pt_Endcap", 500, 0, 500);

  histos.hEta = ibook.book1D(name + "_Probe_sctElectron_Eta", name + "_Probe_sctElectron_Eta", 100, -5.0, 5.0);
  histos.hPhi = ibook.book1D(name + "_Probe_sctElectron_Phi", name + "_Probe_sctElectron_Phi", 100, -3.3, 3.3);

  histos.hHoverE_Barrel =
      ibook.book1D(name + "_Probe_sctElectron_HoverE_Barrel", name + "_Probe_sctElectron_HoverE_Barrel", 500, 0, 0.5);
  histos.hHoverE_Endcap =
      ibook.book1D(name + "_Probe_sctElectron_HoverE_Endcap", name + "_Probe_sctElectron_HoverE_Endcap", 500, 0, 0.5);

  histos.hOoEMOoP_Barrel = ibook.book1D(
      name + "_Probe_sctElectron_OoEMOoP_Barrel", name + "_Probe_sctElectron_OoEMOoP_Barrel", 500, -0.2, 0.2);
  histos.hOoEMOoP_Endcap = ibook.book1D(
      name + "_Probe_sctElectron_OoEMOoP_Endcap", name + "_Probe_sctElectron_OoEMOoP_Endcap", 500, -0.2, 0.2);

  histos.hdPhiIn_Barrel =
      ibook.book1D(name + "_Probe_sctElectron_dPhiIn_Barrel", name + "_Probe_sctElectron_dPhiIn_Barrel", 100, 0, 0.1);
  histos.hdPhiIn_Endcap =
      ibook.book1D(name + "_Probe_sctElectron_dPhiIn_Endcap", name + "_Probe_sctElectron_dPhiIn_Endcap", 100, 0, 0.1);

  histos.hdEtaIn_Barrel =
      ibook.book1D(name + "_Probe_sctElectron_dEtaIn_Barrel", name + "_Probe_sctElectron_dEtaIn_Barrel", 100, 0, 0.1);
  histos.hdEtaIn_Endcap =
      ibook.book1D(name + "_Probe_sctElectron_dEtaIn_Endcap", name + "_Probe_sctElectron_dEtaIn_Endcap", 100, 0, 0.1);

  histos.hSigmaIetaIeta_Barrel = ibook.book1D(
      name + "_Probe_sctElectron_SigmaIetaIeta_Barrel", name + "_Probe_sctElectron_SigmaIetaIeta_Barrel", 500, 0, 0.05);
  histos.hSigmaIetaIeta_Endcap = ibook.book1D(
      name + "_Probe_sctElectron_SigmaIetaIeta_Endcap", name + "_Probe_sctElectron_SigmaIetaIeta_Endcap", 500, 0, 0.05);

  histos.hMissingHits_Barrel = ibook.book1D(
      name + "_Probe_sctElectron_MissingHits_Barrel", name + "_Probe_sctElectron_MissingHits_Barrel", 21, -0.5, 20.5);
  histos.hMissingHits_Endcap = ibook.book1D(
      name + "_Probe_sctElectron_MissingHits_Endcap", name + "_Probe_sctElectron_MissingHits_Endcap", 21, -0.5, 20.5);

  histos.hTrackfbrem_Barrel = ibook.book1D(
      name + "_Probe_sctElectron_Trackfbrem_Barrel", name + "_Probe_sctElectron_Trackfbrem_Barrel", 100, 0, 1.0);
  histos.hTrackfbrem_Endcap = ibook.book1D(
      name + "_Probe_sctElectron_Trackfbrem_Endcap", name + "_Probe_sctElectron_Trackfbrem_Endcap", 100, 0, 1.0);

  histos.hTrack_pt_Barrel = ibook.book1D(
      name + "_Probe_sctElectron_Track_pt_Barrel", name + "_Probe_sctElectron_Track_pt_Barrel", 200, 0, 100.0);
  histos.hTrack_pt_Endcap = ibook.book1D(
      name + "_Probe_sctElectron_Track_pt_Endcap", name + "_Probe_sctElectron_Track_pt_Endcap", 200, 0, 100.0);

  histos.hTrack_pMode_Barrel = ibook.book1D(
      name + "_Probe_sctElectron_Track_pMode_Barrel", name + "_Probe_sctElectron_Track_pMode_Barrel", 50, -0.5, 49.5);
  histos.hTrack_pMode_Endcap = ibook.book1D(
      name + "_Probe_sctElectron_Track_pMode_Endcap", name + "_Probe_sctElectron_Track_pMode_Endcap", 50, -0.5, 49.5);

  histos.hTrack_etaMode_Barrel = ibook.book1D(
      name + "_Probe_sctElectron_Track_etaMode_Barrel", name + "_Probe_sctElectron_Track_etaMode_Barrel", 26, -6.5, 6.5);
  histos.hTrack_etaMode_Endcap = ibook.book1D(
      name + "_Probe_sctElectron_Track_etaMode_Endcap", name + "_Probe_sctElectron_Track_etaMode_Endcap", 26, -6.5, 6.5);

  histos.hTrack_phiMode_Barrel = ibook.book1D(
      name + "_Probe_sctElectron_Track_phiMode_Barrel", name + "_Probe_sctElectron_Track_phiMode_Barrel", 18, -4.5, 4.5);
  histos.hTrack_phiMode_Endcap = ibook.book1D(
      name + "_Probe_sctElectron_Track_phiMode_Endcap", name + "_Probe_sctElectron_Track_phiMode_Endcap", 18, -4.5, 4.5);

  histos.hTrack_qoverpModeError_Barrel = ibook.book1D(name + "_Probe_sctElectron_Track_qoverpModeError_Barrel",
                                                      name + "_Probe_sctElectron_Track_qoverpModeError_Barrel",
                                                      36,
                                                      -4.5,
                                                      4.5);
  histos.hTrack_qoverpModeError_Endcap = ibook.book1D(name + "_Probe_sctElectron_Track_qoverpModeError_Endcap",
                                                      name + "_Probe_sctElectron_Track_qoverpModeError_Endcap",
                                                      36,
                                                      -4.5,
                                                      4.5);

  histos.hRelEcalIsolation_Barrel = ibook.book1D(name + "_Probe_sctElectron_RelEcalIsolation_Barrel",
                                                 name + "_Probe_sctElectron_RelEcalIsolation_Barrel",
                                                 100,
                                                 0,
                                                 1.0);
  histos.hRelEcalIsolation_Endcap = ibook.book1D(name + "_Probe_sctElectron_RelEcalIsolation_Endcap",
                                                 name + "_Probe_sctElectron_RelEcalIsolation_Endcap",
                                                 100,
                                                 0,
                                                 1.0);

  histos.hRelHcalIsolation_Barrel = ibook.book1D(name + "_Probe_sctElectron_RelHcalIsolation_Barrel",
                                                 name + "_Probe_sctElectron_RelHcalIsolation_Barrel",
                                                 100,
                                                 0,
                                                 1.0);
  histos.hRelHcalIsolation_Endcap = ibook.book1D(name + "_Probe_sctElectron_RelHcalIsolation_Endcap",
                                                 name + "_Probe_sctElectron_RelHcalIsolation_Endcap",
                                                 100,
                                                 0,
                                                 1.0);

  histos.hRelTrackIsolation_Barrel = ibook.book1D(name + "_Probe_sctElectron_RelTrackIsolation_Barrel",
                                                  name + "_Probe_sctElectron_RelTrackIsolation_Barrel",
                                                  100,
                                                  0,
                                                  1.0);
  histos.hRelTrackIsolation_Endcap = ibook.book1D(name + "_Probe_sctElectron_RelTrackIsolation_Endcap",
                                                  name + "_Probe_sctElectron_RelTrackIsolation_Endcap",
                                                  100,
                                                  0,
                                                  1.0);
  histos.hInvMass =
      ibook.book1D(name + "_sctElectron_Invariant_Mass", name + "_sctElectron_Invariant_Mass", 800, 0, 200);

  histos.hPt_Barrel_passID = ibook.book1D(
      name + "_Probe_sctElectron_Pt_Barrel_passID", name + "_Probe_sctElectron_Pt_Barrel_passID", 500, 0, 500);
  histos.hPt_Endcap_passID = ibook.book1D(
      name + "_Probe_sctElectron_Pt_Endcap_passID", name + "_Probe_sctElectron_Pt_Endcap_passID", 500, 0, 500);
  histos.hPt_Barrel_passDSTsingleEG = ibook.book1D(name + "_Probe_sctElectron_Pt_Barrel_passDSTsingleEG",
                                                   name + "_Probe_sctElectron_Pt_Barrel_passDSTsingleEG",
                                                   500,
                                                   0,
                                                   500);
  histos.hPt_Endcap_passDSTsingleEG = ibook.book1D(name + "_Probe_sctElectron_Pt_Endcap_passDSTsingleEG",
                                                   name + "_Probe_sctElectron_Pt_Endcap_passDSTsingleEG",
                                                   500,
                                                   0,
                                                   500);
  histos.hPt_Barrel_passDSTdoubleEG = ibook.book1D(name + "_Probe_sctElectron_Pt_Barrel_passDSTdoubleEG",
                                                   name + "_Probe_sctElectron_Pt_Barrel_passDSTdoubleEG",
                                                   500,
                                                   0,
                                                   500);
  histos.hPt_Endcap_passDSTdoubleEG = ibook.book1D(name + "_Probe_sctElectron_Pt_Endcap_passDSTdoubleEG",
                                                   name + "_Probe_sctElectron_Pt_Endcap_passDSTdoubleEG",
                                                   500,
                                                   0,
                                                   500);
}

// ------------ method fills 'descriptions' with the allowed parameters for the
// module  ------------
void ScoutingElectronTagProbeAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("OutputInternalPath", "MY_FOLDER");
  desc.add<edm::InputTag>("TriggerResultTag", edm::InputTag("TriggerResults", "", "HLT"));
  desc.add<std::vector<std::string>>("FilterToMatch",
                                     std::vector<std::string>{"hltPreDSTHLTMuonRun3PFScoutingPixelTracking"});
  desc.add<edm::InputTag>("TriggerObjects", edm::InputTag("slimmedPatTrigger"));
  desc.add<edm::InputTag>("ElectronCollection", edm::InputTag("slimmedElectrons"));
  desc.add<edm::InputTag>("ScoutingElectronCollection", edm::InputTag("Run3ScoutingElectrons"));
  descriptions.add("ScoutingElectronTagProbeAnalyzer", desc);
}

DEFINE_FWK_MODULE(ScoutingElectronTagProbeAnalyzer);

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

#include "ScoutingEGammaCollectionMonitoring.h"

#include <algorithm>
#include <numeric>

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ScoutingEGammaCollectionMonitoring::ScoutingEGammaCollectionMonitoring(const edm::ParameterSet& iConfig)
    : outputInternalPath_(iConfig.getParameter<std::string>("OutputInternalPath")),
      triggerResultsToken_(consumes<edm::TriggerResults>(iConfig.getParameter<edm::InputTag>("TriggerResultTag"))),
      electronCollection_(
          consumes<edm::View<pat::Electron>>(iConfig.getParameter<edm::InputTag>("ElectronCollection"))),
      scoutingElectronCollection_(consumes<std::vector<Run3ScoutingElectron>>(
          iConfig.getParameter<edm::InputTag>("ScoutingElectronCollection"))),
      eleIdMapTightToken_(consumes<edm::ValueMap<bool>>(iConfig.getParameter<edm::InputTag>("eleIdMapTight"))) {}

ScoutingEGammaCollectionMonitoring::~ScoutingEGammaCollectionMonitoring() {}

// Function to convert pseudo-rapidity to theta
double getPtFromEnergyMassEta(double energy, double mass, double eta) {
  double theta = 2.0 * std::atan(std::exp(-eta));
  double pt = std::sqrt(energy * energy - mass * mass) * std::sin(theta);
  return pt;
}

// ------------ method called for each event  ------------

void ScoutingEGammaCollectionMonitoring::dqmAnalyze(edm::Event const& iEvent,
                                                    edm::EventSetup const& iSetup,
                                                    kHistogramsScoutingEGammaCollectionMonitoring const& histos) const {
  ////////////////////////////////////////
  // Get PAT / Scouting Electron Token  //
  ////////////////////////////////////////

  edm::Handle<edm::View<pat::Electron>> patEls;
  iEvent.getByToken(electronCollection_, patEls);
  if (patEls.failedToGet()) {
    edm::LogWarning("ScoutingEGammaCollectionMonitoring") << "pat::Electron collection not found.";
    return;
  }

  edm::Handle<std::vector<Run3ScoutingElectron>> sctEls;
  iEvent.getByToken(scoutingElectronCollection_, sctEls);
  if (sctEls.failedToGet()) {
    edm::LogWarning("ScoutingEGammaCollectionMonitoring") << "Run3ScoutingElectron collection not found.";
    return;
  }

  edm::Handle<edm::ValueMap<bool>> tight_ele_id_decisions;
  iEvent.getByToken(eleIdMapTightToken_, tight_ele_id_decisions);

  edm::LogInfo("ScoutingEGammaCollectionMonitoring") << "Process pat::Electrons: " << patEls->size();
  edm::LogInfo("ScoutingEGammaCollectionMonitoring") << "Process Run3ScoutingElectrons: " << sctEls->size();

  // Trigger
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(triggerResultsToken_, triggerResults);
  const edm::TriggerNames& triggerNames = iEvent.triggerNames(*triggerResults);
  bool fire_doubleEG_DST = hasPatternInHLTPath(triggerNames, "DST_PFScouting_DoubleEG");
  bool fire_singlePhoton_DST = hasPatternInHLTPath(triggerNames, "DST_PFScouting_SinglePhotonEB");

  // Loop to verify the sorting of pat::Electron collection - REMOVE IN ONLINE
  // DQM
  for (size_t i = 1; i < patEls->size(); ++i) {
    if (patEls->ptrAt(i - 1)->pt() < patEls->ptrAt(i)->pt()) {
      edm::LogWarning("ScoutingEGammaCollectionMonitoring")
          << "pat::Electron collection not sorted by PT in descending order"
          << " will result in random histo filling. \n"
          << "pat::Electron[" << i << "].pt() = " << patEls->ptrAt(i)->pt() << "\n"
          << "pat::Electron[" << i + 1 << "].pt() = " << patEls->ptrAt(i + 1)->pt();
    }
  }

  // Fill pat::Electron histograms
  histos.patElectron.h1N->Fill(patEls->size());
  std::vector<int> tight_patElectron_index;
  for (size_t i = 0; i < patEls->size(); ++i) {
    const auto el = patEls->ptrAt(i);
    histos.patElectron.electrons.h1Pt->Fill(el->pt());
    histos.patElectron.electrons.h1Eta->Fill(el->eta());
    histos.patElectron.electrons.h1Phi->Fill(el->phi());
    if (((*tight_ele_id_decisions)[el]))
      tight_patElectron_index.push_back(i);
  }

  // Expect pat electrons to be sorted by pt
  if (!patEls->empty()) {
    histos.patElectron.electron1.h1Pt->Fill(patEls->ptrAt(0)->pt());
    histos.patElectron.electron1.h1Eta->Fill(patEls->ptrAt(0)->eta());
    histos.patElectron.electron1.h1Phi->Fill(patEls->ptrAt(0)->phi());
  }

  if (patEls->size() >= 2) {
    histos.patElectron.electron2.h1Pt->Fill(patEls->ptrAt(1)->pt());
    histos.patElectron.electron2.h1Eta->Fill(patEls->ptrAt(1)->eta());
    histos.patElectron.electron2.h1Phi->Fill(patEls->ptrAt(1)->phi());
    if (!tight_patElectron_index.empty()) {
      histos.patElectron.h1InvMass12->Fill((patEls->ptrAt(0)->p4() + patEls->ptrAt(1)->p4()).mass());
    }
  }

  if (tight_patElectron_index.size() == 2) {
    histos.patElectron.h1InvMassID->Fill(
        (patEls->ptrAt(tight_patElectron_index[0])->p4() + patEls->ptrAt(tight_patElectron_index[1])->p4()).mass());
  }

  // Fill the Run3ScoutingElectron histograms. No sorting assumed.
  histos.sctElectron.h1N->Fill(sctEls->size());
  // unsigned int leadSctElIndx = 0, subleadSctElIndx = -1;

  // Sort scouting electrons by pt - They are unordered in the collection by
  // default Get an index list of the same size as scouting electrons
  std::vector<int> sortedSctIdx(sctEls->size());
  std::iota(sortedSctIdx.begin(), sortedSctIdx.end(), 0);
  // Sort the indices based on the pt of the scouting electrons
  std::sort(
      sortedSctIdx.begin(), sortedSctIdx.end(), [&](int i, int j) { return sctEls->at(i).pt() > sctEls->at(j).pt(); });

  std::vector<int> tight_sctElectron_index;
  for (int idx : sortedSctIdx) {
    histos.sctElectron.electrons.h1Pt->Fill(sctEls->at(idx).pt());
    histos.sctElectron.electrons.h1Eta->Fill(sctEls->at(idx).eta());
    histos.sctElectron.electrons.h1Phi->Fill(sctEls->at(idx).phi());
    if (scoutingElectronID(sctEls->at(idx)))
      tight_sctElectron_index.push_back(idx);
  }
  sortedSctIdx.clear();
  if (!tight_sctElectron_index.empty() && sctEls->size() > 1) {
    math::PtEtaPhiMLorentzVector first_sct_el(
        sctEls->at(0).pt(), sctEls->at(0).eta(), sctEls->at(0).phi(), sctEls->at(0).m());
    math::PtEtaPhiMLorentzVector second_sct_el(
        sctEls->at(1).pt(), sctEls->at(1).eta(), sctEls->at(1).phi(), sctEls->at(1).m());
    // Use function to measure invariant mass with uniquely associated tracks
    histos.sctElectron.h1InvMass12->Fill((first_sct_el + second_sct_el).mass());
  }

  if (tight_sctElectron_index.size() == 2) {
    math::PtEtaPhiMLorentzVector sctEl0(sctEls->at(tight_sctElectron_index[0]).pt(),
                                        sctEls->at(tight_sctElectron_index[0]).eta(),
                                        sctEls->at(tight_sctElectron_index[0]).phi(),
                                        ELECTRON_MASS);
    math::PtEtaPhiMLorentzVector sctEl1(sctEls->at(tight_sctElectron_index[1]).pt(),
                                        sctEls->at(tight_sctElectron_index[1]).eta(),
                                        sctEls->at(tight_sctElectron_index[1]).phi(),
                                        ELECTRON_MASS);
    size_t gsfTrkIdx0 = 9999, gsfTrkIdx1 = 9999;
    bool foundGoodGsfTrkIdx0 = scoutingElectronGsfTrackIdx(sctEls->at(tight_sctElectron_index[0]), gsfTrkIdx0);
    bool foundGoodGsfTrkIdx1 = scoutingElectronGsfTrackIdx(sctEls->at(tight_sctElectron_index[1]), gsfTrkIdx1);

    if (!foundGoodGsfTrkIdx0 || !foundGoodGsfTrkIdx1)
      return;

    math::PtEtaPhiMLorentzVector sctElCombined0(
        getPtFromEnergyMassEta(
            sctEl0.energy(), ELECTRON_MASS, sctEls->at(tight_sctElectron_index[0]).trketa()[gsfTrkIdx0]),
        sctEls->at(tight_sctElectron_index[0]).trketa()[gsfTrkIdx0],
        sctEls->at(tight_sctElectron_index[0]).trkphi()[gsfTrkIdx0],
        ELECTRON_MASS);
    math::PtEtaPhiMLorentzVector sctElCombined1(
        getPtFromEnergyMassEta(
            sctEl1.energy(), ELECTRON_MASS, sctEls->at(tight_sctElectron_index[1]).trketa()[gsfTrkIdx1]),
        sctEls->at(tight_sctElectron_index[1]).trketa()[gsfTrkIdx1],
        sctEls->at(tight_sctElectron_index[1]).trkphi()[gsfTrkIdx1],
        ELECTRON_MASS);

    double invMass = (sctElCombined0 + sctElCombined1).mass();
    histos.sctElectron.h1InvMassID->Fill(invMass);
    if (fabs(sctEls->at(tight_sctElectron_index[0]).eta()) < ELE_etaEB &&
        fabs(sctEls->at(tight_sctElectron_index[1]).eta()) < ELE_etaEB) {
      histos.sctElectron.h1InvMassIDEBEB->Fill(invMass);
    } else if (fabs(sctEls->at(tight_sctElectron_index[0]).eta()) < ELE_etaEB &&
               fabs(sctEls->at(tight_sctElectron_index[1]).eta()) > ELE_etaEB) {
      histos.sctElectron.h1InvMassIDEBEE->Fill(invMass);
    } else if (fabs(sctEls->at(tight_sctElectron_index[0]).eta()) > ELE_etaEB &&
               fabs(sctEls->at(tight_sctElectron_index[1]).eta()) < ELE_etaEB) {
      histos.sctElectron.h1InvMassIDEBEE->Fill(invMass);
    } else {
      histos.sctElectron.h1InvMassIDEEEE->Fill(invMass);
    }

    if (fire_doubleEG_DST) {
      histos.sctElectron.h1InvMassID_passDoubleEG_DST->Fill(invMass);
      if (fabs(sctEls->at(tight_sctElectron_index[0]).eta()) < ELE_etaEB &&
          fabs(sctEls->at(tight_sctElectron_index[1]).eta()) < ELE_etaEB) {
        histos.sctElectron.h1InvMassIDEBEB_passDoubleEG_DST->Fill(invMass);
      } else if (fabs(sctEls->at(tight_sctElectron_index[0]).eta()) < ELE_etaEB &&
                 fabs(sctEls->at(tight_sctElectron_index[1]).eta()) > ELE_etaEB) {
        histos.sctElectron.h1InvMassIDEBEE_passDoubleEG_DST->Fill(invMass);
      } else if (fabs(sctEls->at(tight_sctElectron_index[0]).eta()) > ELE_etaEB &&
                 fabs(sctEls->at(tight_sctElectron_index[1]).eta()) < ELE_etaEB) {
        histos.sctElectron.h1InvMassIDEBEE_passDoubleEG_DST->Fill(invMass);
      } else {
        histos.sctElectron.h1InvMassIDEEEE_passDoubleEG_DST->Fill(invMass);
      }
    }
    if (fire_singlePhoton_DST) {
      histos.sctElectron.h1InvMassID_passSinglePhoton_DST->Fill(invMass);
      if (fabs(sctEls->at(tight_sctElectron_index[0]).eta()) < ELE_etaEB &&
          fabs(sctEls->at(tight_sctElectron_index[1]).eta()) < ELE_etaEB) {
        histos.sctElectron.h1InvMassIDEBEB_passSinglePhoton_DST->Fill(invMass);
      } else if (fabs(sctEls->at(tight_sctElectron_index[0]).eta()) < ELE_etaEB &&
                 fabs(sctEls->at(tight_sctElectron_index[1]).eta()) > ELE_etaEB) {
        histos.sctElectron.h1InvMassIDEBEE_passSinglePhoton_DST->Fill(invMass);
      } else if (fabs(sctEls->at(tight_sctElectron_index[0]).eta()) > ELE_etaEB &&
                 fabs(sctEls->at(tight_sctElectron_index[1]).eta()) < ELE_etaEB) {
        histos.sctElectron.h1InvMassIDEBEE_passSinglePhoton_DST->Fill(invMass);
      } else {
        histos.sctElectron.h1InvMassIDEEEE_passSinglePhoton_DST->Fill(invMass);
      }
    }
  }
}

void ScoutingEGammaCollectionMonitoring::bookHistograms(DQMStore::IBooker& ibook,
                                                        edm::Run const& run,
                                                        edm::EventSetup const& iSetup,
                                                        kHistogramsScoutingEGammaCollectionMonitoring& histos) const {
  ibook.setCurrentFolder(outputInternalPath_);

  // PAT Electron Total Summary
  histos.patElectron.h1N = ibook.book1D("all_patElectron_electrons_N", "all_patElectron_electrons_N", 20, 0., 20.);
  histos.patElectron.electrons.h1Pt =
      ibook.book1D("all_patElectron_electrons_Pt", "all_patElectron_electrons_Pt", 5000, 0., 500.);
  histos.patElectron.electrons.h1Eta =
      ibook.book1D("all_patElectron_electrons_Eta", "all_patElectron_electrons_Eta", 1000, -5., 5.);
  histos.patElectron.electrons.h1Phi =
      ibook.book1D("all_patElectron_electrons_Phi", "all_patElectron_electrons_Phi", 660, -3.3, 3.3);

  // Leading pT PAT Electron Summary
  histos.patElectron.electron1.h1Pt = ibook.book1D("patElectron_leading_Pt", "patElectron_leading_Pt", 5000, 0., 500.);
  histos.patElectron.electron1.h1Eta =
      ibook.book1D("patElectron_leading_Eta", "patElectron_leading_Eta", 1000, -5., 5.);
  histos.patElectron.electron1.h1Phi =
      ibook.book1D("patElectron_leading_Phi", "patElectron_leading_Phi", 660, -3.3, 3.3);

  // Subleading pT PAT Electron Summary
  histos.patElectron.electron2.h1Pt =
      ibook.book1D("patElectron_subleading_Pt", "patElectron_subleading_Pt", 5000, 0., 500.);
  histos.patElectron.electron2.h1Eta =
      ibook.book1D("patElectron_subleading_Eta", "patElectron_subleading_Eta", 1000, -5., 5.);
  histos.patElectron.electron2.h1Phi =
      ibook.book1D("patElectron_subleading_Phi", "patElectron_subleading_Phi", 660, -3.3, 3.3);

  // Inv Mass PAT Electron Summary
  histos.patElectron.h1InvMass12 = ibook.book1D("patElectron_E1E2_invMass", "patElectron_E1E2_invMass", 400, 0., 200.);
  histos.patElectron.h1InvMassID =
      ibook.book1D("patElectron_appliedID_invMass", "patElectron_appliedID_invMass", 400, 0., 200.);
  // Scouting electron summary
  histos.sctElectron.h1N = ibook.book1D("all_sctElectron_electrons_N", "all_sctElectron_electrons_N", 20, 0., 20.);
  histos.sctElectron.electrons.h1Pt =
      ibook.book1D("all_sctElectron_electrons_Pt", "all_sctElectron_electrons_Pt", 5000, 0., 500.);
  histos.sctElectron.electrons.h1Eta =
      ibook.book1D("all_sctElectron_electrons_Eta", "all_sctElectron_electrons_Eta", 1000, -5., 5.);
  histos.sctElectron.electrons.h1Phi =
      ibook.book1D("all_sctElectron_electrons_Phi", "all_sctElectron_electrons_Phi", 660, -3.3, 3.3);

  // Leading Scouting electron summary
  histos.sctElectron.electron1.h1Pt = ibook.book1D("sctElectron_leading_Pt", "sctElectron_leading_Pt", 5000, 0., 500.);
  histos.sctElectron.electron1.h1Eta =
      ibook.book1D("sctElectron_leading_Eta", "sctElectron_leading_Eta", 1000, -5., 5.);
  histos.sctElectron.electron1.h1Phi =
      ibook.book1D("sctElectron_leading_Phi", "sctElectron_leading_Phi", 660, -3.3, 3.3);

  // SubLeading Scouting electron summary
  histos.sctElectron.electron2.h1Pt =
      ibook.book1D("sctElectron_subleading_Pt", "sctElectron_subleading_Pt", 5000, 0., 500.);
  histos.sctElectron.electron2.h1Eta =
      ibook.book1D("sctElectron_subleading_Eta", "sctElectron_subleading_Eta", 1000, -5., 5.);
  histos.sctElectron.electron2.h1Phi =
      ibook.book1D("sctElectron_subleading_Phi", "sctElectron_subleading_Phi", 660, -3.3, 3.3);

  histos.sctElectron.h1InvMass12 = ibook.book1D("sctElectron_E1E2_invMass", "sctElectron_E1E2_invMass", 400, 0., 200.);
  histos.sctElectron.h1InvMassID =
      ibook.book1D("sctElectron_appliedID_invMass", "sctElectron_appliedID_invMass", 400, 0., 200.);
  histos.sctElectron.h1InvMassIDEBEB =
      ibook.book1D("sctElectron_EBEB_appliedID_invMass", "sctElectron_EBEB_appliedID_invMass", 400, 0., 200.);
  histos.sctElectron.h1InvMassIDEBEE =
      ibook.book1D("sctElectron_EBEE_appliedID_invMass", "sctElectron_EBEE_appliedID_invMass", 400, 0., 200.);
  histos.sctElectron.h1InvMassIDEEEE =
      ibook.book1D("sctElectron_EEEE_appliedID_invMass", "sctElectron_EEEE_appliedID_invMass", 400, 0., 200.);

  histos.sctElectron.h1InvMassID_passDoubleEG_DST = ibook.book1D(
      "sctElectron_appliedID_invMass_passDoubleEG_DST", "sctElectron_appliedID_invMass_passDoubleEG_DST", 400, 0., 200.);
  histos.sctElectron.h1InvMassIDEBEB_passDoubleEG_DST =
      ibook.book1D("sctElectron_EBEB_appliedID_invMass_passDoubleEG_DST",
                   "sctElectron_EBEB_appliedID_invMass_passDoubleEG_DST",
                   400,
                   0.,
                   200.);
  histos.sctElectron.h1InvMassIDEBEE_passDoubleEG_DST =
      ibook.book1D("sctElectron_EBEE_appliedID_invMass_passDoubleEG_DST",
                   "sctElectron_EBEE_appliedID_invMass_passDoubleEG_DST",
                   400,
                   0.,
                   200.);
  histos.sctElectron.h1InvMassIDEEEE_passDoubleEG_DST =
      ibook.book1D("sctElectron_EEEE_appliedID_invMass_passDoubleEG_DST",
                   "sctElectron_EEEE_appliedID_invMass_passDoubleEG_DST",
                   400,
                   0.,
                   200.);

  histos.sctElectron.h1InvMassID_passSinglePhoton_DST =
      ibook.book1D("sctElectron_appliedID_invMass_passSinglePhoton_DST",
                   "sctElectron_appliedID_invMass_passSinglePhoton_DST",
                   400,
                   0.,
                   200.);
  histos.sctElectron.h1InvMassIDEBEB_passSinglePhoton_DST =
      ibook.book1D("sctElectron_EBEB_appliedID_invMass_passSinglePhoton_DST",
                   "sctElectron_EBEB_appliedID_invMass_passSinglePhoton_DST",
                   400,
                   0.,
                   200.);
  histos.sctElectron.h1InvMassIDEBEE_passSinglePhoton_DST =
      ibook.book1D("sctElectron_EBEE_appliedID_invMass_passSinglePhoton_DST",
                   "sctElectron_EBEE_appliedID_invMass_passSinglePhoton_DST",
                   400,
                   0.,
                   200.);
  histos.sctElectron.h1InvMassIDEEEE_passSinglePhoton_DST =
      ibook.book1D("sctElectron_EEEE_appliedID_invMass_passSinglePhoton_DST",
                   "sctElectron_EEEE_appliedID_invMass_passSinglePhoton_DST",
                   400,
                   0.,
                   200.);
}

// ------------ method fills 'descriptions' with the allowed parameters for the
// module  ------------
void ScoutingEGammaCollectionMonitoring::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // The following says we do not know what parameters are allowed so do no
  // validation Please change this to state exactly what you do use, even if it
  // is no parameters
  edm::ParameterSetDescription desc;
  desc.add<std::string>("OutputInternalPath", "MY_FOLDER");
  desc.add<edm::InputTag>("TriggerResultTag", edm::InputTag("TriggerResults", "", "HLT"));
  desc.add<edm::InputTag>("ElectronCollection", edm::InputTag("slimmedElectrons"));
  desc.add<edm::InputTag>("ScoutingElectronCollection", edm::InputTag("hltScoutingEgammaPacker"));
  desc.add<edm::InputTag>("eleIdMapTight",
                          edm::InputTag("egmGsfElectronIDs:cutBasedElectronID-RunIIIWinter22-V1-tight"));
  descriptions.add("ScoutingEGammaCollectionMonitoring", desc);
}

bool ScoutingEGammaCollectionMonitoring::scoutingElectronID(const Run3ScoutingElectron el) const {
  bool isEB = (fabs(el.eta()) < ELE_etaEB);
  if (isEB) {
    if (el.sigmaIetaIeta() > 0.015)
      return false;
    if (el.hOverE() > 0.2)
      return false;
    if (fabs(el.dEtaIn()) > 0.008)
      return false;
    if (fabs(el.dPhiIn()) > 0.06)
      return false;
    if (el.ecalIso() / el.pt() > 0.25)
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
    if (el.ecalIso() / el.pt() > 0.1)
      return false;
    return true;
  }
}

bool ScoutingEGammaCollectionMonitoring::scoutingElectronGsfTrackID(const Run3ScoutingElectron el,
                                                                    size_t trackIdx) const {
  if (trackIdx > el.trkpt().size())
    edm::LogError("ScoutingEGammaCollectionMonitoring")
        << "Invalid track index for electron: Exceeds the number of tracks";

  math::PtEtaPhiMLorentzVector particleSC(el.pt(), el.eta(), el.phi(), ELECTRON_MASS);
  math::PtEtaPhiMLorentzVector particleTrk(
      el.trkpt()[trackIdx], el.trketa()[trackIdx], el.trkphi()[trackIdx], ELECTRON_MASS);

  double scEnergy = particleSC.energy();
  double trkEnergy = particleTrk.energy();
  double relEnergyDiff = fabs(scEnergy - trkEnergy) / scEnergy;
  double dPhi = deltaPhi(particleSC.phi(), particleTrk.phi());

  bool isEB = (fabs(el.eta()) < ELE_etaEB);
  if (isEB) {
    if (el.trkpt()[trackIdx] < 12)
      return false;
    if (relEnergyDiff > 1)
      return false;
    if (dPhi > 0.06)
      return false;
    if (el.trkchi2overndf()[trackIdx] > 3)
      return false;
    return true;
  } else {
    if (el.trkpt()[trackIdx] < 12)
      return false;
    if (relEnergyDiff > 1)
      return false;
    if (dPhi > 0.06)
      return false;
    if (el.trkchi2overndf()[trackIdx] > 2)
      return false;
    return true;
  }
}

bool ScoutingEGammaCollectionMonitoring::scoutingElectronGsfTrackIdx(const Run3ScoutingElectron el,
                                                                     size_t& trackIdx) const {
  bool foundGoodGsfTrkIdx = false;
  for (size_t i = 0; i < el.trkpt().size(); ++i) {
    if (scoutingElectronGsfTrackID(el, i)) {
      if (!foundGoodGsfTrkIdx) {
        foundGoodGsfTrkIdx = true;
        trackIdx = i;
      } else {
        double relPtDiff = fabs(el.trkpt()[i] - el.pt()) / el.pt();
        double relPtDiffOld = fabs(el.trkpt()[trackIdx] - el.pt()) / el.pt();
        if (relPtDiff < relPtDiffOld)
          trackIdx = i;
      }
    }
  }
  return foundGoodGsfTrkIdx;
}

bool ScoutingEGammaCollectionMonitoring::hasPatternInHLTPath(const edm::TriggerNames& triggerNames,
                                                             const std::string& pattern) const {
  for (unsigned int i = 0; i < triggerNames.size(); ++i) {
    const std::string& triggerName = triggerNames.triggerName(i);

    // Check if triggerName starts with the specified prefix
    if (triggerName.find(pattern) == 0) {  // Position 0 means it starts with 'prefix'
      return true;                         // Pattern match found
    }
  }
  return false;  // No match found
}

// define this as a plug-in
DEFINE_FWK_MODULE(ScoutingEGammaCollectionMonitoring);

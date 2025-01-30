/*
Scouting Muon DQM core implementation. This code does the following:
  1) Reads muon collection, scouting muon collection and scouting vertex collection
  2) Tag and Probe method: For each event, check whether one of the muons passes a tight ID
     (tag), and pair it with another muon in the event (probe). If this dimuon system is 
     within the mass range of the J/Psi, monitor distributions of the probe and the efficiency
     of the probe to pass certain IDs. For now we are measuring the efficiency of the probe
     passing the tag ID (If the dimuon system is within J/Psi, add it to the denominator
     distributions, and if the probe passes the tag ID, add it to the numerator distributions
     as well.)
  3) Fills histograms
Author: Javier Garcia de Castro, email:javigdc@bu.edu
*/

//Files to include
#include "ScoutingMuonTagProbeAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <cmath>

ScoutingMuonTagProbeAnalyzer::ScoutingMuonTagProbeAnalyzer(const edm::ParameterSet& iConfig)
    : outputInternalPath_(iConfig.getParameter<std::string>("OutputInternalPath")),
      scoutingMuonCollection_(
          consumes<std::vector<Run3ScoutingMuon>>(iConfig.getParameter<edm::InputTag>("ScoutingMuonCollection"))),
      scoutingVtxCollection_(
          consumes<std::vector<Run3ScoutingVertex>>(iConfig.getParameter<edm::InputTag>("ScoutingVtxCollection"))),
      runWithoutVtx_(iConfig.getParameter<bool>("runWithoutVertex")) {}

ScoutingMuonTagProbeAnalyzer::~ScoutingMuonTagProbeAnalyzer() {}

void ScoutingMuonTagProbeAnalyzer::dqmAnalyze(edm::Event const& iEvent,
                                              edm::EventSetup const& iSetup,
                                              kTagProbeMuonHistos const& histos) const {
  //Read scouting muon collection
  edm::Handle<std::vector<Run3ScoutingMuon>> sctMuons;
  iEvent.getByToken(scoutingMuonCollection_, sctMuons);
  if (sctMuons.failedToGet()) {
    edm::LogWarning("ScoutingMonitoring") << "Run3ScoutingMuon collection not found.";
    return;
  }

  //Read scouting vertex collection
  edm::Handle<std::vector<Run3ScoutingVertex>> sctVertex;
  iEvent.getByToken(scoutingVtxCollection_, sctVertex);
  if (sctVertex.failedToGet()) {
    edm::LogWarning("ScoutingMonitoring") << "Run3ScoutingVertex collection not found.";
    return;
  }

  edm::LogInfo("ScoutingMonitoring") << "Process Run3ScoutingMuons: " << sctMuons->size();

  edm::LogInfo("ScoutingMonitoring") << "Process Run3ScoutingVertex: " << sctVertex->size();

  //Core of Tag and Probe implementation
  bool foundTag = false;
  //First find the tag
  for (const auto& sct_mu : *sctMuons) {
    if (!scoutingMuonID(sct_mu))
      continue;
    if (foundTag)
      continue;
    math::PtEtaPhiMLorentzVector tag_sct_mu(sct_mu.pt(), sct_mu.eta(), sct_mu.phi(), sct_mu.m());
    const std::vector<int> vtxIndx_tag = sct_mu.vtxIndx();

    //Then pair the tag with the probe
    for (const auto& sct_mu_second : *sctMuons) {
      if (&sct_mu_second == &sct_mu)
        continue;
      math::PtEtaPhiMLorentzVector probe_sct_mu(
          sct_mu_second.pt(), sct_mu_second.eta(), sct_mu_second.phi(), sct_mu_second.m());
      if (sct_mu_second.pt() < 1)
        continue;
      const std::vector<int> vtxIndx_probe = sct_mu_second.vtxIndx();

      float invMass = (tag_sct_mu + probe_sct_mu).mass();
      edm::LogInfo("ScoutingMonitoring") << "Inv Mass: " << invMass;
      //If dimuon system comes from J/Psi, process event
      if ((2.4 < invMass && invMass < 3.8)) {
        //Boolean added because hltScoutingMuonPackerVtx collection doesn't have vertices for the moment
        if (runWithoutVtx_) {
          Run3ScoutingVertex vertex;
          //If probe passes tag ID, add it to the numerator
          if (scoutingMuonID(sct_mu_second)) {
            fillHistograms_resonance(histos.resonanceJ_numerator, sct_mu_second, vertex, invMass, -99.);
          }
          //Add all events to the denominator
          fillHistograms_resonance(histos.resonanceJ_denominator, sct_mu_second, vertex, invMass, -99.);
        } else {
          if (vtxIndx_tag.empty() || vtxIndx_probe.empty())
            continue;
          for (const auto& commonIdx : vtxIndx_tag) {
            if (std::find(vtxIndx_probe.begin(), vtxIndx_probe.end(), commonIdx) != vtxIndx_probe.end()) {
              const auto& vertex = (*sctVertex)[commonIdx];
              float lxy = sqrt(vertex.x() * vertex.x() + vertex.y() * vertex.y());
              if (scoutingMuonID(sct_mu_second)) {
                fillHistograms_resonance(histos.resonanceJ_numerator, sct_mu_second, vertex, invMass, lxy);
              }
              fillHistograms_resonance(histos.resonanceJ_denominator, sct_mu_second, vertex, invMass, lxy);
            }
          }
        }
      }
    }
    foundTag = true;
  }
}

//Tag ID
bool ScoutingMuonTagProbeAnalyzer::scoutingMuonID(const Run3ScoutingMuon mu) const {
  math::PtEtaPhiMLorentzVector particle(mu.pt(), mu.eta(), mu.phi(), 0.10566);
  double normchisq_threshold = 3.0;
  double pt_threshold = 3.0;
  double eta_threshold = 2.4;
  int layer_threshold = 4;

  if (mu.pt() > pt_threshold && fabs(mu.eta()) < eta_threshold && mu.normalizedChi2() < normchisq_threshold &&
      mu.isGlobalMuon() && mu.nTrackerLayersWithMeasurement() > layer_threshold) {
    return true;
  }
  return false;
}

//Fill histograms
void ScoutingMuonTagProbeAnalyzer::fillHistograms_resonance(const kProbeKinematicMuonHistos histos,
                                                            const Run3ScoutingMuon mu,
                                                            const Run3ScoutingVertex vertex,
                                                            const float inv_mass,
                                                            const float lxy) const {
  histos.hPt->Fill(mu.pt());
  histos.hEta->Fill(mu.eta());
  histos.hPhi->Fill(mu.phi());
  histos.hInvMass->Fill(inv_mass);
  histos.hNormChisq->Fill(mu.normalizedChi2());
  histos.hTrk_dxy->Fill(mu.trk_dxy());
  histos.hTrk_dz->Fill(mu.trk_dz());
  histos.htype->Fill(mu.type());
  histos.hcharge->Fill(mu.charge());
  histos.hecalIso->Fill(mu.ecalIso());
  histos.hhcalIso->Fill(mu.hcalIso());
  histos.htrackIso->Fill(mu.trackIso());
  histos.hnValidStandAloneMuonHits->Fill(mu.nValidStandAloneMuonHits());
  histos.hnStandAloneMuonMatchedStations->Fill(mu.nStandAloneMuonMatchedStations());
  histos.hnValidRecoMuonHits->Fill(mu.nValidRecoMuonHits());
  histos.hnRecoMuonChambers->Fill(mu.nRecoMuonChambers());
  histos.hnRecoMuonChambersCSCorDT->Fill(mu.nRecoMuonChambersCSCorDT());
  histos.hnRecoMuonMatches->Fill(mu.nRecoMuonMatches());
  histos.hnRecoMuonMatchedStations->Fill(mu.nRecoMuonMatchedStations());
  histos.hnRecoMuonExpectedMatchedStations->Fill(mu.nRecoMuonExpectedMatchedStations());
  histos.hnValidPixelHits->Fill(mu.nValidPixelHits());
  histos.hnValidStripHits->Fill(mu.nValidStripHits());
  histos.htrk_chi2->Fill(mu.trk_chi2());
  histos.htrk_ndof->Fill(mu.trk_ndof());
  histos.htrk_lambda->Fill(mu.trk_lambda());
  histos.htrk_pt->Fill(mu.trk_pt());
  histos.htrk_eta->Fill(mu.trk_eta());
  histos.htrk_dxyError->Fill(mu.trk_dxyError());
  histos.htrk_dzError->Fill(mu.trk_dzError());
  histos.htrk_qoverpError->Fill(mu.trk_qoverpError());
  histos.htrk_lambdaError->Fill(mu.trk_lambdaError());
  histos.htrk_phiError->Fill(mu.trk_phiError());
  histos.htrk_dsz->Fill(mu.trk_dsz());
  histos.htrk_dszError->Fill(mu.trk_dszError());
  histos.htrk_dsz->Fill(mu.trk_dsz());
  histos.htrk_vx->Fill(mu.trk_vx());
  histos.htrk_vy->Fill(mu.trk_vy());
  histos.htrk_vz->Fill(mu.trk_vz());
  histos.hnPixel->Fill(mu.nPixelLayersWithMeasurement());
  histos.hnTracker->Fill(mu.nTrackerLayersWithMeasurement());
  histos.htrk_qoverp->Fill(mu.trk_qoverp());

  if (!runWithoutVtx_) {
    histos.hLxy->Fill(lxy);
    histos.hXError->Fill(vertex.xError());
    histos.hYError->Fill(vertex.yError());
    histos.hChi2->Fill(vertex.chi2());
    histos.hZ->Fill(vertex.z());
    histos.hx->Fill(vertex.x());
    histos.hy->Fill(vertex.y());
    histos.hZerror->Fill(vertex.zError());
    histos.htracksSize->Fill(vertex.tracksSize());
  }
}

//Save histograms
void ScoutingMuonTagProbeAnalyzer::bookHistograms(DQMStore::IBooker& ibook,
                                                  edm::Run const& run,
                                                  edm::EventSetup const& iSetup,
                                                  kTagProbeMuonHistos& histos) const {
  ibook.setCurrentFolder(outputInternalPath_);
  bookHistograms_resonance(ibook, run, iSetup, histos.resonanceJ_numerator, "resonanceJ_numerator");
  bookHistograms_resonance(ibook, run, iSetup, histos.resonanceJ_denominator, "resonanceJ_denominator");
}

//Set axes labels and range
void ScoutingMuonTagProbeAnalyzer::bookHistograms_resonance(DQMStore::IBooker& ibook,
                                                            edm::Run const& run,
                                                            edm::EventSetup const& iSetup,
                                                            kProbeKinematicMuonHistos& histos,
                                                            const std::string& name) const {
  ibook.setCurrentFolder(outputInternalPath_);

  histos.hPt = ibook.book1D(name + "_Probe_sctMuon_Pt", name + "_Probe_sctMuon_Pt;Muon pt (GeV); Muons", 60, 0, 50.0);
  histos.hEta = ibook.book1D(name + "_Probe_sctMuon_Eta", name + "_Probe_sctMuon_Eta; Muon eta; Muons", 60, -5.0, 5.0);
  histos.hPhi = ibook.book1D(name + "_Probe_sctMuon_Phi", name + "_Probe_sctMuon_Phi; Muon phi; Muons", 60, -3.3, 3.3);
  histos.hInvMass = ibook.book1D(
      name + "_sctMuon_Invariant_Mass", name + "_sctMuon_Invariant_Mass;Muon Inv mass (GeV); Muons", 100, 0, 5);
  histos.hNormChisq = ibook.book1D(
      name + "_Probe_sctMuon_NormChisq", name + "_Probe_sctMuon_NormChisq; Muon normChi2; Muons", 60, 0, 5.0);
  histos.hTrk_dxy =
      ibook.book1D(name + "_Probe_sctMuon_Trk_dxy", name + "_Probe_sctMuon_Trk_dxy; Track dxy; Muons", 60, 0, 5.0);
  histos.hTrk_dz =
      ibook.book1D(name + "_Probe_sctMuon_Trk_dz", name + "_Probe_sctMuon_Trk_dz; Track dz; Muons", 60, 0, 20.0);
  histos.htype = ibook.book1D(name + "_Probe_sctMuon_type", name + "_Probe_sctMuon_type;Muon type; Muons", 15, 0, 15);
  histos.hcharge =
      ibook.book1D(name + "_Probe_sctMuon_charge", name + "_Probe_sctMuon_charge; Muon charge; Muons", 3, -1, 2);
  histos.hecalIso = ibook.book1D(
      name + "_Probe_sctMuon_ecalIso", name + "_Probe_sctMuon_ecalIso; Muon ecalIso; Muons", 60, 0.0, 20.0);
  histos.hhcalIso = ibook.book1D(
      name + "_Probe_sctMuon_hcalIso", name + "_Probe_sctMuon_hcalIso; Muon hcalIso; Muons", 60, 0.0, 20.0);
  histos.htrackIso =
      ibook.book1D(name + "_sctMuon_trackIso", name + "_sctMuon_trackIso;Muon trackIso; Muons", 100, 0, 7);
  histos.hnValidStandAloneMuonHits =
      ibook.book1D(name + "_Probe_sctMuon_nValidStandAloneMuonHits",
                   name + "_Probe_sctMuon_nValidStandAloneMuonHits;nValidStandAloneMuonHits; Muons",
                   25,
                   0,
                   25);
  histos.hnValidRecoMuonHits = ibook.book1D(name + "_Probe_sctMuon_nValidRecoMuonHits",
                                            name + "_Probe_sctMuon_nValidRecoMuonHits;nValidRecoMuonHits; Muons",
                                            25,
                                            0,
                                            25);
  histos.hnRecoMuonChambers = ibook.book1D(name + "_Probe_sctMuon_nRecoMuonChambers",
                                           name + "_Probe_sctMuon_nRecoMuonChambers; nRecoMuonChambers; Muons",
                                           10,
                                           0,
                                           10.0);
  histos.hnStandAloneMuonMatchedStations =
      ibook.book1D(name + "_Probe_sctMuon_nStandAloneMuonMatchedStations",
                   name + "_Probe_sctMuon_nStandAloneMuonMatchedStations; nStandAloneMuonMatchedStations; Muons",
                   5,
                   0,
                   5.0);
  histos.hnRecoMuonChambersCSCorDT =
      ibook.book1D(name + "_Probe_sctMuon_nRecoMuonChambersCSCorDT",
                   name + "_Probe_sctMuon_nRecoMuonChambersCSCorDT;nRecoMuonChambersCSCorDT; Muons",
                   10,
                   0,
                   10.0);
  histos.hnRecoMuonMatches = ibook.book1D(name + "_Probe_sctMuon_nRecoMuonMatches",
                                          name + "_Probe_sctMuon_nRecoMuonMatches; nRecoMuonMatches; Muons",
                                          6,
                                          0,
                                          6);
  histos.hnRecoMuonMatchedStations =
      ibook.book1D(name + "_Probe_sctMuon_nRecoMuonMatchedStations",
                   name + "_Probe_sctMuon_nRecoMuonMatchedStations; nRecoMuonMatchedStations; Muons",
                   5,
                   0.0,
                   5.0);
  histos.hnRecoMuonExpectedMatchedStations =
      ibook.book1D(name + "_sctMuon_nRecoMuonExpectedMatchedStations",
                   name + "_sctMuon_nRecoMuonExpectedMatchedStations;nRecoMuonExpectedMatchedStations; Muons",
                   6,
                   0,
                   6.0);
  histos.hnValidPixelHits = ibook.book1D(name + "_Probe_sctMuon_nValidPixelHits",
                                         name + "_Probe_sctMuon_nValidPixelHits;nValidPixelHits; Muons",
                                         14,
                                         0,
                                         14.0);
  histos.hnValidStripHits = ibook.book1D(name + "_Probe_sctMuon_nValidStripHits",
                                         name + "_Probe_sctMuon_nValidStripHits; nValidStripHits; Muons",
                                         25,
                                         0,
                                         25);
  histos.htrk_chi2 =
      ibook.book1D(name + "_Probe_sctMuon_trk_chi2", name + "_Probe_sctMuon_trk_chi2; trk_chi2; Muons", 60, 0.0, 20);
  histos.htrk_ndof = ibook.book1D(name + "_sctMuon_trk_ndof", name + "_sctMuon_trk_ndof;trk_ndof; Muons", 100, 0, 50);
  histos.htrk_lambda = ibook.book1D(
      name + "_Probe_sctMuon_trk_lambda", name + "_Probe_sctMuon_trk_lambda; trk_lambda; Muons", 60, -5, 5);
  histos.htrk_pt =
      ibook.book1D(name + "_Probe_sctMuon_trk_pt", name + "_Probe_sctMuon_trk_pt; trk_pt; Muons", 60, 0, 50.0);
  histos.htrk_eta =
      ibook.book1D(name + "_Probe_sctMuon_trk_eta", name + "_Probe_sctMuon_trk_eta;trk_eta; Muons", 60, -3.3, 3.3);
  histos.htrk_dxyError = ibook.book1D(
      name + "_Probe_sctMuon_trk_dxyError", name + "_Probe_sctMuon_trk_dxyError; trk_dxyError; Muons", 60, 0.0, 0.8);
  histos.htrk_dzError = ibook.book1D(
      name + "_Probe_sctMuon_trk_dzError", name + "_Probe_sctMuon_trk_dzError; trk_dzError; Muons", 60, 0.0, 2);
  histos.htrk_qoverpError = ibook.book1D(
      name + "_sctMuon_trk_qoverpError", name + "_sctMuon_trk_qoverpError;trk_qoverpError; Muons", 100, 0, 0.05);
  histos.htrk_lambdaError = ibook.book1D(name + "_Probe_sctMuon_trk_lambdaError",
                                         name + "_Probe_sctMuon_trk_lambdaError; trk_lambdaError; Muons",
                                         60,
                                         0,
                                         0.2);
  histos.htrk_phiError = ibook.book1D(
      name + "_Probe_sctMuon_trk_phiError", name + "_Probe_sctMuon_trk_phiError; trk_phiError; Muons", 60, 0, 0.2);
  histos.htrk_dsz =
      ibook.book1D(name + "_Probe_sctMuon_trk_dsz", name + "_Probe_sctMuon_trk_dsz; trk_dsz; Muons", 60, -20.0, 20.0);
  histos.htrk_dszError = ibook.book1D(
      name + "_Probe_sctMuon_trk_dszError", name + "_Probe_sctMuon_trk_dszError; trk_dszError; Muons", 60, 0, 0.8);
  histos.htrk_vx =
      ibook.book1D(name + "_Probe_sctMuon_trk_vx", name + "_Probe_sctMuon_trk_vx; trk_vx; Muons", 60, -1, 1);
  histos.htrk_vy =
      ibook.book1D(name + "_Probe_sctMuon_trk_vy", name + "_Probe_sctMuon_trk_vy; trk_vy; Muons", 60, -1, 1);
  histos.htrk_vz =
      ibook.book1D(name + "_Probe_sctMuon_trk_vz", name + "_Probe_sctMuon_trk_vz; trk_vz; Muons", 60, -20, 20);
  histos.hnPixel =
      ibook.book1D(name + "_Probe_sctMuon_nPixel", name + "_Probe_sctMuon_nPixel; n Pixel Layers; Muons", 6, 0, 6);
  histos.hnTracker = ibook.book1D(
      name + "_Probe_sctMuon_nTracker", name + "_Probe_sctMuon_nTracker; n Tracker Layers; Muons", 14, 0, 25);
  histos.htrk_qoverp = ibook.book1D(
      name + "_Probe_sctMuon_trk_qoverp", name + "_Probe_sctMuon_trk_qoverp; n Track q/p; Muons", 40, -1, 1);
  histos.hLxy = ibook.book1D(name + "_Vertex_Lxy", name + "_Vertex_Lxy; lxy; Muons", 60, 0, 20);
  histos.hXError = ibook.book1D(name + "_Vertex_Xerror", name + "_Vertex_Xerror; vertex x error; Muons", 60, 0, 1);
  histos.hYError = ibook.book1D(name + "_Vertex_Yerror", name + "_Vertex_Yerror; vertex y error; Muons", 60, 0, 1);
  histos.hChi2 = ibook.book1D(name + "_Vertex_chi2", name + "_Vertex_chi2; vertex chi2; Muons", 60, 0, 15);
  histos.hZ = ibook.book1D(name + "_Vertex_z", name + "_Vertex_z; vertex z; Muons", 60, 0, 15);
  histos.hx = ibook.book1D(name + "_Vertex_x", name + "_Vertex_x; vertex x; Muons", 60, -1, 1);
  histos.hy = ibook.book1D(name + "_Vertex_y", name + "_Vertex_y; vertex y; Muons", 60, -1, 1);
  histos.hZerror = ibook.book1D(name + "_Vertex_z error", name + "_Vertex_z_error; vertex z error; Muons", 60, 0, 3);
  histos.htracksSize =
      ibook.book1D(name + "_Vertex_tracksSize", name + "_Vertex_tracksSize; vertex tracksSize; Muons", 60, 0, 10);
}

//Descriptions to read the collections
void ScoutingMuonTagProbeAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("OutputInternalPath", "MY_FOLDER");
  desc.add<edm::InputTag>("ScoutingMuonCollection", edm::InputTag("Run3ScoutingMuons"));
  desc.add<edm::InputTag>("ScoutingVtxCollection", edm::InputTag("hltScoutingMuonPackerNoVtx"));
  desc.add<bool>("runWithoutVertex", true);
  descriptions.add("ScoutingMuonTagProbeAnalyzer", desc);
}

DEFINE_FWK_MODULE(ScoutingMuonTagProbeAnalyzer);

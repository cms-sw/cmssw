/*
 *  $Date: 2012/01/11 13:53:29 $
 *  $Revision: 1.14 $
 *  \author M. Marienfeld - DESY Hamburg
 */

#include "TLorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"
#include "DQM/Physics/src/TopDiLeptonDQM.h"
#include "FWCore/Common/interface/TriggerNames.h"

using namespace std;
using namespace edm;

TopDiLeptonDQM::TopDiLeptonDQM(const edm::ParameterSet& ps) {
  moduleName_ = ps.getUntrackedParameter<string>("moduleName");
  outputFile_ = ps.getUntrackedParameter<string>("outputFile");
  triggerResults_ = consumes<TriggerResults>(ps.getParameter<edm::InputTag>("TriggerResults"));
  hltPaths_ = ps.getParameter<vector<string> >("hltPaths");
  hltPaths_sig_ = ps.getParameter<vector<string> >("hltPaths_sig");
  hltPaths_trig_ = ps.getParameter<vector<string> >("hltPaths_trig");

  vertex_ = consumes<reco::VertexCollection>(ps.getParameter<edm::InputTag>("vertexCollection"));
  vertex_X_cut_ = ps.getParameter<double>("vertex_X_cut");
  vertex_Y_cut_ = ps.getParameter<double>("vertex_Y_cut");
  vertex_Z_cut_ = ps.getParameter<double>("vertex_Z_cut");

  muons_ = consumes<reco::MuonCollection>(ps.getParameter<edm::InputTag>("muonCollection"));
  muon_pT_cut_ = ps.getParameter<double>("muon_pT_cut");
  muon_eta_cut_ = ps.getParameter<double>("muon_eta_cut");
  muon_iso_cut_ = ps.getParameter<double>("muon_iso_cut");

  elecs_ = consumes<reco::GsfElectronCollection>(ps.getParameter<edm::InputTag>("elecCollection"));
  elec_pT_cut_ = ps.getParameter<double>("elec_pT_cut");
  elec_eta_cut_ = ps.getParameter<double>("elec_eta_cut");
  elec_iso_cut_ = ps.getParameter<double>("elec_iso_cut");
  elec_emf_cut_ = ps.getParameter<double>("elec_emf_cut");

  MassWindow_up_ = ps.getParameter<double>("MassWindow_up");
  MassWindow_down_ = ps.getParameter<double>("MassWindow_down");

  for (int i = 0; i < 100; ++i) {
    N_sig[i] = 0;
    N_trig[i] = 0;
    Eff[i] = 0.;
  }

  N_mumu = 0;
  N_muel = 0;
  N_elel = 0;
  Events_ = nullptr;
  Trigs_ = nullptr;
  TriggerEff_ = nullptr;
  Ntracks_ = nullptr;
  Nmuons_ = nullptr;
  Nmuons_iso_ = nullptr;
  Nmuons_charge_ = nullptr;
  VxVy_muons_ = nullptr;
  Vz_muons_ = nullptr;
  pT_muons_ = nullptr;
  eta_muons_ = nullptr;
  phi_muons_ = nullptr;
  Nelecs_ = nullptr;
  Nelecs_iso_ = nullptr;
  Nelecs_charge_ = nullptr;
  HoverE_elecs_ = nullptr;
  pT_elecs_ = nullptr;
  eta_elecs_ = nullptr;
  phi_elecs_ = nullptr;
  MuIso_emEt03_ = nullptr;
  MuIso_hadEt03_ = nullptr;
  MuIso_hoEt03_ = nullptr;
  MuIso_nJets03_ = nullptr;
  MuIso_nTracks03_ = nullptr;
  MuIso_sumPt03_ = nullptr;
  MuIso_CombRelIso03_ = nullptr;
  ElecIso_cal_ = nullptr;
  ElecIso_trk_ = nullptr;
  ElecIso_CombRelIso_ = nullptr;
  dimassRC_ = nullptr;
  dimassWC_ = nullptr;
  dimassRC_LOGX_ = nullptr;
  dimassWC_LOGX_ = nullptr;
  dimassRC_LOG10_ = nullptr;
  dimassWC_LOG10_ = nullptr;
  D_eta_muons_ = nullptr;
  D_phi_muons_ = nullptr;
  D_eta_elecs_ = nullptr;
  D_phi_elecs_ = nullptr;
  D_eta_lepts_ = nullptr;
  D_phi_lepts_ = nullptr;
}

TopDiLeptonDQM::~TopDiLeptonDQM() {}

void TopDiLeptonDQM::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const&, edm::EventSetup const&) {
  iBooker.setCurrentFolder(moduleName_);

  Events_ = iBooker.book1D("00_Events", "Isolated dilepton events", 5, 0., 5.);
  Events_->setBinLabel(2, "#mu #mu", 1);
  Events_->setBinLabel(3, "#mu e", 1);
  Events_->setBinLabel(4, "e e", 1);

  Trigs_ = iBooker.book1D("01_Trigs", "Fired muon/electron triggers", 15, 0., 15.);
  TriggerEff_ = iBooker.book1D("02_TriggerEff", "HL Trigger Efficiencies", 10, 0., 10.);
  TriggerEff_->setTitle(
      "HL Trigger Efficiencies #epsilon_{signal} = #frac{[signal] && "
      "[control]}{[control]}");
  Ntracks_ = iBooker.book1D("Ntracks", "Number of tracks", 50, 0., 50.);

  Nmuons_ = iBooker.book1D("03_Nmuons", "Number of muons", 20, 0., 10.);
  Nmuons_iso_ = iBooker.book1D("04_Nmuons_iso", "Number of isolated muons", 20, 0., 10.);
  Nmuons_charge_ = iBooker.book1D("Nmuons_charge", "Number of muons * moun charge", 19, -10., 10.);
  VxVy_muons_ = iBooker.book2D("VxVy_muons", "Vertex x-y-positon (global)", 40, -1., 1., 40, -1., 1.);
  Vz_muons_ = iBooker.book1D("Vz_muons", "Vertex z-positon (global)", 40, -20., 20.);
  pT_muons_ = iBooker.book1D("pT_muons", "P_T of muons", 40, 0., 200.);
  eta_muons_ = iBooker.book1D("eta_muons", "Eta of muons", 50, -5., 5.);
  phi_muons_ = iBooker.book1D("phi_muons", "Phi of muons", 40, -4., 4.);

  Nelecs_ = iBooker.book1D("05_Nelecs", "Number of electrons", 20, 0., 10.);
  Nelecs_iso_ = iBooker.book1D("06_Nelecs_iso", "Number of isolated electrons", 20, 0., 10.);
  Nelecs_charge_ = iBooker.book1D("Nelecs_charge", "Number of elecs * elec charge", 19, -10., 10.);
  HoverE_elecs_ = iBooker.book1D("HoverE_elecs", "Hadronic over Ecal energy", 50, 0., 1.);
  pT_elecs_ = iBooker.book1D("pT_elecs", "P_T of electrons", 40, 0., 200.);
  eta_elecs_ = iBooker.book1D("eta_elecs", "Eta of electrons", 50, -5., 5.);
  phi_elecs_ = iBooker.book1D("phi_elecs", "Phi of electrons", 40, -4., 4.);

  MuIso_emEt03_ = iBooker.book1D("MuIso_emEt03", "Muon emEt03", 20, 0., 20.);
  MuIso_hadEt03_ = iBooker.book1D("MuIso_hadEt03", "Muon hadEt03", 20, 0., 20.);
  MuIso_hoEt03_ = iBooker.book1D("MuIso_hoEt03", "Muon hoEt03", 20, 0., 20.);
  MuIso_nJets03_ = iBooker.book1D("MuIso_nJets03", "Muon nJets03", 10, 0., 10.);
  MuIso_nTracks03_ = iBooker.book1D("MuIso_nTracks03", "Muon nTracks03", 20, 0., 20.);
  MuIso_sumPt03_ = iBooker.book1D("MuIso_sumPt03", "Muon sumPt03", 20, 0., 40.);
  MuIso_CombRelIso03_ = iBooker.book1D("07_MuIso_CombRelIso03", "Muon CombRelIso03", 20, 0., 1.);

  ElecIso_cal_ = iBooker.book1D("ElecIso_cal", "Electron Iso_cal", 21, -1., 20.);
  ElecIso_trk_ = iBooker.book1D("ElecIso_trk", "Electron Iso_trk", 21, -2., 40.);
  ElecIso_CombRelIso_ = iBooker.book1D("08_ElecIso_CombRelIso", "Electron CombRelIso", 20, 0., 1.);

  const int nbins = 200;

  double logmin = 0.;
  double logmax = 3.;  // 10^(3.)=1000

  float bins[nbins + 1];

  for (int i = 0; i <= nbins; i++) {
    double log = logmin + (logmax - logmin) * i / nbins;
    bins[i] = std::pow(10.0, log);
  }

  dimassRC_ = iBooker.book1D("09_dimassRC", "Dilepton mass RC", 50, 0., 200.);
  dimassWC_ = iBooker.book1D("11_dimassWC", "Dilepton mass WC", 50, 0., 200.);
  dimassRC_LOGX_ = iBooker.book1D("10_dimassRC_LOGX", "Dilepton mass RC LOG", nbins, &bins[0]);
  dimassWC_LOGX_ = iBooker.book1D("12_dimassWC_LOGX", "Dilepton mass WC LOG", nbins, &bins[0]);
  dimassRC_LOG10_ = iBooker.book1D("dimassRC_LOG10", "Dilepton mass RC LOG", 50, 0., 2.5);
  dimassWC_LOG10_ = iBooker.book1D("dimassWC_LOG10", "Dilepton mass WC LOG", 50, 0., 2.5);

  D_eta_muons_ = iBooker.book1D("13_D_eta_muons", "#Delta eta_muons", 20, -5., 5.);
  D_phi_muons_ = iBooker.book1D("14_D_phi_muons", "#Delta phi_muons", 20, -5., 5.);
  D_eta_elecs_ = iBooker.book1D("D_eta_elecs", "#Delta eta_elecs", 20, -5., 5.);
  D_phi_elecs_ = iBooker.book1D("D_phi_elecs", "#Delta phi_elecs", 20, -5., 5.);
  D_eta_lepts_ = iBooker.book1D("D_eta_lepts", "#Delta eta_lepts", 20, -5., 5.);
  D_phi_lepts_ = iBooker.book1D("D_phi_lepts", "#Delta phi_lepts", 20, -5., 5.);
}

void TopDiLeptonDQM::analyze(const edm::Event& evt, const edm::EventSetup& context) {
  // ------------------------
  //  Global Event Variables
  // ------------------------

  const int N_TriggerPaths = hltPaths_.size();
  const int N_SignalPaths = hltPaths_sig_.size();
  const int N_ControlPaths = hltPaths_trig_.size();

  bool Fired_Signal_Trigger[100] = {false};
  bool Fired_Control_Trigger[100] = {false};

  int N_leptons = 0;
  int N_iso_mu = 0;
  int N_iso_el = 0;

  double DilepMass = 0.;

  double vertex_X = 100.;
  double vertex_Y = 100.;
  double vertex_Z = 100.;

  // ------------------------
  //  Analyze Primary Vertex
  // ------------------------

  edm::Handle<reco::VertexCollection> vertexs;
  evt.getByToken(vertex_, vertexs);

  if (!vertexs.failedToGet()) {
    reco::Vertex primaryVertex = vertexs->front();

    int numberTracks = primaryVertex.tracksSize();
    //    double ndof      = primaryVertex.ndof();
    bool fake = primaryVertex.isFake();

    Ntracks_->Fill(numberTracks);

    if (!fake && numberTracks > 3) {
      vertex_X = primaryVertex.x();
      vertex_Y = primaryVertex.y();
      vertex_Z = primaryVertex.z();
    }
  }

  // -------------------------
  //  Analyze Trigger Results
  // -------------------------

  edm::Handle<TriggerResults> trigResults;
  evt.getByToken(triggerResults_, trigResults);

  if (!trigResults.failedToGet()) {
    int N_Triggers = trigResults->size();

    const edm::TriggerNames& trigName = evt.triggerNames(*trigResults);

    for (int i_Trig = 0; i_Trig < N_Triggers; ++i_Trig) {
      if (trigResults.product()->accept(i_Trig)) {
        // Check for all trigger paths

        for (int i = 0; i < N_TriggerPaths; i++) {
          if (trigName.triggerName(i_Trig) == hltPaths_[i]) {
            Trigs_->Fill(i);
            Trigs_->setBinLabel(i + 1, hltPaths_[i], 1);
          }
        }

        // Check for signal & control trigger paths

        for (int j = 0; j < N_SignalPaths; ++j) {
          if (trigName.triggerName(i_Trig) == hltPaths_sig_[j])
            Fired_Signal_Trigger[j] = true;
        }

        for (int k = 0; k < N_ControlPaths; ++k) {
          if (trigName.triggerName(i_Trig) == hltPaths_trig_[k])
            Fired_Control_Trigger[k] = true;
        }
      }
    }
  }

  // ------------------------
  //  Analyze Muon Isolation
  // ------------------------

  edm::Handle<reco::MuonCollection> muons;
  evt.getByToken(muons_, muons);

  reco::MuonCollection::const_iterator muon;

  if (!muons.failedToGet()) {
    Nmuons_->Fill(muons->size());

    N_leptons = N_leptons + muons->size();

    for (muon = muons->begin(); muon != muons->end(); ++muon) {
      float N_muons = muons->size();
      float Q_muon = muon->charge();

      Nmuons_charge_->Fill(N_muons * Q_muon);

      double track_X = 100.;
      double track_Y = 100.;
      double track_Z = 100.;

      if (muon->isGlobalMuon()) {
        reco::TrackRef track = muon->globalTrack();

        track_X = track->vx();
        track_Y = track->vy();
        track_Z = track->vz();

        VxVy_muons_->Fill(track_X, track_Y);
        Vz_muons_->Fill(track_Z);
      }

      // Vertex and kinematic cuts

      if (track_X > vertex_X_cut_)
        continue;
      if (track_Y > vertex_Y_cut_)
        continue;
      if (track_Z > vertex_Z_cut_)
        continue;
      if (muon->pt() < muon_pT_cut_)
        continue;
      if (abs(muon->eta()) > muon_eta_cut_)
        continue;

      reco::MuonIsolation muIso03 = muon->isolationR03();

      double muonCombRelIso = 1.;

      if (muon->pt() != 0.)
        muonCombRelIso = (muIso03.emEt + muIso03.hadEt + muIso03.hoEt + muIso03.sumPt) / muon->pt();

      MuIso_CombRelIso03_->Fill(muonCombRelIso);

      MuIso_emEt03_->Fill(muIso03.emEt);
      MuIso_hadEt03_->Fill(muIso03.hadEt);
      MuIso_hoEt03_->Fill(muIso03.hoEt);
      MuIso_nJets03_->Fill(muIso03.nJets);
      MuIso_nTracks03_->Fill(muIso03.nTracks);
      MuIso_sumPt03_->Fill(muIso03.sumPt);

      if (muonCombRelIso < muon_iso_cut_)
        ++N_iso_mu;
    }

    Nmuons_iso_->Fill(N_iso_mu);
  }

  // ----------------------------
  //  Analyze Electron Isolation
  // ----------------------------

  edm::Handle<reco::GsfElectronCollection> elecs;
  evt.getByToken(elecs_, elecs);

  reco::GsfElectronCollection::const_iterator elec;

  if (!elecs.failedToGet()) {
    Nelecs_->Fill(elecs->size());

    N_leptons = N_leptons + elecs->size();

    for (elec = elecs->begin(); elec != elecs->end(); ++elec) {
      float N_elecs = elecs->size();
      float Q_elec = elec->charge();
      float HoverE = elec->hcalOverEcal();

      HoverE_elecs_->Fill(HoverE);

      Nelecs_charge_->Fill(N_elecs * Q_elec);

      double track_X = 100.;
      double track_Y = 100.;
      double track_Z = 100.;

      reco::GsfTrackRef track = elec->gsfTrack();

      track_X = track->vx();
      track_Y = track->vy();
      track_Z = track->vz();

      // Vertex and kinematic cuts

      if (track_X > vertex_X_cut_)
        continue;
      if (track_Y > vertex_Y_cut_)
        continue;
      if (track_Z > vertex_Z_cut_)
        continue;
      if (elec->pt() < elec_pT_cut_)
        continue;
      if (abs(elec->eta()) > elec_eta_cut_)
        continue;
      if (HoverE > elec_emf_cut_)
        continue;

      reco::GsfElectron::IsolationVariables elecIso = elec->dr03IsolationVariables();

      double elecCombRelIso = 1.;

      if (elec->et() != 0.)
        elecCombRelIso = (elecIso.ecalRecHitSumEt + elecIso.hcalRecHitSumEt[0] + elecIso.tkSumPt) / elec->et();

      ElecIso_CombRelIso_->Fill(elecCombRelIso);

      ElecIso_cal_->Fill(elecIso.ecalRecHitSumEt);
      ElecIso_trk_->Fill(elecIso.tkSumPt);

      if (elecCombRelIso < elec_iso_cut_)
        ++N_iso_el;
    }

    Nelecs_iso_->Fill(N_iso_el);
  }

  // --------------------
  //  TWO Isolated MUONS
  // --------------------

  if (N_iso_mu > 1) {
    // Vertex cut

    if (vertex_X < vertex_X_cut_ && vertex_Y < vertex_Y_cut_ && vertex_Z < vertex_Z_cut_) {
      ++N_mumu;

      Events_->Fill(1.);

      reco::MuonCollection::const_reference mu1 = muons->at(0);
      reco::MuonCollection::const_reference mu2 = muons->at(1);

      DilepMass = sqrt((mu1.energy() + mu2.energy()) * (mu1.energy() + mu2.energy()) -
                       (mu1.px() + mu2.px()) * (mu1.px() + mu2.px()) - (mu1.py() + mu2.py()) * (mu1.py() + mu2.py()) -
                       (mu1.pz() + mu2.pz()) * (mu1.pz() + mu2.pz()));

      // Opposite muon charges -> Right Charge (RC)

      if (mu1.charge() * mu2.charge() < 0.) {
        dimassRC_LOG10_->Fill(log10(DilepMass));
        dimassRC_->Fill(DilepMass);
        dimassRC_LOGX_->Fill(DilepMass);

        if (DilepMass > MassWindow_down_ && DilepMass < MassWindow_up_) {
          for (muon = muons->begin(); muon != muons->end(); ++muon) {
            pT_muons_->Fill(muon->pt());
            eta_muons_->Fill(muon->eta());
            phi_muons_->Fill(muon->phi());
          }

          D_eta_muons_->Fill(mu1.eta() - mu2.eta());
          D_phi_muons_->Fill(mu1.phi() - mu2.phi());

          // Determinating trigger efficiencies

          for (int k = 0; k < N_SignalPaths; ++k) {
            if (Fired_Signal_Trigger[k] && Fired_Control_Trigger[k])
              ++N_sig[k];

            if (Fired_Control_Trigger[k])
              ++N_trig[k];

            if (N_trig[k] != 0)
              Eff[k] = N_sig[k] / static_cast<float>(N_trig[k]);

            TriggerEff_->setBinContent(k + 1, Eff[k]);
            TriggerEff_->setBinLabel(k + 1, "#frac{[" + hltPaths_sig_[k] + "]}{vs. [" + hltPaths_trig_[k] + "]}", 1);
          }
        }
      }

      // Same muon charges -> Wrong Charge (WC)

      if (mu1.charge() * mu2.charge() > 0.) {
        dimassWC_LOG10_->Fill(log10(DilepMass));
        dimassWC_->Fill(DilepMass);
        dimassWC_LOGX_->Fill(DilepMass);
      }
    }
  }

  // -----------------------------
  //  TWO Isolated LEPTONS (mu/e)
  // -----------------------------

  if (N_iso_el > 0 && N_iso_mu > 0) {
    // Vertex cut

    if (vertex_X < vertex_X_cut_ && vertex_Y < vertex_Y_cut_ && vertex_Z < vertex_Z_cut_) {
      ++N_muel;

      Events_->Fill(2.);

      reco::MuonCollection::const_reference mu1 = muons->at(0);
      reco::GsfElectronCollection::const_reference el1 = elecs->at(0);

      DilepMass = sqrt((mu1.energy() + el1.energy()) * (mu1.energy() + el1.energy()) -
                       (mu1.px() + el1.px()) * (mu1.px() + el1.px()) - (mu1.py() + el1.py()) * (mu1.py() + el1.py()) -
                       (mu1.pz() + el1.pz()) * (mu1.pz() + el1.pz()));

      // Opposite lepton charges -> Right Charge (RC)

      if (mu1.charge() * el1.charge() < 0.) {
        dimassRC_LOG10_->Fill(log10(DilepMass));
        dimassRC_->Fill(DilepMass);
        dimassRC_LOGX_->Fill(DilepMass);

        if (DilepMass > MassWindow_down_ && DilepMass < MassWindow_up_) {
          for (muon = muons->begin(); muon != muons->end(); ++muon) {
            pT_muons_->Fill(muon->pt());
            eta_muons_->Fill(muon->eta());
            phi_muons_->Fill(muon->phi());
          }

          for (elec = elecs->begin(); elec != elecs->end(); ++elec) {
            pT_elecs_->Fill(elec->pt());
            eta_elecs_->Fill(elec->eta());
            phi_elecs_->Fill(elec->phi());
          }

          D_eta_lepts_->Fill(mu1.eta() - el1.eta());
          D_phi_lepts_->Fill(mu1.phi() - el1.phi());

          // Determinating trigger efficiencies

          for (int k = 0; k < N_SignalPaths; ++k) {
            if (Fired_Signal_Trigger[k] && Fired_Control_Trigger[k])
              ++N_sig[k];

            if (Fired_Control_Trigger[k])
              ++N_trig[k];

            if (N_trig[k] != 0)
              Eff[k] = N_sig[k] / static_cast<float>(N_trig[k]);

            TriggerEff_->setBinContent(k + 1, Eff[k]);
            TriggerEff_->setBinLabel(k + 1, "#frac{[" + hltPaths_sig_[k] + "]}{vs. [" + hltPaths_trig_[k] + "]}", 1);
          }
        }
      }

      // Same muon charges -> Wrong Charge (WC)

      if (mu1.charge() * el1.charge() > 0.) {
        dimassWC_LOG10_->Fill(log10(DilepMass));
        dimassWC_->Fill(DilepMass);
        dimassWC_LOGX_->Fill(DilepMass);
      }
    }
  }

  // ------------------------
  //  TWO Isolated ELECTRONS
  // ------------------------

  if (N_iso_el > 1) {
    // Vertex cut

    if (vertex_X < vertex_X_cut_ && vertex_Y < vertex_Y_cut_ && vertex_Z < vertex_Z_cut_) {
      ++N_elel;

      Events_->Fill(3.);

      reco::GsfElectronCollection::const_reference el1 = elecs->at(0);
      reco::GsfElectronCollection::const_reference el2 = elecs->at(1);

      DilepMass = sqrt((el1.energy() + el2.energy()) * (el1.energy() + el2.energy()) -
                       (el1.px() + el2.px()) * (el1.px() + el2.px()) - (el1.py() + el2.py()) * (el1.py() + el2.py()) -
                       (el1.pz() + el2.pz()) * (el1.pz() + el2.pz()));

      // Opposite lepton charges -> Right Charge (RC)

      if (el1.charge() * el2.charge() < 0.) {
        dimassRC_LOG10_->Fill(log10(DilepMass));
        dimassRC_->Fill(DilepMass);
        dimassRC_LOGX_->Fill(DilepMass);

        if (DilepMass > MassWindow_down_ && DilepMass < MassWindow_up_) {
          for (elec = elecs->begin(); elec != elecs->end(); ++elec) {
            pT_elecs_->Fill(elec->pt());
            eta_elecs_->Fill(elec->eta());
            phi_elecs_->Fill(elec->phi());
          }

          D_eta_elecs_->Fill(el1.eta() - el2.eta());
          D_phi_elecs_->Fill(el1.phi() - el2.phi());

          // Determinating trigger efficiencies

          for (int k = 0; k < N_SignalPaths; ++k) {
            if (Fired_Signal_Trigger[k] && Fired_Control_Trigger[k])
              ++N_sig[k];

            if (Fired_Control_Trigger[k])
              ++N_trig[k];

            if (N_trig[k] != 0)
              Eff[k] = N_sig[k] / static_cast<float>(N_trig[k]);

            TriggerEff_->setBinContent(k + 1, Eff[k]);
            TriggerEff_->setBinLabel(k + 1, "#frac{[" + hltPaths_sig_[k] + "]}{vs. [" + hltPaths_trig_[k] + "]}", 1);
          }
        }
      }

      // Same muon charges -> Wrong Charge (WC)

      if (el1.charge() * el2.charge() > 0.) {
        dimassWC_LOG10_->Fill(log10(DilepMass));
        dimassWC_->Fill(DilepMass);
        dimassWC_LOGX_->Fill(DilepMass);
      }
    }
  }
}

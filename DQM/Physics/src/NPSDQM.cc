#include "DQM/Physics/interface/NPSDQM.h"

NPSDQM::NPSDQM(const edm::ParameterSet& iConfig) {
  jetLabels_ = iConfig.getParameter<std::vector<std::string>>("jetLabels");
  for (const auto& label : jetLabels_) {
    jetTokens_.push_back(consumes<edm::View<reco::Jet>>(edm::InputTag(label)));
  }
  
  btagDeepCSVToken_ = consumes<reco::JetTagCollection>(iConfig.getParameter<edm::InputTag>("btagDeepCSV"));
  btagDeepJetToken_ = consumes<reco::JetTagCollection>(iConfig.getParameter<edm::InputTag>("btagDeepJet"));
  btagParticleNetToken_ = consumes<reco::JetTagCollection>(iConfig.getParameter<edm::InputTag>("btagParticleNet"));
  btagRobustParTToken_ = consumes<reco::JetTagCollection>(iConfig.getParameter<edm::InputTag>("btagRobustParT"));
  
  PFMETToken_ = consumes<std::vector<reco::PFMET>>(iConfig.getParameter<edm::InputTag>("pfMETCollection"));
  muonToken_ = consumes<edm::View<reco::Muon>>(iConfig.getParameter<edm::InputTag>("muonCollection"));
  electronToken_ = consumes<edm::View<reco::GsfElectron>>(iConfig.getParameter<edm::InputTag>("electronCollection"));
  photonToken_ = consumes<edm::View<reco::Photon>>(iConfig.getParameter<edm::InputTag>("photonCollection"));
}

void NPSDQM::bookHistograms(DQMStore::IBooker& bei, edm::Run const&, edm::EventSetup const&) {
  bei.setCurrentFolder("Physics/NPS");

  MET_pt = bei.book1D("MET", "Missing E_{T}; GeV", 50, 0, 1000);
  MET_phi = bei.book1D("METphi", "Missing E_{T} #phi", 35, -3.5, 3.5);

  LT = bei.book1D("LT", "Scalar Sum of Lepton p_{T} (pT > 10); GeV", 50, 0, 1000);
  N_Leptons = bei.book1D("N_Leptons", "Number of Leptons (e/#mu, pT > 10)", 10, 0, 10);

  for (int i = 0; i < 3; ++i) {
    std::string idx = std::to_string(i);
    pt_muons[i]  = bei.book1D("pt_muons" + idx, "Muon p_{T} " + idx + "; GeV", 50, 0, 500);
    eta_muons[i] = bei.book1D("eta_muons" + idx, "Muon #eta " + idx, 60, -3.0, 3.0);
    phi_muons[i] = bei.book1D("phi_muons" + idx, "Muon #phi " + idx, 60, -3.14, 3.14);
    mt_muons[i]  = bei.book1D("mt_muons" + idx, "Muon M_{T} " + idx + "; GeV", 50, 0, 500);

    pt_elecs[i]  = bei.book1D("pt_elecs" + idx, "Electron p_{T} " + idx + "; GeV", 50, 0, 500);
    eta_elecs[i] = bei.book1D("eta_elecs" + idx, "Electron #eta " + idx, 60, -3.0, 3.0);
    phi_elecs[i] = bei.book1D("phi_elecs" + idx, "Electron #phi " + idx, 60, -3.14, 3.14);
    mt_elecs[i]  = bei.book1D("mt_elecs" + idx, "Electron M_{T} " + idx + "; GeV", 50, 0, 500);
  }

  pt_photons = bei.book1D("pt_photons", "Photon p_{T} (pT > 10); GeV", 50, 0, 500);
  eta_photons = bei.book1D("eta_photons", "Photon #eta (pT > 10)", 60, -3.0, 3.0);
  phi_photons = bei.book1D("phi_photons", "Photon #phi (pT > 10)", 60, -3.14, 3.14);
  N_photons = bei.book1D("N_photons", "Number of Photons (pT > 10)", 10, 0, 10);

  for (const auto& label : jetLabels_) {
    CentralJet_HT.push_back(bei.book1D("CentralJet_HT_" + label, "Central Jet HT (|#eta| < 2.5, pT > 30) " + label + "; GeV", 50, 0, 2000));
    ForwardJet_HT.push_back(bei.book1D("ForwardJet_HT_" + label, "Forward Jet HT (|#eta| > 2.5, pT > 30) " + label + "; GeV", 50, 0, 2000));
    N_CentralJets.push_back(bei.book1D("N_CentralJets_" + label, "Number of Central Jets " + label, 20, 0, 20));
    N_ForwardJets.push_back(bei.book1D("N_ForwardJets_" + label, "Number of Forward Jets " + label, 20, 0, 20));

    Jet_pt.push_back(bei.book1D("Jet_pt_" + label, "Jet p_{T} " + label + "; GeV", 50, 0, 1000));
    Jet_eta.push_back(bei.book1D("Jet_eta_" + label, "Jet #eta " + label, 60, -3.0, 3.0));
    Jet_phi.push_back(bei.book1D("Jet_phi_" + label, "Jet #phi " + label, 60, -3.14, 3.14));
    Jet_m.push_back(bei.book1D("Jet_m_" + label, "Jet Mass " + label + "; GeV", 50, 0, 200));
    
    Jet_btagDeepCSV.push_back(bei.book1D("Jet_btagDeepCSV_" + label, "Jet b-tag Score (DeepCSV) " + label, 50, 0, 1.0));
    Jet_btagDeepJet.push_back(bei.book1D("Jet_btagDeepJet_" + label, "Jet b-tag Score (DeepJet) " + label, 50, 0, 1.0));
    Jet_btagParticleNet.push_back(bei.book1D("Jet_btagParticleNet_" + label, "Jet b-tag Score (ParticleNet) " + label, 50, 0, 1.0));
    Jet_btagRobustParT.push_back(bei.book1D("Jet_btagRobustParT_" + label, "Jet b-tag Score (RobustParT) " + label, 50, 0, 1.0));
    
    Jet1_pt.push_back(bei.book1D("Jet1_pt_" + label, "Leading Jet p_{T} " + label + "; GeV", 50, 0, 1000));
    Jet1_eta.push_back(bei.book1D("Jet1_eta_" + label, "Leading Jet #eta " + label, 60, -3.0, 3.0));
    Jet1_phi.push_back(bei.book1D("Jet1_phi_" + label, "Leading Jet #phi " + label, 60, -3.14, 3.14));

    dPhi_Jet1_MET.push_back(bei.book1D("dPhi_Jet1_MET_" + label, "|#Delta#phi|(Jet 1, MET) " + label, 60, 0, 3.142));
    dPhi_Jet2_MET.push_back(bei.book1D("dPhi_Jet2_MET_" + label, "|#Delta#phi|(Jet 2, MET) " + label, 60, 0, 3.142));
    dPhi_Jet3_MET.push_back(bei.book1D("dPhi_Jet3_MET_" + label, "|#Delta#phi|(Jet 3, MET) " + label, 60, 0, 3.142));
    dPhi_Jet4_MET.push_back(bei.book1D("dPhi_Jet4_MET_" + label, "|#Delta#phi|(Jet 4, MET) " + label, 60, 0, 3.142));

    Jet_chef.push_back(bei.book1D("Jet_chef_" + label, "Jet CHEF " + label, 50, 0, 1.0));
    Jet_nhef.push_back(bei.book1D("Jet_nhef_" + label, "Jet NHEF " + label, 50, 0, 1.0));
    Jet_cemf.push_back(bei.book1D("Jet_cemf_" + label, "Jet CEMF " + label, 50, 0, 1.0));
    Jet_nemf.push_back(bei.book1D("Jet_nemf_" + label, "Jet NEMF " + label, 50, 0, 1.0));
  }
}

void NPSDQM::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<std::vector<reco::PFMET>> pfMETCollection;
  double met_pt = 0.0;
  double met_phi = 0.0;
  bool validMET = false;
  
  if (iEvent.getByToken(PFMETToken_, pfMETCollection) && !pfMETCollection->empty()) {
    met_pt = (*pfMETCollection)[0].pt();
    met_phi = (*pfMETCollection)[0].phi();
    MET_pt->Fill(met_pt);
    MET_phi->Fill(met_phi);
    validMET = true;
  }

  double current_LT = 0.0;
  int n_leps = 0;

  edm::Handle<edm::View<reco::Muon>> muonCollection;
  if (iEvent.getByToken(muonToken_, muonCollection)) {
    int mu_count = 0;
    for (auto const& muon : *muonCollection) {
      if (muon.pt() <= 10.0) continue;
      current_LT += muon.pt();
      n_leps++;
      if (mu_count < 3) {
        pt_muons[mu_count]->Fill(muon.pt());
        eta_muons[mu_count]->Fill(muon.eta());
        phi_muons[mu_count]->Fill(muon.phi());
        if (validMET) {
          double mt = std::sqrt(2.0 * muon.pt() * met_pt * (1.0 - std::cos(reco::deltaPhi(muon.phi(), met_phi))));
          mt_muons[mu_count]->Fill(mt);
        }
      }
      mu_count++;
    }
  }

  edm::Handle<edm::View<reco::GsfElectron>> electronCollection;
  if (iEvent.getByToken(electronToken_, electronCollection)) {
    int el_count = 0;
    for (auto const& elec : *electronCollection) {
      if (elec.pt() <= 10.0) continue;
      current_LT += elec.pt();
      n_leps++;
      if (el_count < 3) {
        pt_elecs[el_count]->Fill(elec.pt());
        eta_elecs[el_count]->Fill(elec.eta());
        phi_elecs[el_count]->Fill(elec.phi());
        if (validMET) {
          double mt = std::sqrt(2.0 * elec.pt() * met_pt * (1.0 - std::cos(reco::deltaPhi(elec.phi(), met_phi))));
          mt_elecs[el_count]->Fill(mt);
        }
      }
      el_count++;
    }
  }

  LT->Fill(current_LT);
  N_Leptons->Fill(n_leps);

  edm::Handle<edm::View<reco::Photon>> photonCollection;
  if (iEvent.getByToken(photonToken_, photonCollection)) {
    int n_ph = 0;
    for (auto const& photon : *photonCollection) {
      if (photon.pt() <= 10.0) continue;
      pt_photons->Fill(photon.pt());
      eta_photons->Fill(photon.eta());
      phi_photons->Fill(photon.phi());
      n_ph++;
    }
    N_photons->Fill(n_ph);
  }

  for (unsigned int icoll = 0; icoll < jetLabels_.size(); ++icoll) {
    edm::Handle<edm::View<reco::Jet>> pfJetCollection;
    if (!iEvent.getByToken(jetTokens_[icoll], pfJetCollection)) continue;

    edm::Handle<reco::JetTagCollection> btagDeepCSVColl;
    bool validDeepCSV = iEvent.getByToken(btagDeepCSVToken_, btagDeepCSVColl);
    
    edm::Handle<reco::JetTagCollection> btagDeepJetColl;
    bool validDeepJet = iEvent.getByToken(btagDeepJetToken_, btagDeepJetColl);
    
    edm::Handle<reco::JetTagCollection> btagParticleNetColl;
    bool validParticleNet = iEvent.getByToken(btagParticleNetToken_, btagParticleNetColl);
    
    edm::Handle<reco::JetTagCollection> btagRobustParTColl;
    bool validRobustParT = iEvent.getByToken(btagRobustParTToken_, btagRobustParTColl);

    double ht_central = 0.0, ht_forward = 0.0;
    int n_central = 0, n_forward = 0;
    int jet_count = 0;

    for (unsigned int ijet = 0; ijet < pfJetCollection->size(); ++ijet) {
      edm::RefToBase<reco::Jet> jetRef = pfJetCollection->refAt(ijet);
      auto const& jet = *jetRef;

      Jet_pt[icoll]->Fill(jet.pt());
      Jet_eta[icoll]->Fill(jet.eta());
      Jet_phi[icoll]->Fill(jet.phi());
      Jet_m[icoll]->Fill(jet.mass());

      if (jet_count == 0) {
        Jet1_pt[icoll]->Fill(jet.pt());
        Jet1_eta[icoll]->Fill(jet.eta());
        Jet1_phi[icoll]->Fill(jet.phi());
      }

      if (jet_count < 4 && validMET) {
        double dphi = std::abs(reco::deltaPhi(jet.phi(), met_phi));
        if (jet_count == 0) dPhi_Jet1_MET[icoll]->Fill(dphi);
        else if (jet_count == 1) dPhi_Jet2_MET[icoll]->Fill(dphi);
        else if (jet_count == 2) dPhi_Jet3_MET[icoll]->Fill(dphi);
        else if (jet_count == 3) dPhi_Jet4_MET[icoll]->Fill(dphi);
      }

      // Safely fill all four B-Tagging algorithms!
      if (validDeepCSV) Jet_btagDeepCSV[icoll]->Fill((*btagDeepCSVColl)[jetRef]);
      if (validDeepJet) Jet_btagDeepJet[icoll]->Fill((*btagDeepJetColl)[jetRef]);
      if (validParticleNet) Jet_btagParticleNet[icoll]->Fill((*btagParticleNetColl)[jetRef]);
      if (validRobustParT) Jet_btagRobustParT[icoll]->Fill((*btagRobustParTColl)[jetRef]);

      reco::PFJet const* pfjet = dynamic_cast<reco::PFJet const*>(&jet);
      if (pfjet != nullptr) {
        Jet_chef[icoll]->Fill(pfjet->chargedHadronEnergyFraction());
        Jet_nhef[icoll]->Fill(pfjet->neutralHadronEnergyFraction());
        Jet_cemf[icoll]->Fill(pfjet->chargedEmEnergyFraction());
        Jet_nemf[icoll]->Fill(pfjet->neutralEmEnergyFraction());
      }

      if (jet.pt() > 30.0) {
        if (std::abs(jet.eta()) < 2.5) {
          ht_central += jet.pt();
          n_central++;
        } else {
          ht_forward += jet.pt();
          n_forward++;
        }
      }
      jet_count++;
    }
    
    CentralJet_HT[icoll]->Fill(ht_central);
    ForwardJet_HT[icoll]->Fill(ht_forward);
    N_CentralJets[icoll]->Fill(n_central);
    N_ForwardJets[icoll]->Fill(n_forward);
  }
}

DEFINE_FWK_MODULE(NPSDQM);

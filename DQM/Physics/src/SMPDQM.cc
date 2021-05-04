#include "DQM/Physics/src/SMPDQM.h"

using namespace std;
using namespace reco;

struct SortByPt

{
  bool operator()(const TLorentzVector& a, const TLorentzVector& b) const { return a.Pt() > b.Pt(); }
};
SMPDQM::SMPDQM(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed
  muons_ = consumes<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>("muonCollection"));
  pvs_ = consumes<edm::View<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("pvs"));

  elecs_ = consumes<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("elecCollection"));
  jets_ = consumes<edm::View<reco::PFJet>>(iConfig.getParameter<edm::InputTag>("jets"));

  for (edm::InputTag const& tag : iConfig.getParameter<std::vector<edm::InputTag>>("mets"))
    mets_.push_back(consumes<edm::View<reco::MET>>(tag));
}

SMPDQM::~SMPDQM() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}
// ------------ method called for each event  ------------
void SMPDQM::bookHistograms(DQMStore::IBooker& bei, edm::Run const&, edm::EventSetup const&) {
  bei.setCurrentFolder("Physics/SMP");

  NPV = bei.book1D("NPV", "Number of primary vertices", 40, 0., 80.);
  MET = bei.book1D("MET", "MET", 100, 0.0, 200);
  METphi = bei.book1D("METphi", "#phi(MET)", 50, -3.14, 3.14);

  pt_muons = bei.book1D("pt_muons", "p_{T}(muons)", 40, 0., 200.);
  eta_muons = bei.book1D("eta_muons", "#eta(muons)", 50, -5., 5.);
  phi_muons = bei.book1D("phi_muons", "#phi(muons)", 32, -3.2, 3.2);
  muIso_CombRelIso03 = bei.book1D("muIso_CombRelIso03", "Iso_{rel}^{#mu}", 20, 0., 1.);
  Nmuons = bei.book1D("Nmuons", "Number of muons", 20, 0., 10.);
  isGlobalmuon = bei.book1D("isGlobalmuon", "isGlobalmuon", 2, 0, 1);
  isTrackermuon = bei.book1D("isTrackermuon", "isTrackermuon", 2, 0, 1);
  isStandalonemuon = bei.book1D("isStandalonemuon", "isStandalonemuon", 2, 0, 1);
  isPFmuon = bei.book1D("isPFmuon", "isPFmuon", 2, 0, 1);
  muIso_TrackerBased03 = bei.book1D("muIso_TrackerBased03", "Iso_{trk03}^{#mu}", 20, 0, 10);

  Nelecs = bei.book1D("Nelecs", "Number of electrons", 20, 0., 10.);
  HoverE_elecs = bei.book1D("HoverE_elecs", "HoverE", 50, 0., 1.);
  pt_elecs = bei.book1D("pt_elecs", "p_{T}(elecs)", 40, 0., 200.);
  eta_elecs = bei.book1D("eta_elecs", "#eta(elecs)", 50, -5., 5.);
  phi_elecs = bei.book1D("phi_elecs", "#phielecs)", 32, -3.2, 3.2);
  elIso_cal = bei.book1D("elIso_cal", "Iso_{cal}^{el}", 21, -1., 20.);
  elIso_trk = bei.book1D("elIso_trk", "Iso_{trk}^{el}", 21, -2., 40.);
  elIso_CombRelIso = bei.book1D("elIso_CombRelIso", "Iso_{rel}^{el}", 20, 0., 1.);

  PFJetpt = bei.book1D("PFJetpt", "p_{T}(jets)", 100, 0.0, 100);
  PFJeteta = bei.book1D("PFJeteta", "#eta(jets)", 50, -2.5, 2.5);
  PFJetphi = bei.book1D("PFJetphi", "#phi(jets)", 50, -3.14, 3.14);
  PFJetMulti = bei.book1D("PFJetMulti", "jet multiplicity", 5, -0.5, 4.5);
  PFJetRapidity = bei.book1D("PFJetRapidity", "y(jets)", 50, -6.0, 6.0);
  mjj = bei.book1D("mjj", "m_{jj}", 100, 0, 1000);
  detajj = bei.book1D("detajj", "#Delta#etajj", 20, 0, 5);

  dphi_lepMET = bei.book1D("dphi_lepMET", "#Delta#phi(lep,MET)", 60, -3.2, 3.2);
  mass_lepMET = bei.book1D("mass_lepMET", "m(lep,MET)", 200, 0, 200);
  pt_lepMET = bei.book1D("pt_lepMET", "p_{T}(lep,MET)", 200, 0, 200);
  detall = bei.book1D("detall", "#Delta#etall", 20, -5, 5);
  dphill = bei.book1D("dphill", "#Delta#phill", 20, -6.4, 6.4);
  mll = bei.book1D("mll", "mll", 200, 0, 200);
  etall = bei.book1D("etall", "#Delta#etall", 60, -6, 6);
  ptll = bei.book1D("ptll", "p_{T}ll", 200, 0, 200);
  mjj = bei.book1D("mjj", "mjj", 100, 0, 1000);
  detajj = bei.book1D("detajj", "#Delta#etajj", 20, 0, 5);

  dphi_lepjet1 = bei.book1D("dphi_lepjet1", "#Delta#phi(lep,jet1)", 60, -3.2, 3.2);

  dphi_lep1jet1 = bei.book1D("dphi_lep1jet1", "#Delta#phi(lep1,jet1)", 60, -3.2, 3.2);
  dphi_lep2jet1 = bei.book1D("dphi_lep2jet1", "#Delta#phi(lep2,jet1)", 60, -3.2, 3.2);
}
void SMPDQM::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::vector<TLorentzVector> recoPFJets;
  recoPFJets.clear();
  TLorentzVector imet;
  imet.Clear();
  std::vector<TLorentzVector> selected_recoPFJets;
  selected_recoPFJets.clear();
  std::vector<TLorentzVector> selected_lep;
  selected_lep.clear();

  for (std::vector<edm::EDGetTokenT<edm::View<reco::MET>>>::const_iterator met_ = mets_.begin(); met_ != mets_.end();
       ++met_) {
    edm::Handle<edm::View<reco::MET>> met;
    if (!iEvent.getByToken(*met_, met))
      continue;
    if (met->begin() != met->end()) {
      MET->Fill(met->begin()->et());
      METphi->Fill(met->begin()->phi());
      imet.SetPtEtaPhiM(met->begin()->et(), 0., met->begin()->phi(), 0.0);
    }
  }

  // Muons

  edm::Handle<edm::View<reco::Vertex>> pvs;
  if (!iEvent.getByToken(pvs_, pvs)) {
    return;
  }

  unsigned int pvMult = 0;

  for (edm::View<reco::Vertex>::const_iterator pv = pvs->begin(); pv != pvs->end(); ++pv) {
    if (pv->position().Rho() < 2 && abs(pv->position().z()) <= 24. && pv->ndof() > 4 && !pv->isFake()) {
      pvMult++;
    }
  }
  NPV->Fill(pvMult);

  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByToken(muons_, muons);
  reco::MuonCollection::const_iterator mu;
  if (!muons.failedToGet()) {
    Nmuons->Fill(muons->size());

    for (mu = muons->begin(); mu != muons->end(); ++mu) {
      if (mu->pt() < 3.0)
        continue;
      TLorentzVector Mu;
      Mu.SetPtEtaPhiM(mu->pt(), mu->eta(), mu->phi(), 0.0);
      selected_lep.push_back(Mu);
      pt_muons->Fill(mu->pt());
      eta_muons->Fill(mu->eta());
      phi_muons->Fill(mu->phi());
      isGlobalmuon->Fill(mu->isGlobalMuon());
      isTrackermuon->Fill(mu->isTrackerMuon());
      isStandalonemuon->Fill(mu->isStandAloneMuon());
      isPFmuon->Fill(mu->isPFMuon());

      reco::MuonIsolation muIso03 = mu->isolationR03();
      double muonCombRelIso = 1.;

      muonCombRelIso = (muIso03.emEt + muIso03.hadEt + muIso03.hoEt + muIso03.sumPt) / mu->pt();

      muIso_TrackerBased03->Fill(muIso03.sumPt / mu->pt());
      muIso_CombRelIso03->Fill(muonCombRelIso);

    }  //size of muons

  }  // muons

  // electrons

  edm::Handle<reco::GsfElectronCollection> elecs;
  iEvent.getByToken(elecs_, elecs);
  reco::GsfElectronCollection::const_iterator elec;

  if (!elecs.failedToGet()) {
    Nelecs->Fill(elecs->size());

    for (elec = elecs->begin(); elec != elecs->end(); ++elec) {
      if (elec->pt() < 5.0)
        continue;
      TLorentzVector El;
      El.SetPtEtaPhiM(elec->pt(), elec->eta(), elec->phi(), 0.0);
      selected_lep.push_back(El);

      HoverE_elecs->Fill(elec->hcalOverEcal());
      pt_elecs->Fill(elec->pt());
      eta_elecs->Fill(elec->eta());
      phi_elecs->Fill(elec->phi());

      reco::GsfTrackRef track = elec->gsfTrack();
      reco::GsfElectron::IsolationVariables elecIso = elec->dr03IsolationVariables();

      double elecCombRelIso = 1.;

      elecCombRelIso = (elecIso.ecalRecHitSumEt + elecIso.hcalDepth1TowerSumEt + elecIso.tkSumPt) / elec->pt();
      elIso_CombRelIso->Fill(elecCombRelIso);
      elIso_cal->Fill(elecIso.ecalRecHitSumEt);
      elIso_trk->Fill(elecIso.tkSumPt);
    }

  }  // electrons
  // jets

  edm::Handle<edm::View<reco::PFJet>> jets;
  if (!iEvent.getByToken(jets_, jets)) {
    return;
  }

  for (edm::View<reco::PFJet>::const_iterator jet = jets->begin(); jet != jets->end(); ++jet) {
    if (jet->pt() < 15.0)
      continue;
    TLorentzVector ijet;
    ijet.SetPtEtaPhiM(jet->pt(), jet->eta(), jet->phi(), jet->mass());
    recoPFJets.push_back(ijet);
  }

  std::sort(recoPFJets.begin(), recoPFJets.end(), SortByPt());
  std::sort(selected_lep.begin(), selected_lep.end(), SortByPt());

  for (unsigned int i = 0; i < recoPFJets.size(); i++) {
    bool goodjet = false;
    for (unsigned int j = 0; j < selected_lep.size(); j++) {
      if (recoPFJets[i].DeltaR(selected_lep[j]) > 0.4) {
        goodjet = true;
        continue;
      } else {
        goodjet = false;
        break;
      }
    }
    if (goodjet) {
      TLorentzVector temp;
      temp.Clear();
      temp.SetPtEtaPhiM(recoPFJets[i].Pt(), recoPFJets[i].Eta(), recoPFJets[i].Phi(), recoPFJets[i].M());
      selected_recoPFJets.push_back(temp);
    }
  }

  std::sort(selected_recoPFJets.begin(), selected_recoPFJets.end(), SortByPt());  // for safety
  int njet = 0;
  for (unsigned int k = 0; k < selected_recoPFJets.size(); k++) {
    if (k > 4)
      break;
    else {
      njet++;
      PFJetpt->Fill(selected_recoPFJets.at(k).Pt());
      PFJeteta->Fill(selected_recoPFJets.at(k).Eta());
      PFJetphi->Fill(selected_recoPFJets.at(k).Phi());
      PFJetRapidity->Fill(selected_recoPFJets.at(k).Rapidity());
    }
  }
  PFJetMulti->Fill(njet);

  // now we have selected jet and lepton collections

  if (selected_lep.size() > 1) {
    detall->Fill(selected_lep[0].Eta() - selected_lep[1].Eta());
    dphill->Fill(selected_lep[0].DeltaPhi(selected_lep[1]));
    mll->Fill((selected_lep[0] + selected_lep[1]).M());
    ptll->Fill((selected_lep[0] + selected_lep[1]).Pt());
    etall->Fill((selected_lep[0] + selected_lep[1]).Eta());
    if (!selected_recoPFJets.empty()) {
      dphi_lep1jet1->Fill(selected_recoPFJets[0].DeltaPhi(selected_lep[0]));
      dphi_lep2jet1->Fill(selected_recoPFJets[0].DeltaPhi(selected_lep[1]));
    }
  }

  else if (selected_lep.size() == 1) {
    dphi_lepMET->Fill(selected_lep[0].DeltaPhi(imet));
    mass_lepMET->Fill((selected_lep[0] + imet).M());
    pt_lepMET->Fill((selected_lep[0] + imet).Pt());
    if (!selected_recoPFJets.empty()) {
      dphi_lepjet1->Fill(selected_recoPFJets[0].DeltaPhi(selected_lep[0]));
    }
  }  // W case

  else {
    // std::cout << "zero lepton case" << endl;
  }
  if (selected_recoPFJets.size() > 1) {
    detajj->Fill(abs(selected_recoPFJets[0].Eta() - selected_recoPFJets[1].Eta()));
    mjj->Fill((selected_recoPFJets[0] + selected_recoPFJets[1]).M());
  }

}  // analyze

//define this as a plug-in
//DEFINE_FWK_MODULE(SMPDQM);

//dilepton eta,pt  and phi using lorentz vectors
//first five jets
// dphi(lepton jet)
//vbf detajj, mjj

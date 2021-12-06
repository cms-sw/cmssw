#include "DQM/Physics/src/ExoticaDQM.h"

using namespace edm;
using namespace std;
using namespace reco;
using namespace trigger;

typedef vector<string> vstring;

//
// -- Constructor
//
ExoticaDQM::ExoticaDQM(const edm::ParameterSet& ps) {
  edm::LogInfo("ExoticaDQM") << " Starting ExoticaDQM "
                             << "\n";

  typedef std::vector<edm::InputTag> vtag;

  // Get parameters from configuration file
  // Trigger
  TriggerToken_ = consumes<TriggerResults>(ps.getParameter<edm::InputTag>("TriggerResults"));
  HltPaths_ = ps.getParameter<vector<string> >("HltPaths");
  //
  VertexToken_ = consumes<reco::VertexCollection>(ps.getParameter<InputTag>("vertexCollection"));
  //
  ElectronToken_ = consumes<reco::GsfElectronCollection>(ps.getParameter<InputTag>("electronCollection"));
  //
  MuonToken_ = consumes<reco::MuonCollection>(ps.getParameter<InputTag>("muonCollection"));
  //
  PhotonToken_ = consumes<reco::PhotonCollection>(ps.getParameter<InputTag>("photonCollection"));
  //
  PFJetToken_ = consumes<reco::PFJetCollection>(ps.getParameter<InputTag>("pfJetCollection"));
  //
  DiJetPFJetCollection_ = ps.getParameter<std::vector<edm::InputTag> >("DiJetPFJetCollection");
  for (std::vector<edm::InputTag>::const_iterator jetlabel = DiJetPFJetCollection_.begin(),
                                                  jetlabelEnd = DiJetPFJetCollection_.end();
       jetlabel != jetlabelEnd;
       ++jetlabel) {
    DiJetPFJetToken_.push_back(consumes<reco::PFJetCollection>(*jetlabel));
  }
  //
  PFMETToken_ = consumes<reco::PFMETCollection>(ps.getParameter<InputTag>("pfMETCollection"));
  //
  ecalBarrelRecHitToken_ = consumes<EBRecHitCollection>(
      ps.getUntrackedParameter<InputTag>("ecalBarrelRecHit", InputTag("reducedEcalRecHitsEB")));
  //
  ecalEndcapRecHitToken_ = consumes<EERecHitCollection>(
      ps.getUntrackedParameter<InputTag>("ecalEndcapRecHit", InputTag("reducedEcalRecHitsEE")));
  //
  TrackToken_ = consumes<reco::TrackCollection>(ps.getParameter<InputTag>("trackCollection"));
  //
  MuonDispToken_ = consumes<reco::TrackCollection>(ps.getParameter<InputTag>("displacedMuonCollection"));
  //
  MuonDispSAToken_ = consumes<reco::TrackCollection>(ps.getParameter<InputTag>("displacedSAMuonCollection"));
  //
  GenParticleToken_ = consumes<reco::GenParticleCollection>(ps.getParameter<InputTag>("genParticleCollection"));

  JetCorrectorToken_ = consumes<reco::JetCorrector>(ps.getParameter<edm::InputTag>("jetCorrector"));

  magFieldToken_ = esConsumes();
  //Cuts - MultiJets
  jetID = new reco::helper::JetIDHelper(ps.getParameter<ParameterSet>("JetIDParams"), consumesCollector());

  //Varibles and Cuts for each Module:
  //Dijet
  dijet_PFJet1_pt_cut_ = ps.getParameter<double>("dijet_PFJet1_pt_cut");
  dijet_PFJet2_pt_cut_ = ps.getParameter<double>("dijet_PFJet2_pt_cut");
  //DiMuon
  dimuon_Muon1_pt_cut_ = ps.getParameter<double>("dimuon_Muon1_pt_cut");
  dimuon_Muon2_pt_cut_ = ps.getParameter<double>("dimuon_Muon2_pt_cut");
  //DiElectron
  dielectron_Electron1_pt_cut_ = ps.getParameter<double>("dielectron_Electron2_pt_cut");
  dielectron_Electron2_pt_cut_ = ps.getParameter<double>("dielectron_Electron2_pt_cut");
  //DiPhoton
  diphoton_Photon1_pt_cut_ = ps.getParameter<double>("diphoton_Photon2_pt_cut");
  diphoton_Photon2_pt_cut_ = ps.getParameter<double>("diphoton_Photon2_pt_cut");
  //MonoJet
  monojet_PFJet_pt_cut_ = ps.getParameter<double>("monojet_PFJet_pt_cut");
  monojet_PFJet_met_cut_ = ps.getParameter<double>("monojet_PFJet_met_cut");
  //MonoMuon
  monomuon_Muon_pt_cut_ = ps.getParameter<double>("monomuon_Muon_pt_cut");
  monomuon_Muon_met_cut_ = ps.getParameter<double>("monomuon_Muon_met_cut");
  //MonoElectron
  monoelectron_Electron_pt_cut_ = ps.getParameter<double>("monoelectron_Electron_pt_cut");
  monoelectron_Electron_met_cut_ = ps.getParameter<double>("monoelectron_Electron_met_cut");
  //MonoPhoton
  monophoton_Photon_pt_cut_ = ps.getParameter<double>("monophoton_Photon_pt_cut");
  monophoton_Photon_met_cut_ = ps.getParameter<double>("monophoton_Photon_met_cut");
  // Displaced lepton or jet
  dispFermion_eta_cut_ = ps.getParameter<double>("dispFermion_eta_cut");
  dispFermion_pt_cut_ = ps.getParameter<double>("dispFermion_pt_cut");
}

//
// -- Destructor
//
ExoticaDQM::~ExoticaDQM() {
  edm::LogInfo("ExoticaDQM") << " Deleting ExoticaDQM "
                             << "\n";
}

//
//  -- Book histograms
//
void ExoticaDQM::bookHistograms(DQMStore::IBooker& bei, edm::Run const&, edm::EventSetup const&) {
  bei.cd();

  //--- DiJet
  for (unsigned int icoll = 0; icoll < DiJetPFJetCollection_.size(); ++icoll) {
    std::stringstream ss;
    ss << "Physics/Exotica/Dijets/" << DiJetPFJetCollection_[icoll].label();
    bei.setCurrentFolder(ss.str());
    //bei.setCurrentFolder("Physics/Exotica/Dijets");
    dijet_PFJet_pt.push_back(bei.book1D("dijet_PFJet_pt", "Pt of PFJet (GeV)", 50, 30.0, 5000));
    dijet_PFJet_eta.push_back(bei.book1D("dijet_PFJet_eta", "#eta (PFJet)", 50, -2.5, 2.5));
    dijet_PFJet_phi.push_back(bei.book1D("dijet_PFJet_phi", "#phi (PFJet)", 50, -3.14, 3.14));
    dijet_PFJet_rapidity.push_back(bei.book1D("dijet_PFJet_rapidity", "Rapidity (PFJet)", 50, -6.0, 6.0));
    dijet_PFJet_mass.push_back(bei.book1D("dijet_PFJet_mass", "Mass (PFJets)", 50, 0., 500.));
    dijet_deltaPhiPFJet1PFJet2.push_back(
        bei.book1D("dijet_deltaPhiPFJet1PFJet2", "#Delta#phi(Leading PFJet, Sub PFJet)", 40, 0., 3.15));
    dijet_deltaEtaPFJet1PFJet2.push_back(
        bei.book1D("dijet_deltaEtaPFJet1PFJet2", "#Delta#eta(Leading PFJet, Sub PFJet)", 40, -5., 5.));
    dijet_deltaRPFJet1PFJet2.push_back(
        bei.book1D("dijet_deltaRPFJet1PFJet2", "#DeltaR(Leading PFJet, Sub PFJet)", 50, 0., 6.));
    dijet_invMassPFJet1PFJet2.push_back(
        bei.book1D("dijet_invMassPFJet1PFJet2", "Leading PFJet, SubLeading PFJet Invariant mass (GeV)", 50, 0., 8000.));
    dijet_PFchef.push_back(bei.book1D("dijet_PFchef", "Leading PFJet CHEF", 50, 0.0, 1.0));
    dijet_PFnhef.push_back(bei.book1D("dijet_PFnhef", "Leading PFJet NHEF", 50, 0.0, 1.0));
    dijet_PFcemf.push_back(bei.book1D("dijet_PFcemf", "Leading PFJet CEMF", 50, 0.0, 1.0));
    dijet_PFnemf.push_back(bei.book1D("dijet_PFnemf", "Leading PFJEt NEMF", 50, 0.0, 1.0));
    dijet_PFJetMulti.push_back(bei.book1D("dijet_PFJetMulti", "No. of PFJets", 10, 0., 10.));
  }
  //--- DiMuon
  bei.setCurrentFolder("Physics/Exotica/DiMuons");
  dimuon_Muon_pt = bei.book1D("dimuon_Muon_pt", "Pt of Muon (GeV)", 50, 30.0, 2000);
  dimuon_Muon_eta = bei.book1D("dimuon_Muon_eta", "#eta (Muon)", 50, -2.5, 2.5);
  dimuon_Muon_phi = bei.book1D("dimuon_Muon_phi", "#phi (Muon)", 50, -3.14, 3.14);
  dimuon_Charge = bei.book1D("dimuon_Charge", "Charge of the Muon", 10, -5., 5.);
  dimuon_deltaEtaMuon1Muon2 =
      bei.book1D("dimuon_deltaEtaMuon1Muon2", "#Delta#eta(Leading Muon, Sub Muon)", 40, -5., 5.);
  dimuon_deltaPhiMuon1Muon2 =
      bei.book1D("dimuon_deltaPhiMuon1Muon2", "#Delta#phi(Leading Muon, Sub Muon)", 40, 0., 3.15);
  dimuon_deltaRMuon1Muon2 = bei.book1D("dimuon_deltaRMuon1Muon2", "#DeltaR(Leading Muon, Sub Muon)", 50, 0., 6.);
  dimuon_invMassMuon1Muon2 =
      bei.book1D("dimuon_invMassMuon1Muon2", "Leading Muon, SubLeading Muon Low Invariant mass (GeV)", 50, 500., 4500.);
  dimuon_MuonMulti = bei.book1D("dimuon_MuonMulti", "No. of Muons", 10, 0., 10.);
  //--- DiElectrons
  bei.setCurrentFolder("Physics/Exotica/DiElectrons");
  dielectron_Electron_pt = bei.book1D("dielectron_Electron_pt", "Pt of Electron (GeV)", 50, 30.0, 2000);
  dielectron_Electron_eta = bei.book1D("dielectron_Electron_eta", "#eta (Electron)", 50, -2.5, 2.5);
  dielectron_Electron_phi = bei.book1D("dielectron_Electron_phi", "#phi (Electron)", 50, -3.14, 3.14);
  dielectron_Charge = bei.book1D("dielectron_Charge", "Charge of the Electron", 10, -5., 5.);
  dielectron_deltaEtaElectron1Electron2 =
      bei.book1D("dielectron_deltaEtaElectron1Electron2", "#Delta#eta(Leading Electron, Sub Electron)", 40, -5., 5.);
  dielectron_deltaPhiElectron1Electron2 =
      bei.book1D("dielectron_deltaPhiElectron1Electron2", "#Delta#phi(Leading Electron, Sub Electron)", 40, 0., 3.15);
  dielectron_deltaRElectron1Electron2 =
      bei.book1D("dielectron_deltaRElectron1Electron2", "#DeltaR(Leading Electron, Sub Electron)", 50, 0., 6.);
  dielectron_invMassElectron1Electron2 = bei.book1D("dielectron_invMassElectron1Electron2",
                                                    "Leading Electron, SubLeading Electron Invariant mass (GeV)",
                                                    50,
                                                    500.,
                                                    4500.);
  dielectron_ElectronMulti = bei.book1D("dielectron_ElectronMulti", "No. of Electrons", 10, 0., 10.);
  //--- DiPhotons
  bei.setCurrentFolder("Physics/Exotica/DiPhotons");
  diphoton_Photon_energy = bei.book1D("diphoton_Photon_energy", "Energy of Photon (GeV)", 50, 30.0, 300);
  diphoton_Photon_et = bei.book1D("diphoton_Photon_et", "Et of Photon (GeV)", 50, 30.0, 300);
  diphoton_Photon_pt = bei.book1D("diphoton_Photon_pt", "Pt of Photon (GeV)", 50, 30.0, 300);
  diphoton_Photon_eta = bei.book1D("diphoton_Photon_eta", "#eta (Photon)", 50, -2.5, 2.5);
  diphoton_Photon_etasc = bei.book1D("diphoton_Photon_etasc", "#eta sc(Photon)", 50, -2.5, 2.5);
  diphoton_Photon_phi = bei.book1D("diphoton_Photon_phi", "#phi (Photon)", 50, -3.14, 3.14);
  diphoton_Photon_hovere_eb = bei.book1D("diphoton_Photon_hovere_eb", "H/E (Photon) EB", 50, 0., 0.50);
  diphoton_Photon_hovere_ee = bei.book1D("diphoton_Photon_hovere_ee", "H/E (Photon) EE", 50, 0., 0.50);
  diphoton_Photon_sigmaietaieta_eb =
      bei.book1D("diphoton_Photon_sigmaietaieta_eb", "#sigma_{i #eta i #eta} (Photon) EB", 50, 0., 0.03);
  diphoton_Photon_sigmaietaieta_ee =
      bei.book1D("diphoton_Photon_sigmaietaieta_ee", "#sigma_{i #eta i #eta} (Photon) EE", 50, 0., 0.03);
  diphoton_Photon_trksumptsolidconedr03_eb =
      bei.book1D("diphoton_Photon_trksumptsolidconedr03_eb", "TrkSumPtDr03 (Photon) EB", 50, 0., 15.);
  diphoton_Photon_trksumptsolidconedr03_ee =
      bei.book1D("diphoton_Photon_trksumptsolidconedr03_ee", "TrkSumPtDr03 (Photon) EE", 50, 0., 15.);
  diphoton_Photon_e1x5e5x5_eb = bei.book1D("diphoton_Photon_e1x5e5x5_eb", "E_{1x5}/E_{5x5} (Photon) EB", 50, 0., 1.);
  diphoton_Photon_e1x5e5x5_ee = bei.book1D("diphoton_Photon_e1x5e5x5_ee", "E_{1x5}/E_{5x5} (Photon) EE", 50, 0., 1.);
  diphoton_Photon_e2x5e5x5_eb = bei.book1D("diphoton_Photon_e2x5e5x5_eb", "E_{2x5}/E_{5x5} (Photon) EB", 50, 0., 1.);
  diphoton_Photon_e2x5e5x5_ee = bei.book1D("diphoton_Photon_e2x5e5x5_ee", "E_{2x5}/E_{5x5} (Photon) EE", 50, 0., 1.);
  diphoton_deltaEtaPhoton1Photon2 =
      bei.book1D("diphoton_deltaEtaPhoton1Photon2", "#Delta#eta(SubLeading Photon, Sub Photon)", 40, -5., 5.);
  diphoton_deltaPhiPhoton1Photon2 =
      bei.book1D("diphoton_deltaPhiPhoton1Photon2", "#Delta#phi(SubLeading Photon, Sub Photon)", 40, 0., 3.15);
  diphoton_deltaRPhoton1Photon2 =
      bei.book1D("diphoton_deltaRPhoton1Photon2", "#DeltaR(SubLeading Photon, Sub Photon)", 50, 0., 6.);
  diphoton_invMassPhoton1Photon2 = bei.book1D(
      "diphoton_invMassPhoton1Photon2", "SubLeading Photon, SubSubLeading Photon Invariant mass (GeV)", 50, 500., 4500.);
  diphoton_PhotonMulti = bei.book1D("diphoton_PhotonMulti", "No. of Photons", 10, 0., 10.);
  //--- MonoJet
  bei.setCurrentFolder("Physics/Exotica/MonoJet");
  monojet_PFJet_pt = bei.book1D("monojet_PFJet_pt", "Pt of MonoJet (GeV)", 50, 30.0, 1000);
  monojet_PFJet_eta = bei.book1D("monojet_PFJet_eta", "#eta(MonoJet)", 50, -2.5, 2.5);
  monojet_PFJet_phi = bei.book1D("monojet_PFJet_phi", "#phi(MonoJet)", 50, -3.14, 3.14);
  monojet_PFMet = bei.book1D("monojet_PFMet", "Pt of PFMET (GeV)", 40, 0.0, 1000);
  monojet_PFMet_phi = bei.book1D("monojet_PFMet_phi", "#phi(PFMET #phi)", 50, -3.14, 3.14);
  monojet_PFJetPtOverPFMet = bei.book1D("monojet_PFJetPtOverPFMet", "Pt of MonoJet/MET (GeV)", 40, 0.0, 5.);
  monojet_deltaPhiPFJetPFMet = bei.book1D("monojet_deltaPhiPFJetPFMet", "#Delta#phi(MonoJet, PFMet)", 40, 0., 3.15);
  monojet_PFchef = bei.book1D("monojet_PFchef", "MonojetJet CHEF", 50, 0.0, 1.0);
  monojet_PFnhef = bei.book1D("monojet_PFnhef", "MonojetJet NHEF", 50, 0.0, 1.0);
  monojet_PFcemf = bei.book1D("monojet_PFcemf", "MonojetJet CEMF", 50, 0.0, 1.0);
  monojet_PFnemf = bei.book1D("monojet_PFnemf", "MonojetJet NEMF", 50, 0.0, 1.0);
  monojet_PFJetMulti = bei.book1D("monojet_PFJetMulti", "No. of PFJets", 10, 0., 10.);
  //--- MonoMuon
  bei.setCurrentFolder("Physics/Exotica/MonoMuon");
  monomuon_Muon_pt = bei.book1D("monomuon_Muon_pt", "Pt of Monomuon (GeV)", 50, 30.0, 2000);
  monomuon_Muon_eta = bei.book1D("monomuon_Muon_eta", "#eta(Monomuon)", 50, -2.5, 2.5);
  monomuon_Muon_phi = bei.book1D("monomuon_Muon_phi", "#phi(Monomuon)", 50, -3.14, 3.14);
  monomuon_Charge = bei.book1D("monomuon_Charge", "Charge of the MonoMuon", 10, -5., 5.);
  monomuon_PFMet = bei.book1D("monomuon_PFMet", "Pt of PFMET (GeV)", 40, 0.0, 2000);
  monomuon_PFMet_phi = bei.book1D("monomuon_PFMet_phi", "PFMET #phi", 50, -3.14, 3.14);
  monomuon_MuonPtOverPFMet = bei.book1D("monomuon_MuonPtOverPFMet", "Pt of Monomuon/PFMet", 40, 0.0, 5.);
  monomuon_deltaPhiMuonPFMet = bei.book1D("monomuon_deltaPhiMuonPFMet", "#Delta#phi(Monomuon, PFMet)", 40, 0., 3.15);
  monomuon_TransverseMass = bei.book1D("monomuon_TransverseMass", "Transverse Mass M_{T} GeV", 40, 200., 3000.);
  monomuon_MuonMulti = bei.book1D("monomuon_MuonMulti", "No. of Muons", 10, 0., 10.);
  //--- MonoElectron
  bei.setCurrentFolder("Physics/Exotica/MonoElectron");
  monoelectron_Electron_pt = bei.book1D("monoelectron_Electron_pt", "Pt of Monoelectron (GeV)", 50, 30.0, 4000);
  monoelectron_Electron_eta = bei.book1D("monoelectron_Electron_eta", "#eta(MonoElectron)", 50, -2.5, 2.5);
  monoelectron_Electron_phi = bei.book1D("monoelectron_Electron_phi", "#phi(MonoElectron)", 50, -3.14, 3.14);
  monoelectron_Charge = bei.book1D("monoelectron_Charge", "Charge of the MonoElectron", 10, -5., 5.);
  monoelectron_PFMet = bei.book1D("monoelectron_PFMet", "Pt of PFMET (GeV)", 40, 0.0, 4000);
  monoelectron_PFMet_phi = bei.book1D("monoelectron_PFMet_phi", "PFMET #phi", 50, -3.14, 3.14);
  monoelectron_ElectronPtOverPFMet =
      bei.book1D("monoelectron_ElectronPtOverPFMet", "Pt of Monoelectron/PFMet", 40, 0.0, 5.);
  monoelectron_deltaPhiElectronPFMet =
      bei.book1D("monoelectron_deltaPhiElectronPFMet", "#Delta#phi(MonoElectron, PFMet)", 40, 0., 3.15);
  monoelectron_TransverseMass = bei.book1D("monoelectron_TransverseMass", "Transverse Mass M_{T} GeV", 40, 200., 4000.);
  monoelectron_ElectronMulti = bei.book1D("monoelectron_ElectronMulti", "No. of Electrons", 10, 0., 10.);

  //--- DiPhotons
  bei.setCurrentFolder("Physics/Exotica/MonoPhotons");
  monophoton_Photon_energy = bei.book1D("monophoton_Photon_energy", "Energy of Leading Photon (GeV)", 50, 30.0, 1000);
  monophoton_Photon_et = bei.book1D("monophoton_Photon_et", "Et of Leading Photon (GeV)", 50, 30.0, 1000);
  monophoton_Photon_pt = bei.book1D("monophoton_Photon_pt", "Pt of Leading Photon (GeV)", 50, 30.0, 1000);
  monophoton_Photon_eta = bei.book1D("monophoton_Photon_eta", "#eta (Leading Photon)", 50, -2.5, 2.5);
  monophoton_Photon_etasc = bei.book1D("monophoton_Photon_etasc", "#eta sc(Leading Photon)", 50, -2.5, 2.5);
  monophoton_Photon_phi = bei.book1D("monophoton_Photon_phi", "#phi(Leading Photon)", 50, -3.14, 3.14);
  monophoton_Photon_hovere = bei.book1D("monophoton_Photon_hovere", "H/E (Leading Photon)", 50, 0., 0.50);
  monophoton_Photon_sigmaietaieta =
      bei.book1D("monophoton_Photon_sigmaietaieta", "#sigma_{i #eta i #eta} (Leading Photon)", 50, 0., 0.03);
  monophoton_Photon_trksumptsolidconedr03 =
      bei.book1D("monophoton_Photon_trksumptsolidconedr03", "TrkSumPtDr03 (Leading Photon)", 50, 0., 15.);
  monophoton_Photon_e1x5e5x5 = bei.book1D("monophoton_Photon_e1x5e5x5", "E_{1x5}/E_{5x5} (Leading Photon)", 50, 0., 1.);
  monophoton_Photon_e2x5e5x5 = bei.book1D("monophoton_Photon_e2x5e5x5", "E_{2x5}/E_{5x5} (Leading Photon)", 50, 0., 1.);
  monophoton_PFMet = bei.book1D("monophoton_PFMet", "Pt of PFMET (GeV)", 40, 0.0, 1000);
  monophoton_PFMet_phi = bei.book1D("monophoton_PFMet_phi", "PFMET #phi", 50, -3.14, 3.14);
  monophoton_PhotonPtOverPFMet = bei.book1D("monophoton_PhotonPtOverPFMet", "Pt of Monophoton/PFMet", 40, 0.0, 5.);
  monophoton_deltaPhiPhotonPFMet =
      bei.book1D("monophoton_deltaPhiPhotonPFMet", "#Delta#phi(SubLeading Photon, PFMet)", 40, 0., 3.15);
  monophoton_PhotonMulti = bei.book1D("monophoton_PhotonMulti", "No. of Photons", 10, 0., 10.);

  //--- Displaced Leptons (filled using only leptons from long-lived stop decay).
  bei.setCurrentFolder("Physics/Exotica/DisplacedFermions");
  dispElec_track_effi_lxy = bei.bookProfile("dispElec_track_effi_lxy",
                                            "Electron channel; transverse decay length (cm); track reco efficiency",
                                            10,
                                            0.,
                                            100.,
                                            -999.,
                                            999,
                                            "");
  dispElec_elec_effi_lxy = bei.bookProfile("dispElec_elec_effi_lxy",
                                           "Electron channel; transverse decay length (cm); electron reco efficiency",
                                           10,
                                           0.,
                                           100.,
                                           -999.,
                                           999,
                                           "");
  dispMuon_track_effi_lxy = bei.bookProfile("dispMuon_track_effi_lxy",
                                            "Muon channel; transverse decay length (cm); track reco efficiency",
                                            10,
                                            0.,
                                            100.,
                                            -999.,
                                            999,
                                            "");
  dispMuon_muon_effi_lxy = bei.bookProfile("dispMuon_muon_effi_lxy",
                                           "Muon channel; transverse decay length (cm); muon reco efficiency",
                                           10,
                                           0.,
                                           100.,
                                           -999.,
                                           999,
                                           "");
  dispMuon_muonDisp_effi_lxy =
      bei.bookProfile("dispMuon_muonDisp_effi_lxy",
                      "Muon channel; transverse decay length (cm); displacedMuon reco efficiency",
                      10,
                      0.,
                      100.,
                      -999.,
                      999,
                      "");
  dispMuon_muonDispSA_effi_lxy =
      bei.bookProfile("dispMuon_muonDispSA_effi_lxy",
                      "Muon channel; transverse decay length (cm); displacedSAMuon reco efficiency",
                      10,
                      0.,
                      400.,
                      -999.,
                      999,
                      "");
  //--- Displaced Jets (filled using only tracks or jets from long-lived stop decay).
  dispJet_track_effi_lxy = bei.bookProfile("dispJet_track_effi_lxy",
                                           "Jet channel; transverse decay length (cm); track reco efficiency",
                                           10,
                                           0.,
                                           100.,
                                           -999.,
                                           999,
                                           "");

  bei.cd();
}

//
//  -- Analyze
//
void ExoticaDQM::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // objects

  //Trigger
  bool ValidTriggers = iEvent.getByToken(TriggerToken_, TriggerResults_);
  if (!ValidTriggers)
    return;

  // Vertices
  bool ValidVertices = iEvent.getByToken(VertexToken_, VertexCollection_);
  if (!ValidVertices)
    return;

  // Electrons
  bool ValidGedGsfElectron = iEvent.getByToken(ElectronToken_, ElectronCollection_);
  if (!ValidGedGsfElectron)
    return;

  // Muons
  bool ValidPFMuon = iEvent.getByToken(MuonToken_, MuonCollection_);
  if (!ValidPFMuon)
    return;

  // Jets
  bool ValidPFJet = iEvent.getByToken(PFJetToken_, pfJetCollection_);
  pfjets = *pfJetCollection_;
  if (!ValidPFJet)
    return;

  // PFMETs
  bool ValidPFMET = iEvent.getByToken(PFMETToken_, pfMETCollection_);
  if (!ValidPFMET)
    return;

  // Photons
  bool ValidCaloPhoton = iEvent.getByToken(PhotonToken_, PhotonCollection_);
  if (!ValidCaloPhoton)
    return;

  // Tracks
  bool ValidTracks = iEvent.getByToken(TrackToken_, TrackCollection_);
  if (!ValidTracks)
    return;

  // Special collections for displaced particles
  iEvent.getByToken(MuonDispToken_, MuonDispCollection_);
  iEvent.getByToken(MuonDispSAToken_, MuonDispSACollection_);

  // MC truth
  bool ValidGenParticles = iEvent.getByToken(GenParticleToken_, GenCollection_);

  // JetCorrector
  bool ValidJetCorrector = iEvent.getByToken(JetCorrectorToken_, JetCorrector_);

  //Trigger

  int N_Triggers = TriggerResults_->size();
  int N_GoodTriggerPaths = HltPaths_.size();
  bool triggered_event = false;
  const edm::TriggerNames& trigName = iEvent.triggerNames(*TriggerResults_);
  for (int i_Trig = 0; i_Trig < N_Triggers; ++i_Trig) {
    if (TriggerResults_.product()->accept(i_Trig)) {
      for (int n = 0; n < N_GoodTriggerPaths; n++) {
        if (trigName.triggerName(i_Trig).find(HltPaths_[n]) != std::string::npos) {
          triggered_event = true;
        }
      }
    }
  }
  if (triggered_event == false)
    return;

  for (int i = 0; i < 2; i++) {
    //Jets
    PFJetPx[i] = 0.;
    PFJetPy[i] = 0.;
    PFJetPt[i] = 0.;
    PFJetEta[i] = 0.;
    PFJetPhi[i] = 0.;
    PFJetNHEF[i] = 0.;
    PFJetCHEF[i] = 0.;
    PFJetNEMF[i] = 0.;
    PFJetCEMF[i] = 0.;
    //Muons
    MuonPx[i] = 0.;
    MuonPy[i] = 0.;
    MuonPt[i] = 0.;
    MuonEta[i] = 0.;
    MuonPhi[i] = 0.;
    MuonCharge[i] = 0.;
    //Electrons
    ElectronPx[i] = 0.;
    ElectronPy[i] = 0.;
    ElectronPt[i] = 0.;
    ElectronEta[i] = 0.;
    ElectronPhi[i] = 0.;
    ElectronCharge[i] = 0.;
    //Photons
    PhotonEnergy[i] = 0.;
    PhotonPt[i] = 0.;
    PhotonEt[i] = 0.;
    PhotonEta[i] = 0.;
    PhotonEtaSc[i] = 0.;
    PhotonPhi[i] = 0.;
    PhotonHoverE[i] = 0.;
    PhotonSigmaIetaIeta[i] = 0.;
    PhotonTrkSumPtSolidConeDR03[i] = 0.;
    PhotonE1x5E5x5[i] = 0.;
    PhotonE2x5E5x5[i] = 0.;
  }

  //Getting information from the RecoObjects
  dijet_countPFJet_ = 0;
  monojet_countPFJet_ = 0;

  PFJetCollection::const_iterator pfjet_ = pfjets.begin();
  for (; pfjet_ != pfjets.end(); ++pfjet_) {
    double scale = 1.;
    if (ValidJetCorrector)
      scale = JetCorrector_->correction(*pfjet_);
    if (scale * pfjet_->pt() > PFJetPt[0]) {
      PFJetPt[1] = PFJetPt[0];
      PFJetPx[1] = PFJetPx[0];
      PFJetPy[1] = PFJetPy[0];
      PFJetEta[1] = PFJetEta[0];
      PFJetPhi[1] = PFJetPhi[0];
      PFJetRapidity[1] = PFJetRapidity[0];
      PFJetMass[1] = PFJetMass[0];
      PFJetNHEF[1] = PFJetNHEF[0];
      PFJetCHEF[1] = PFJetCHEF[0];
      PFJetNEMF[1] = PFJetNEMF[0];
      PFJetCEMF[1] = PFJetCEMF[0];
      //
      PFJetPt[0] = scale * pfjet_->pt();
      PFJetPx[0] = scale * pfjet_->px();
      PFJetPy[0] = scale * pfjet_->py();
      PFJetEta[0] = pfjet_->eta();
      PFJetPhi[0] = pfjet_->phi();
      PFJetRapidity[0] = pfjet_->rapidity();
      PFJetMass[0] = pfjet_->mass();
      PFJetNHEF[0] = pfjet_->neutralHadronEnergyFraction();
      PFJetCHEF[0] = pfjet_->chargedHadronEnergyFraction();
      PFJetNEMF[0] = pfjet_->neutralEmEnergyFraction();
      PFJetCEMF[0] = pfjet_->chargedEmEnergyFraction();
    } else if (scale * pfjet_->pt() < PFJetPt[0] && scale * pfjet_->pt() > PFJetPt[1]) {
      PFJetPt[1] = scale * pfjet_->pt();
      PFJetPx[1] = scale * pfjet_->px();
      PFJetPy[1] = scale * pfjet_->py();
      PFJetEta[1] = pfjet_->eta();
      PFJetPhi[1] = pfjet_->phi();
      PFJetRapidity[1] = pfjet_->rapidity();
      PFJetMass[1] = pfjet_->mass();
      PFJetNHEF[1] = pfjet_->neutralHadronEnergyFraction();
      PFJetCHEF[1] = pfjet_->chargedHadronEnergyFraction();
      PFJetNEMF[1] = pfjet_->neutralEmEnergyFraction();
      PFJetCEMF[1] = pfjet_->chargedEmEnergyFraction();
    } else {
    }
    if (scale * pfjet_->pt() > dijet_PFJet1_pt_cut_)
      dijet_countPFJet_++;
    if (scale * pfjet_->pt() > dijet_PFJet1_pt_cut_)
      monojet_countPFJet_++;
  }

  VertexCollection vertexCollection = *(VertexCollection_.product());
  reco::VertexCollection::const_iterator primaryVertex_ = vertexCollection.begin();

  dimuon_countMuon_ = 0;
  monomuon_countMuon_ = 0;
  reco::MuonCollection::const_iterator muon_ = MuonCollection_->begin();
  for (; muon_ != MuonCollection_->end(); muon_++) {
    // Muon High Pt ID
    bool HighPt = false;
    if (muon_->isGlobalMuon() && muon_->globalTrack()->hitPattern().numberOfValidMuonHits() > 0 &&
        muon_->numberOfMatchedStations() > 1 && muon_->innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5 &&
        muon_->innerTrack()->hitPattern().numberOfValidPixelHits() > 0 &&
        muon_->muonBestTrack()->ptError() / muon_->muonBestTrack()->pt() < 0.3 &&
        fabs(muon_->muonBestTrack()->dxy(primaryVertex_->position())) < 0.2 &&
        fabs(muon_->bestTrack()->dz(primaryVertex_->position())) < 0.5 && fabs(muon_->eta()) < 2.1)
      HighPt = true;

    if (HighPt == true) {
      if (muon_->pt() > MuonPt[0]) {
        MuonPt[1] = MuonPt[0];
        MuonPx[1] = MuonPx[0];
        MuonPy[1] = MuonPy[0];
        MuonEta[1] = MuonEta[0];
        MuonPhi[1] = MuonPhi[0];
        MuonCharge[1] = MuonCharge[0];
        //
        MuonPt[0] = muon_->pt();
        MuonPx[0] = muon_->px();
        MuonPy[0] = muon_->py();
        MuonEta[0] = muon_->eta();
        MuonPhi[0] = muon_->phi();
        MuonCharge[0] = muon_->charge();
      }
    }
    if (muon_->pt() > dimuon_Muon1_pt_cut_)
      dimuon_countMuon_++;
    if (muon_->pt() > dimuon_Muon1_pt_cut_)
      monomuon_countMuon_++;
  }

  dielectron_countElectron_ = 0;
  monoelectron_countElectron_ = 0;
  reco::GsfElectronCollection::const_iterator electron_ = ElectronCollection_->begin();
  for (; electron_ != ElectronCollection_->end(); electron_++) {
    //HEEP Selection 4.1 (some cuts)
    if (electron_->e5x5() <= 0)
      continue;
    if (electron_->gsfTrack().isNull())
      continue;
    bool HEPP_ele = false;
    double sceta = electron_->caloPosition().eta();
    double dEtaIn = fabs(electron_->deltaEtaSuperClusterTrackAtVtx());
    double dPhiIn = fabs(electron_->deltaPhiSuperClusterTrackAtVtx());
    double HoverE = electron_->hadronicOverEm();
    int missingHits = electron_->gsfTrack()->hitPattern().numberOfLostTrackerHits(HitPattern::MISSING_INNER_HITS);
    double dxy = electron_->gsfTrack()->dxy(primaryVertex_->position());
    double tkIso = electron_->dr03TkSumPt();
    double e2x5Fraction = electron_->e2x5Max() / electron_->e5x5();
    double e1x5Fraction = electron_->e1x5() / electron_->e5x5();
    double scSigmaIetaIeta = electron_->scSigmaIEtaIEta();
    if (electron_->ecalDriven() && electron_->pt() > 35.) {
      if (fabs(sceta) < 1.442) {  // barrel
        if (fabs(dEtaIn) < 0.005 && fabs(dPhiIn) < 0.06 && HoverE < 0.05 && tkIso < 5. && missingHits <= 1 &&
            fabs(dxy) < 0.02 && (e2x5Fraction > 0.94 || e1x5Fraction > 0.83))
          HEPP_ele = true;
      } else if (fabs(sceta) > 1.56 && fabs(sceta) < 2.5) {  // endcap
        if (fabs(dEtaIn) < 0.007 && fabs(dPhiIn) < 0.06 && HoverE < 0.05 && tkIso < 5. && missingHits <= 1 &&
            fabs(dxy) < 0.02 && scSigmaIetaIeta < 0.03)
          HEPP_ele = true;
      }
    }
    //
    if (HEPP_ele == false)
      continue;
    if (electron_->pt() > ElectronPt[0]) {
      ElectronPt[1] = ElectronPt[0];
      ElectronPx[1] = ElectronPx[0];
      ElectronPy[1] = ElectronPy[0];
      ElectronEta[1] = ElectronEta[0];
      ElectronPhi[1] = ElectronPhi[0];
      ElectronCharge[1] = ElectronCharge[0];
      //
      ElectronPt[0] = electron_->pt();
      ElectronPx[0] = electron_->px();
      ElectronPy[0] = electron_->py();
      ElectronEta[0] = electron_->eta();
      ElectronPhi[0] = electron_->phi();
      ElectronCharge[0] = electron_->charge();
    }
    if (electron_->pt() > dielectron_Electron1_pt_cut_)
      dielectron_countElectron_++;
    if (electron_->pt() > dielectron_Electron1_pt_cut_)
      monoelectron_countElectron_++;
  }

  diphoton_countPhoton_ = 0.;
  reco::PhotonCollection::const_iterator photon_ = PhotonCollection_->begin();
  for (; photon_ != PhotonCollection_->end(); ++photon_) {
    if (photon_->pt() > PhotonPt[0]) {
      PhotonEnergy[1] = PhotonEnergy[0];
      PhotonPt[1] = PhotonPt[0];
      PhotonEt[1] = PhotonEt[0];
      PhotonEta[1] = PhotonEta[0];
      PhotonEtaSc[1] = PhotonEtaSc[0];
      PhotonPhi[1] = PhotonPhi[0];
      PhotonHoverE[1] = PhotonHoverE[0];
      PhotonSigmaIetaIeta[1] = PhotonSigmaIetaIeta[0];
      PhotonTrkSumPtSolidConeDR03[1] = PhotonTrkSumPtSolidConeDR03[0];
      PhotonE1x5E5x5[1] = PhotonE1x5E5x5[0];
      PhotonE2x5E5x5[1] = PhotonE2x5E5x5[0];

      PhotonEnergy[0] = photon_->energy();
      PhotonPt[0] = photon_->pt();
      PhotonEt[0] = photon_->et();
      PhotonEta[0] = photon_->eta();
      PhotonEtaSc[0] = photon_->caloPosition().eta();
      PhotonPhi[0] = photon_->phi();
      PhotonHoverE[0] = photon_->hadronicOverEm();
      PhotonSigmaIetaIeta[0] = photon_->sigmaIetaIeta();
      PhotonTrkSumPtSolidConeDR03[0] = photon_->trkSumPtSolidConeDR03();
      PhotonE1x5E5x5[0] = photon_->e1x5() / photon_->e5x5();
      PhotonE2x5E5x5[0] = photon_->e2x5() / photon_->e5x5();

      if (photon_->pt() > dielectron_Electron1_pt_cut_)
        diphoton_countPhoton_++;
    }
  }

  //
  // Analyze
  //

  //Resonances
  analyzeDiJets(iEvent);
  analyzeDiMuons(iEvent);
  analyzeDiElectrons(iEvent);
  analyzeDiPhotons(iEvent);

  //MonoSearches
  analyzeMonoJets(iEvent);
  analyzeMonoMuons(iEvent);
  analyzeMonoElectrons(iEvent);

  //LongLived
  if (ValidGenParticles) {
    analyzeDisplacedLeptons(iEvent, iSetup);
    analyzeDisplacedJets(iEvent, iSetup);
  }
}

void ExoticaDQM::analyzeDisplacedLeptons(const Event& iEvent, const edm::EventSetup& iSetup) {
  //=== This is designed to run on MC events in which a pair of long-lived stop quarks each decay to a displaced lepton + displaced b jet.

  // Initialisation

  const unsigned int stop1 = 1000006;  // PDG identifier of top squark1
  const unsigned int stop2 = 2000006;  // PDG identifier of top squark2
  const float deltaRcut = 0.01;        // Cone size for matching reco to true leptons.
  const float invPtcut = 0.1;          // Cut in 1/Pt consistency for matching reco tracks to genParticles.

  //--- Measure the efficiency to reconstruct leptons from long-lived stop quark decay.

  for (const reco::GenParticle& gen : *GenCollection_) {
    unsigned int idPdg = abs(gen.pdgId());
    // Find electrons/muons from long-lived stop decay.
    if (idPdg == stop1 || idPdg == stop2) {
      unsigned int nDau = gen.numberOfDaughters();
      for (unsigned int i = 0; i < nDau; i++) {
        const reco::GenParticle* dau = (const reco::GenParticle*)gen.daughter(i);
        // Only measure efficiency using leptons passing pt & eta cuts. (The pt cut is almost irrelevant, since leptons from stop decay are hard).
        if (fabs(dau->eta()) < dispFermion_eta_cut_ && dau->pt() > dispFermion_pt_cut_) {
          unsigned int pdgIdDau = abs(dau->pdgId());

          if (pdgIdDau == 11 || pdgIdDau == 13) {  // electron or muon from stop decay

            // Get transverse decay length of stop quark.
            float lxy = dau->vertex().rho();

            // Get momentum vector of daughter genParticle trajectory extrapolated to beam-line.
            GlobalVector genP = this->getGenParticleTrajectoryAtBeamline(iSetup, dau);

            if (pdgIdDau == 11) {  // electron from stop decay

              // Find matching reco track if any.
              bool recoedTrk = false;
              for (const reco::Track& trk : *TrackCollection_) {
                if (reco::deltaR(genP, trk) < deltaRcut && fabs(1 / dau->pt() - 1 / trk.pt()) < invPtcut) {
                  //cout<<"MATCH ELEC TRK "<<dau->pt()<<" "<<trk.pt()<<" "<<reco::deltaR(genP, trk)<<endl;
                  recoedTrk = true;
                }
              }
              dispElec_track_effi_lxy->Fill(lxy, recoedTrk);

              // Find matching reco electron if any.
              bool recoedE = false;
              for (const reco::GsfElectron& eReco : *ElectronCollection_) {
                if (reco::deltaR(genP, eReco) < deltaRcut && fabs(1 / dau->pt() - 1 / eReco.pt()) < invPtcut)
                  recoedE = true;
              }
              dispElec_elec_effi_lxy->Fill(lxy, recoedE);

            } else if (pdgIdDau == 13) {  // muon from stop decay

              // Find matching reco track if any.
              bool recoedTrk = false;
              for (const reco::Track& trk : *TrackCollection_) {
                if (reco::deltaR(genP, trk) < deltaRcut && fabs(1 / dau->pt() - 1 / trk.pt()) < invPtcut) {
                  //cout<<"MATCH MUON TRK "<<dau->pt()<<" "<<trk.pt()<<" "<<reco::deltaR(genP, trk)<<endl;
                  recoedTrk = true;
                }
              }
              dispMuon_track_effi_lxy->Fill(lxy, recoedTrk);

              // Find matching reco muon, if any, in normal global muon collection.
              bool recoedMu = false;
              for (const reco::Muon& muReco : *MuonCollection_) {
                if (reco::deltaR(genP, muReco) < deltaRcut && fabs(1 / dau->pt() - 1 / muReco.pt()) < invPtcut)
                  recoedMu = true;
              }
              dispMuon_muon_effi_lxy->Fill(lxy, recoedMu);

              // Find matching reco muon, if any, in displaced global muon collection.
              bool recoedMuDisp = false;
              for (const reco::Track& muDispReco : *MuonDispCollection_) {
                if (reco::deltaR(genP, muDispReco) < deltaRcut && fabs(1 / dau->pt() - 1 / muDispReco.pt()) < invPtcut)
                  recoedMuDisp = true;
              }
              dispMuon_muonDisp_effi_lxy->Fill(lxy, recoedMuDisp);

              // Find matching reco muon, if any, in displaced SA muon collection.
              bool recoedMuDispSA = false;
              for (const reco::Track& muDispSAReco : *MuonDispSACollection_) {
                if (reco::deltaR(genP, muDispSAReco) < deltaRcut &&
                    fabs(1 / dau->pt() - 1 / muDispSAReco.pt()) < invPtcut)
                  recoedMuDispSA = true;
              }
              dispMuon_muonDispSA_effi_lxy->Fill(lxy, recoedMuDispSA);
            }
          }
        }
      }
    }
  }
}
void ExoticaDQM::analyzeDisplacedJets(const Event& iEvent, const edm::EventSetup& iSetup) {
  //=== This is designed to run on MC events in which a pair of long-lived stop quarks each decay to a displaced lepton + displaced b jet.

  // Initialisation

  // Define function to identify R-hadrons containing stop quarks from PDG particle code.
  // N.B. Jets originate not just from stop quark, but also from its partner SM quark inside the R hadron.
  auto isRhadron = [](unsigned int pdgId) { return (pdgId / 100) == 10006 || (pdgId / 1000) == 1006; };

  const float deltaRcut = 0.01;  // Cone size for matching reco tracks to genParticles.
  const float invPtcut = 0.1;    // Cut in 1/Pt consistency for matching reco tracks to genParticles.

  //--- Measure the efficiency to reconstruct tracks in jet(s) from long-lived stop quark decay.

  for (const reco::GenParticle& gen : *GenCollection_) {
    unsigned int idPdg = abs(gen.pdgId());
    // Only measure efficiency using charged e, mu pi, K, p
    if (idPdg == 11 || idPdg == 13 || idPdg == 211 || idPdg == 321 || idPdg == 2212) {
      // Only measure efficiency using leptons passing pt & eta cuts. (The pt cut is almost irrelevant, since leptons from stop decay are hard).
      if (fabs(gen.eta()) < dispFermion_eta_cut_ && gen.pt() > dispFermion_pt_cut_) {
        // Check if this particle came (maybe indirectly) from an R hadron decay.
        const reco::GenParticle* genMoth = &gen;
        const reco::GenParticle* genRhadron = nullptr;
        bool foundParton = false;
        while (genMoth->numberOfMothers() > 0) {
          genMoth = (const reco::GenParticle*)genMoth->mother(0);
          unsigned int idPdgMoth = abs(genMoth->pdgId());
          // Check that the R-hadron decayed via a quark/gluon before yielding genParticle "gen".
          // This ensures that gen is from the jet, and not a lepton produced directly from the stop quark decay.
          if ((idPdgMoth >= 1 && idPdgMoth <= 6) || idPdgMoth == 21)
            foundParton = true;
          // Note if ancestor was R hadron
          if (isRhadron(idPdgMoth)) {
            genRhadron = genMoth;
            break;
          }
        }

        if (foundParton && genRhadron != nullptr) {  // This GenParticle came (maybe indirectly) from an R hadron decay.

          // Get transverse decay length of R hadron.
          float lxy = genRhadron->daughter(0)->vertex().rho();

          // Get momentum vector of genParticle trajectory extrapolated to beam-line.
          GlobalVector genP = this->getGenParticleTrajectoryAtBeamline(iSetup, &gen);

          // Find matching reco track if any.
          bool recoedTrk = false;
          for (const reco::Track& trk : *TrackCollection_) {
            if (reco::deltaR(genP, trk) < deltaRcut && fabs(1 / gen.pt() - 1 / trk.pt()) < invPtcut) {
              //cout<<"MATCH TRK "<<gen.pt()<<" "<<trk.pt()<<" "<<reco::deltaR(gen, trk)<<endl;
              recoedTrk = true;
            }
          }
          dispJet_track_effi_lxy->Fill(lxy, recoedTrk);
        }
      }
    }
  }
}
GlobalVector ExoticaDQM::getGenParticleTrajectoryAtBeamline(const edm::EventSetup& iSetup,
                                                            const reco::GenParticle* gen) {
  //=== Estimate the momentum vector that a GenParticle would have at its trajectory's point of closest
  //=== approach to the beam-line.

  // Get the magnetic field
  const MagneticField* theMagField = &iSetup.getData(magFieldToken_);

  // Make FreeTrajectoryState of this gen particle
  FreeTrajectoryState fts(GlobalPoint(gen->vx(), gen->vy(), gen->vz()),
                          GlobalVector(gen->px(), gen->py(), gen->pz()),
                          gen->charge(),
                          theMagField);

  // Get trajectory closest to beam line
  TSCBLBuilderNoMaterial tscblBuilder;
  const BeamSpot beamspot;  // Simple beam-spot at (0,0,0). Good enough.
  TrajectoryStateClosestToBeamLine tsAtClosestApproach = tscblBuilder(fts, beamspot);

  GlobalVector p = tsAtClosestApproach.trackStateAtPCA().momentum();

  return p;
}

void ExoticaDQM::analyzeDiJets(const Event& iEvent) {
  for (unsigned int icoll = 0; icoll < DiJetPFJetCollection_.size(); ++icoll) {
    dijet_countPFJet_ = 0;
    bool ValidDiJetPFJets = iEvent.getByToken(DiJetPFJetToken_[icoll], DiJetpfJetCollection_);
    if (!ValidDiJetPFJets)
      continue;
    DiJetpfjets = *DiJetpfJetCollection_;
    for (int i = 0; i < 2; i++) {
      PFJetPx[i] = 0.;
      PFJetPy[i] = 0.;
      PFJetPt[i] = 0.;
      PFJetEta[i] = 0.;
      PFJetPhi[i] = 0.;
      PFJetNHEF[i] = 0.;
      PFJetCHEF[i] = 0.;
      PFJetNEMF[i] = 0.;
      PFJetCEMF[i] = 0.;
    }
    PFJetCollection::const_iterator DiJetpfjet_ = DiJetpfjets.begin();
    for (; DiJetpfjet_ != DiJetpfjets.end(); ++DiJetpfjet_) {
      double scale = 1.;
      if (scale * DiJetpfjet_->pt() > PFJetPt[0]) {
        PFJetPt[1] = PFJetPt[0];
        PFJetPx[1] = PFJetPx[0];
        PFJetPy[1] = PFJetPy[0];
        PFJetEta[1] = PFJetEta[0];
        PFJetPhi[1] = PFJetPhi[0];
        PFJetRapidity[1] = DiJetpfjet_->rapidity();
        PFJetMass[1] = DiJetpfjet_->mass();
        PFJetNHEF[1] = PFJetNHEF[0];
        PFJetCHEF[1] = PFJetCHEF[0];
        PFJetNEMF[1] = PFJetNEMF[0];
        PFJetCEMF[1] = PFJetCEMF[0];
        PFJetPt[0] = scale * DiJetpfjet_->pt();
        PFJetPx[0] = scale * DiJetpfjet_->px();
        PFJetPy[0] = scale * DiJetpfjet_->py();
        PFJetEta[0] = DiJetpfjet_->eta();
        PFJetPhi[0] = DiJetpfjet_->phi();
        PFJetRapidity[0] = DiJetpfjet_->rapidity();
        PFJetMass[0] = DiJetpfjet_->mass();
        PFJetNHEF[0] = DiJetpfjet_->neutralHadronEnergyFraction();
        PFJetCHEF[0] = DiJetpfjet_->chargedHadronEnergyFraction();
        PFJetNEMF[0] = DiJetpfjet_->neutralEmEnergyFraction();
        PFJetCEMF[0] = DiJetpfjet_->chargedEmEnergyFraction();
      } else if (scale * DiJetpfjet_->pt() < PFJetPt[0] && scale * DiJetpfjet_->pt() > PFJetPt[1]) {
        PFJetPt[1] = scale * DiJetpfjet_->pt();
        PFJetPx[1] = scale * DiJetpfjet_->px();
        PFJetPy[1] = scale * DiJetpfjet_->py();
        PFJetEta[1] = DiJetpfjet_->eta();
        PFJetPhi[1] = DiJetpfjet_->phi();
        PFJetRapidity[1] = DiJetpfjet_->rapidity();
        PFJetMass[1] = DiJetpfjet_->mass();
        PFJetNHEF[1] = DiJetpfjet_->neutralHadronEnergyFraction();
        PFJetCHEF[1] = DiJetpfjet_->chargedHadronEnergyFraction();
        PFJetNEMF[1] = DiJetpfjet_->neutralEmEnergyFraction();
        PFJetCEMF[1] = DiJetpfjet_->chargedEmEnergyFraction();
      } else {
      }
      if (scale * DiJetpfjet_->pt() > dijet_PFJet1_pt_cut_)
        dijet_countPFJet_++;
    }
    if (PFJetPt[0] > dijet_PFJet1_pt_cut_ && PFJetPt[1] > dijet_PFJet2_pt_cut_) {
      dijet_PFJet_pt[icoll]->Fill(PFJetPt[0]);
      dijet_PFJet_eta[icoll]->Fill(PFJetEta[0]);
      dijet_PFJet_phi[icoll]->Fill(PFJetPhi[0]);
      dijet_PFJet_rapidity[icoll]->Fill(PFJetRapidity[0]);
      dijet_PFJet_mass[icoll]->Fill(PFJetMass[0]);
      dijet_PFJet_pt[icoll]->Fill(PFJetPt[1]);
      dijet_PFJet_eta[icoll]->Fill(PFJetEta[1]);
      dijet_PFJet_phi[icoll]->Fill(PFJetPhi[1]);
      dijet_PFJet_rapidity[icoll]->Fill(PFJetRapidity[1]);
      dijet_PFJet_mass[icoll]->Fill(PFJetMass[1]);
      dijet_deltaPhiPFJet1PFJet2[icoll]->Fill(deltaPhi(PFJetPhi[0], PFJetPhi[1]));
      dijet_deltaEtaPFJet1PFJet2[icoll]->Fill(PFJetEta[0] - PFJetEta[1]);
      dijet_deltaRPFJet1PFJet2[icoll]->Fill(deltaR(PFJetEta[0], PFJetPhi[0], PFJetEta[1], PFJetPhi[1]));
      dijet_invMassPFJet1PFJet2[icoll]->Fill(
          sqrt(2 * PFJetPt[0] * PFJetPt[1] * (cosh(PFJetEta[0] - PFJetEta[1]) - cos(PFJetPhi[0] - PFJetPhi[1]))));
      dijet_PFchef[icoll]->Fill(PFJetCHEF[0]);
      dijet_PFnhef[icoll]->Fill(PFJetNHEF[0]);
      dijet_PFcemf[icoll]->Fill(PFJetCEMF[0]);
      dijet_PFnemf[icoll]->Fill(PFJetNEMF[0]);
      dijet_PFJetMulti[icoll]->Fill(dijet_countPFJet_);
    }
  }
}

void ExoticaDQM::analyzeDiMuons(const Event& iEvent) {
  if (MuonPt[0] > dimuon_Muon1_pt_cut_ && MuonPt[1] > dimuon_Muon2_pt_cut_ && MuonCharge[0] * MuonCharge[1] == -1) {
    dimuon_Muon_pt->Fill(MuonPt[0]);
    dimuon_Muon_eta->Fill(MuonEta[0]);
    dimuon_Muon_phi->Fill(MuonPhi[0]);
    dimuon_Muon_pt->Fill(MuonPt[1]);
    dimuon_Muon_eta->Fill(MuonEta[1]);
    dimuon_Muon_phi->Fill(MuonPhi[1]);
    dimuon_Charge->Fill(MuonCharge[0]);
    dimuon_Charge->Fill(MuonCharge[1]);
    dimuon_deltaPhiMuon1Muon2->Fill(deltaPhi(MuonPhi[0], MuonPhi[1]));
    dimuon_deltaEtaMuon1Muon2->Fill(MuonEta[0] - MuonEta[1]);
    dimuon_deltaRMuon1Muon2->Fill(deltaR(MuonEta[0], MuonPhi[0], MuonEta[1], MuonPhi[1]));
    dimuon_invMassMuon1Muon2->Fill(
        sqrt(2 * MuonPt[0] * MuonPt[1] * (cosh(MuonEta[0] - MuonEta[1]) - cos(MuonPhi[0] - MuonPhi[1]))));
    dimuon_MuonMulti->Fill(dimuon_countMuon_);
  }
}

void ExoticaDQM::analyzeDiElectrons(const Event& iEvent) {
  if (ElectronPt[0] > dielectron_Electron1_pt_cut_ && ElectronPt[1] > dielectron_Electron2_pt_cut_ &&
      ElectronCharge[0] * ElectronCharge[1] == -1.) {
    dielectron_Electron_pt->Fill(ElectronPt[0]);
    dielectron_Electron_eta->Fill(ElectronEta[0]);
    dielectron_Electron_phi->Fill(ElectronPhi[0]);
    dielectron_Electron_pt->Fill(ElectronPt[1]);
    dielectron_Electron_eta->Fill(ElectronEta[1]);
    dielectron_Electron_phi->Fill(ElectronPhi[1]);
    dielectron_Charge->Fill(ElectronCharge[0]);
    dielectron_Charge->Fill(ElectronCharge[1]);
    dielectron_deltaPhiElectron1Electron2->Fill(deltaPhi(ElectronPhi[0], ElectronPhi[1]));
    dielectron_deltaEtaElectron1Electron2->Fill(ElectronEta[0] - ElectronEta[1]);
    dielectron_deltaRElectron1Electron2->Fill(deltaR(ElectronEta[0], ElectronPhi[0], ElectronEta[1], ElectronPhi[1]));
    dielectron_invMassElectron1Electron2->Fill(
        sqrt(2 * ElectronPt[0] * ElectronPt[1] *
             (cosh(ElectronEta[0] - ElectronEta[1]) - cos(ElectronPhi[0] - ElectronPhi[1]))));
    dielectron_ElectronMulti->Fill(dielectron_countElectron_);
  }
}

void ExoticaDQM::analyzeDiPhotons(const Event& iEvent) {
  if (PhotonPt[0] > diphoton_Photon1_pt_cut_ && PhotonPt[1] > diphoton_Photon2_pt_cut_) {
    diphoton_Photon_energy->Fill(PhotonEnergy[0]);
    diphoton_Photon_pt->Fill(PhotonPt[0]);
    diphoton_Photon_et->Fill(PhotonEt[0]);
    diphoton_Photon_eta->Fill(PhotonEta[0]);
    diphoton_Photon_etasc->Fill(PhotonEtaSc[0]);
    diphoton_Photon_phi->Fill(PhotonPhi[0]);
    if (fabs(PhotonEtaSc[0]) < 1.442) {
      diphoton_Photon_hovere_eb->Fill(PhotonHoverE[0]);
      diphoton_Photon_sigmaietaieta_eb->Fill(PhotonSigmaIetaIeta[0]);
      diphoton_Photon_trksumptsolidconedr03_eb->Fill(PhotonTrkSumPtSolidConeDR03[0]);
      diphoton_Photon_e1x5e5x5_eb->Fill(PhotonE1x5E5x5[0]);
      diphoton_Photon_e2x5e5x5_eb->Fill(PhotonE2x5E5x5[0]);
    }
    if (fabs(PhotonEtaSc[0]) > 1.566 && fabs(PhotonEtaSc[0]) < 2.5) {
      diphoton_Photon_hovere_ee->Fill(PhotonHoverE[0]);
      diphoton_Photon_sigmaietaieta_ee->Fill(PhotonSigmaIetaIeta[0]);
      diphoton_Photon_trksumptsolidconedr03_ee->Fill(PhotonTrkSumPtSolidConeDR03[0]);
      diphoton_Photon_e1x5e5x5_ee->Fill(PhotonE1x5E5x5[0]);
      diphoton_Photon_e2x5e5x5_ee->Fill(PhotonE2x5E5x5[0]);
    }
    diphoton_Photon_energy->Fill(PhotonEnergy[1]);
    diphoton_Photon_pt->Fill(PhotonPt[1]);
    diphoton_Photon_et->Fill(PhotonEt[1]);
    diphoton_Photon_eta->Fill(PhotonEta[1]);
    diphoton_Photon_etasc->Fill(PhotonEtaSc[1]);
    diphoton_Photon_phi->Fill(PhotonPhi[1]);
    if (fabs(PhotonEtaSc[1]) < 1.4442) {
      diphoton_Photon_hovere_eb->Fill(PhotonHoverE[1]);
      diphoton_Photon_sigmaietaieta_eb->Fill(PhotonSigmaIetaIeta[1]);
      diphoton_Photon_trksumptsolidconedr03_eb->Fill(PhotonTrkSumPtSolidConeDR03[1]);
      diphoton_Photon_e1x5e5x5_eb->Fill(PhotonE1x5E5x5[1]);
      diphoton_Photon_e2x5e5x5_eb->Fill(PhotonE2x5E5x5[1]);
    }
    if (fabs(PhotonEtaSc[1]) > 1.566 && fabs(PhotonEtaSc[1]) < 2.5) {
      diphoton_Photon_hovere_ee->Fill(PhotonHoverE[1]);
      diphoton_Photon_sigmaietaieta_ee->Fill(PhotonSigmaIetaIeta[1]);
      diphoton_Photon_trksumptsolidconedr03_ee->Fill(PhotonTrkSumPtSolidConeDR03[1]);
      diphoton_Photon_e1x5e5x5_ee->Fill(PhotonE1x5E5x5[1]);
      diphoton_Photon_e2x5e5x5_ee->Fill(PhotonE2x5E5x5[1]);
    }
    diphoton_deltaPhiPhoton1Photon2->Fill(deltaPhi(PhotonPhi[0], PhotonPhi[1]));
    diphoton_deltaEtaPhoton1Photon2->Fill(PhotonEta[0] - PhotonEta[1]);
    diphoton_deltaRPhoton1Photon2->Fill(deltaR(PhotonEta[0], PhotonPhi[0], PhotonEta[1], PhotonPhi[1]));
    diphoton_invMassPhoton1Photon2->Fill(
        sqrt(2 * PhotonPt[0] * PhotonPt[1] * (cosh(PhotonEta[0] - PhotonEta[1]) - cos(PhotonPhi[0] - PhotonPhi[1]))));
    diphoton_PhotonMulti->Fill(diphoton_countPhoton_);
  }
}

void ExoticaDQM::analyzeMonoJets(const Event& iEvent) {
  const PFMETCollection* pfmetcol = pfMETCollection_.product();
  const PFMET pfmet = pfmetcol->front();
  if (PFJetPt[0] > monojet_PFJet_pt_cut_ && pfmet.et() > monojet_PFJet_met_cut_) {
    monojet_PFJet_pt->Fill(PFJetPt[0]);
    monojet_PFJet_eta->Fill(PFJetEta[0]);
    monojet_PFJet_phi->Fill(PFJetPhi[0]);
    monojet_PFMet->Fill(pfmet.et());
    monojet_PFMet_phi->Fill(pfmet.phi());
    monojet_PFJetPtOverPFMet->Fill(PFJetPt[0] / pfmet.et());
    monojet_deltaPhiPFJetPFMet->Fill(deltaPhi(PFJetPhi[0], pfmet.phi()));
    monojet_PFchef->Fill(PFJetCHEF[0]);
    monojet_PFnhef->Fill(PFJetNHEF[0]);
    monojet_PFcemf->Fill(PFJetCEMF[0]);
    monojet_PFnemf->Fill(PFJetNEMF[0]);
    monojet_PFJetMulti->Fill(monojet_countPFJet_);
  }
}

void ExoticaDQM::analyzeMonoMuons(const Event& iEvent) {
  const PFMETCollection* pfmetcol = pfMETCollection_.product();
  const PFMET pfmet = pfmetcol->front();
  if (MuonPt[0] > monomuon_Muon_pt_cut_ && pfmet.et() > monomuon_Muon_met_cut_) {
    monomuon_Muon_pt->Fill(MuonPt[0]);
    monomuon_Muon_eta->Fill(MuonEta[0]);
    monomuon_Muon_phi->Fill(MuonPhi[0]);
    monomuon_Charge->Fill(MuonCharge[0]);
    monomuon_PFMet->Fill(pfmet.et());
    monomuon_PFMet_phi->Fill(pfmet.phi());
    monomuon_MuonPtOverPFMet->Fill(MuonPt[0] / pfmet.et());
    monomuon_deltaPhiMuonPFMet->Fill(deltaPhi(MuonPhi[0], pfmet.phi()));
    monomuon_TransverseMass->Fill(sqrt(2 * MuonPt[0] * pfmet.et() * (1 - cos(deltaPhi(MuonPhi[0], pfmet.phi())))));
    monomuon_MuonMulti->Fill(monomuon_countMuon_);
  }
}

void ExoticaDQM::analyzeMonoElectrons(const Event& iEvent) {
  const PFMETCollection* pfmetcol = pfMETCollection_.product();
  const PFMET pfmet = pfmetcol->front();
  if (ElectronPt[0] > monoelectron_Electron_pt_cut_ && pfmet.et() > monoelectron_Electron_met_cut_) {
    monoelectron_Electron_pt->Fill(ElectronPt[0]);
    monoelectron_Electron_eta->Fill(ElectronEta[0]);
    monoelectron_Electron_phi->Fill(ElectronPhi[0]);
    monoelectron_Charge->Fill(ElectronCharge[0]);
    monoelectron_PFMet->Fill(pfmet.et());
    monoelectron_PFMet_phi->Fill(pfmet.phi());
    monoelectron_ElectronPtOverPFMet->Fill(ElectronPt[0] / pfmet.et());
    monoelectron_deltaPhiElectronPFMet->Fill(deltaPhi(ElectronPhi[0], pfmet.phi()));
    monoelectron_TransverseMass->Fill(
        sqrt(2 * ElectronPt[0] * pfmet.et() * (1 - cos(deltaPhi(ElectronPhi[0], pfmet.phi())))));
    monoelectron_ElectronMulti->Fill(monoelectron_countElectron_);
  }
}

void ExoticaDQM::analyzeMonoPhotons(const Event& iEvent) {
  const PFMETCollection* pfmetcol = pfMETCollection_.product();
  const PFMET pfmet = pfmetcol->front();
  if (PhotonPt[0] > monophoton_Photon_pt_cut_ && pfmet.et() > monophoton_Photon_met_cut_) {
    monophoton_Photon_energy->Fill(PhotonEnergy[0]);
    monophoton_Photon_pt->Fill(PhotonPt[0]);
    monophoton_Photon_et->Fill(PhotonEt[0]);
    monophoton_Photon_eta->Fill(PhotonEta[0]);
    monophoton_Photon_etasc->Fill(PhotonEtaSc[0]);
    monophoton_Photon_phi->Fill(PhotonPhi[0]);
    monophoton_Photon_hovere->Fill(PhotonHoverE[0]);
    monophoton_Photon_sigmaietaieta->Fill(PhotonSigmaIetaIeta[0]);
    monophoton_Photon_trksumptsolidconedr03->Fill(PhotonTrkSumPtSolidConeDR03[0]);
    monophoton_Photon_e1x5e5x5->Fill(PhotonE1x5E5x5[0]);
    monophoton_Photon_e2x5e5x5->Fill(PhotonE2x5E5x5[0]);
    monophoton_PFMet->Fill(pfmet.et());
    monophoton_PFMet_phi->Fill(pfmet.phi());
    monophoton_PhotonPtOverPFMet->Fill(PhotonPt[0] / pfmet.et());
    monophoton_deltaPhiPhotonPFMet->Fill(deltaPhi(PhotonPhi[0], pfmet.phi()));
    monophoton_PhotonMulti->Fill(monophoton_countPhoton_);
  }
}

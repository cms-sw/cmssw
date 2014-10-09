#include "DQM/Physics/src/ExoticaDQM.h"

#include <memory>

// DQM
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Run.h"
#include "DataFormats/Provenance/interface/EventID.h"

// Candidate handling
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidateFwd.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"

// Vertex utilities
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// Other
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/RefToBase.h"

// Math
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

// vertexing

// Transient tracks
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

//JetCorrection
#include "JetMETCorrections/Objects/interface/JetCorrector.h"

// ROOT
#include "TLorentzVector.h"

// STDLIB
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;
using namespace reco;
using namespace trigger;

typedef vector<string> vstring;

struct SortCandByDecreasingPt {
  bool operator()( const Candidate &c1, const Candidate &c2) const {
    return c1.pt() > c2.pt();
  }
};


//
// -- Constructor
//
ExoticaDQM::ExoticaDQM(const edm::ParameterSet& ps){

  edm::LogInfo("ExoticaDQM") <<  " Starting ExoticaDQM " << "\n" ;

  typedef std::vector<edm::InputTag> vtag;

  // Get parameters from configuration file
  // Trigger
  theTriggerResultsCollection = ps.getParameter<InputTag>("triggerResultsCollection");
  //
  theTriggerForMultiJetsList  = ps.getParameter<vstring>("triggerMultiJetsList");
  theTriggerForLongLivedList  = ps.getParameter<vstring>("triggerLongLivedList");

  //
  ElectronToken_      = consumes<reco::GsfElectronCollection>(
      ps.getParameter<InputTag>("electronCollection"));
  PFElectronTokenEI_  = consumes<reco::PFCandidateCollection>(
      ps.getParameter<InputTag>("pfelectronCollectionEI"));
  //
  MuonToken_          = consumes<reco::MuonCollection>(
      ps.getParameter<InputTag>("muonCollection"));
  PFMuonTokenEI_      = consumes<reco::PFCandidateCollection>(
      ps.getParameter<InputTag>("pfmuonCollectionEI"));
  //
  TauToken_           = consumes<reco::CaloTauCollection>(
      ps.getParameter<InputTag>("tauCollection"));
  //PFTauLabel_       = ps.getParameter<InputTag>("pftauCollection");
  //
  PhotonToken_        = consumes<reco::PhotonCollection>(
      ps.getParameter<InputTag>("photonCollection"));
  //PFPhotonLabel_    = ps.getParameter<InputTag>("pfphotonCollection");
  //
  CaloJetToken_       = consumes<reco::CaloJetCollection>(
      ps.getParameter<InputTag>("caloJetCollection"));
  PFJetToken_         = consumes<reco::PFJetCollection>(
      ps.getParameter<InputTag>("pfJetCollection"));
  PFJetTokenEI_       = consumes<reco::PFJetCollection>(
      ps.getParameter<InputTag>("pfJetCollectionEI"));

  //
  CaloMETToken_       = consumes<reco::CaloMETCollection>(
      ps.getParameter<InputTag>("caloMETCollection"));
  PFMETToken_         = consumes<reco::PFMETCollection>(
      ps.getParameter<InputTag>("pfMETCollection"));
  PFMETTokenEI_       = consumes<reco::PFMETCollection>(
      ps.getParameter<InputTag>("pfMETCollectionEI"));

  ecalBarrelRecHitToken_ = consumes<EBRecHitCollection>(
      ps.getUntrackedParameter<InputTag>("ecalBarrelRecHit", InputTag("reducedEcalRecHitsEB")));
  ecalEndcapRecHitToken_ = consumes<EERecHitCollection>(
      ps.getUntrackedParameter<InputTag>("ecalEndcapRecHit", InputTag("reducedEcalRecHitsEE")));

  //Cuts - MultiJets
  jetID                    = new reco::helper::JetIDHelper(ps.getParameter<ParameterSet>("JetIDParams"), consumesCollector());
  CaloJetCorService_       = ps.getParameter<std::string>("CaloJetCorService");
  PFJetCorService_         = ps.getParameter<std::string>("PFJetCorService");

  //Varibles and Cuts for each Module:
  //Dijet
  dijet_PFJet1_pt_cut_      = ps.getParameter<double>("dijet_PFJet1_pt_cut");
  dijet_PFJet2_pt_cut_      = ps.getParameter<double>("dijet_PFJet2_pt_cut");
  //DiMuon
  dimuon_Muon1_pt_cut_      = ps.getParameter<double>("dimuon_Muon1_pt_cut");
  dimuon_Muon2_pt_cut_      = ps.getParameter<double>("dimuon_Muon2_pt_cut");
  //DiElectron
  dielectron_Electron1_pt_cut_ = ps.getParameter<double>("dielectron_Electron2_pt_cut");
  dielectron_Electron2_pt_cut_ = ps.getParameter<double>("dielectron_Electron2_pt_cut");
  //MonoJet
  monojet_PFJet_pt_cut_     = ps.getParameter<double>("monojet_PFJet_pt_cut");
  monojet_PFJet_met_cut_    = ps.getParameter<double>("monojet_PFJet_met_cut");
  //MonoMuon
  monomuon_Muon_pt_cut_  = ps.getParameter<double>("monomuon_Muon_pt_cut");
  monomuon_Muon_met_cut_ = ps.getParameter<double>("monomuon_Muon_met_cut");
  //MonoElectron
  monoelectron_Electron_pt_cut_  = ps.getParameter<double>("monoelectron_Electron_pt_cut");
  monoelectron_Electron_met_cut_ = ps.getParameter<double>("monoelectron_Electron_met_cut");

  // just to initialize
  //isValidHltConfig_ = false;
}


//
// -- Destructor
//
ExoticaDQM::~ExoticaDQM(){
  edm::LogInfo("ExoticaDQM") <<  " Deleting ExoticaDQM " << "\n" ;
}


//
// -- Begin Job
//
void ExoticaDQM::beginJob(){
  nLumiSecs_ = 0;
  nEvents_   = 0;
  pi = 3.14159265;
}


//
// -- Begin Run
//
void ExoticaDQM::beginRun(Run const& run, edm::EventSetup const& eSetup) {
  edm::LogInfo ("ExoticaDQM") <<"[ExoticaDQM]: Begining of Run";

  bei_ = Service<DQMStore>().operator->();
  bei_->setCurrentFolder("Physics/Exotica");
  bookHistos(bei_);

  // passed as parameter to HLTConfigProvider::init(), not yet used
  bool isConfigChanged = false;

  // isValidHltConfig_ used to short-circuit analyze() in case of problems
  //  const std::string hltProcessName( "HLT" );
  const std::string hltProcessName = theTriggerResultsCollection.process();
  isValidHltConfig_ = hltConfigProvider_.init( run, eSetup, hltProcessName, isConfigChanged );

}

//
// -- Begin  Luminosity Block
//
void ExoticaDQM::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
                                            edm::EventSetup const& context) {
  //edm::LogInfo ("ExoticaDQM") <<"[ExoticaDQM]: Begin of LS transition";
}


//
//  -- Book histograms
//
void ExoticaDQM::bookHistos(DQMStore* bei){

  bei->cd();

  //--- DiJet
  bei->setCurrentFolder("Physics/Exotica/Dijets");
  dijet_PFJet1_pt            = bei->book1D("dijet_PFJet1_pt",  "Pt of Leading PFJet (GeV)", 50, 30.0 , 3000);
  dijet_PFJet1_eta           = bei->book1D("dijet_PFJet1_eta", "#eta(Leading PFJet)", 50, -2.5, 2.5);
  dijet_PFJet1_phi           = bei->book1D("dijet_PFJet1_phi", "#phi(Leading PFJet)", 50, -3.14,3.14);
  dijet_PFJet2_pt            = bei->book1D("dijet_PFJet2_pt",  "Pt of SubLeading PFJet (GeV)", 50, 30.0 , 3000);
  dijet_PFJet2_eta           = bei->book1D("dijet_PFJet2_eta", "#eta(SubLeading PFJet)", 50, -5.0, 5.0);
  dijet_PFJet2_phi           = bei->book1D("dijet_PFJet2_phi", "#phi(SubLeading PFJet)", 50, -3.14,3.14);
  dijet_deltaPhiPFJet1PFJet2 = bei->book1D("dijet_deltaPhiPFJet1PFJet2", "#Delta#phi(Leading PFJet, Sub PFJet)", 40, 0., 3.15);
  dijet_deltaEtaPFJet1PFJet2 = bei->book1D("dijet_deltaEtaPFJet1PFJet2", "#Delta#eta(Leading PFJet, Sub PFJet)", 40, -5., 5.);
  dijet_deltaRPFJet1PFJet2   = bei->book1D("dijet_deltaRPFJet1PFJet2",   "#DeltaR(Leading PFJet, Sub PFJet)", 50, 0., 6.);
  dijet_invMassPFJet1PFJet2  = bei->book1D("dijet_invMassPFJet1PFJet2", "Leading PFJet, SubLeading PFJet Invariant mass (GeV)", 50, 30. , 6000.);
  dijet_PFchef               = bei->book1D("dijet_PFchef", "Leading PFJet CHEF", 50, 0.0 , 1.0);
  dijet_PFnhef               = bei->book1D("dijet_PFnhef", "Leading PFJet NHEF", 50, 0.0 , 1.0);
  dijet_PFcemf               = bei->book1D("dijet_PFcemf", "Leading PFJet CEMF", 50, 0.0 , 1.0);
  dijet_PFnemf               = bei->book1D("dijet_PFnemf", "Leading PFJEt NEMF", 50, 0.0 , 1.0);
  dijet_PFJetMulti           = bei->book1D("dijet_PFJetMulti", "No. of PFJets", 10, 0., 10.);
  //--- DiMuon
  bei->setCurrentFolder("Physics/Exotica/DiMuons");
  dimuon_Muon1_pt            = bei->book1D("dimuon_Muon1_pt",  "Pt of Leading Muon (GeV)", 50, 30.0 , 2000);
  dimuon_Muon1_eta           = bei->book1D("dimuon_Muon1_eta", "#eta(Leading Muon)", 50, -2.5, 2.5);
  dimuon_Muon1_phi           = bei->book1D("dimuon_Muon1_phi", "#phi(Leading Muon)", 50, -3.14,3.14);
  dimuon_Muon2_pt            = bei->book1D("dimuon_Muon2_pt",  "Pt of SubLeading Muon (GeV)", 50, 30.0 , 2000);
  dimuon_Muon2_eta           = bei->book1D("dimuon_Muon2_eta", "#eta(SubLeading Muon)", 50, -5.0, 5.0);
  dimuon_Muon2_phi           = bei->book1D("dimuon_Muon2_phi", "#phi(SubLeading Muon)", 50, -3.14,3.14);
  dimuon_Charge              = bei->book1D("dimuon_Charge", "Charge of the Muon", 10, -5., 5.);
  dimuon_deltaEtaMuon1Muon2  = bei->book1D("dimuon_deltaEtaMuon1Muon2", "#Delta#eta(Leading Muon, Sub Muon)", 40, -5., 5.);
  dimuon_deltaPhiMuon1Muon2  = bei->book1D("dimuon_deltaPhiMuon1Muon2", "#Delta#phi(Leading Muon, Sub Muon)", 40, 0., 3.15);
  dimuon_deltaRMuon1Muon2    = bei->book1D("dimuon_deltaRMuon1Muon2",   "#DeltaR(Leading Muon, Sub Muon)", 50, 0., 6.);
  dimuon_invMassMuon1Muon2   = bei->book1D("dimuon_invMassMuon1Muon2", "Leading Muon, SubLeading Muon Invariant mass (GeV)", 50, 30. , 4000.);
  dimuon_MuonMulti           = bei->book1D("dimuon_MuonMulti", "No. of Muons", 10, 0., 10.);
  //--- DiElectrons
  bei->setCurrentFolder("Physics/Exotica/DiElectrons");
  dielectron_Electron1_pt            = bei->book1D("dielectron_Electron1_pt",  "Pt of Leading Electron (GeV)", 50, 30.0 , 2000);
  dielectron_Electron1_eta           = bei->book1D("dielectron_Electron1_eta", "#eta(Leading Electron)", 50, -2.5, 2.5);
  dielectron_Electron1_phi           = bei->book1D("dielectron_Electron1_phi", "#phi(Leading Electron)", 50, -3.14,3.14);
  dielectron_Electron2_pt            = bei->book1D("dielectron_Electron2_pt",  "Pt of SubLeading Electron (GeV)", 50, 30.0 , 2000);
  dielectron_Electron2_eta           = bei->book1D("dielectron_Electron2_eta", "#eta(SubLeading Electron)", 50, -5.0, 5.0);
  dielectron_Electron2_phi           = bei->book1D("dielectron_Electron2_phi", "#phi(SubLeading Electron)", 50, -3.14,3.14);
  dielectron_Charge                  = bei->book1D("dielectron_Charge", "Charge of the Electron", 10, -5., 5.);
  dielectron_deltaEtaElectron1Electron2  = bei->book1D("dielectron_deltaEtaElectron1Electron2", "#Delta#eta(Leading Electron, Sub Electron)", 40, -5., 5.);
  dielectron_deltaPhiElectron1Electron2  = bei->book1D("dielectron_deltaPhiElectron1Electron2", "#Delta#phi(Leading Electron, Sub Electron)", 40, 0., 3.15);
  dielectron_deltaRElectron1Electron2    = bei->book1D("dielectron_deltaRElectron1Electron2",   "#DeltaR(Leading Electron, Sub Electron)", 50, 0., 6.);
  dielectron_invMassElectron1Electron2   = bei->book1D("dielectron_invMassElectron1Electron2", "Leading Electron, SubLeading Electron Invariant mass (GeV)", 50, 30. , 4000.);
  dielectron_ElectronMulti           = bei->book1D("dielectron_ElectronMulti", "No. of Electrons", 10, 0., 10.);
  //--- MonoJet
  bei->setCurrentFolder("Physics/Exotica/MonoJet");
  monojet_PFJet_pt            = bei->book1D("monojet_PFJet_pt",  "Pt of MonoJet (GeV)", 50, 30.0 , 1000);
  monojet_PFJet_eta           = bei->book1D("monojet_PFJet_eta", "#eta(MonoJet)", 50, -2.5, 2.5);
  monojet_PFJet_phi           = bei->book1D("monojet_PFJet_phi", "#phi(MonoJet)", 50, -3.14,3.14);
  monojet_PFMet               = bei->book1D("monojet_PFMet",      "Pt of PFMET (GeV)", 40, 0.0 , 1000);
  monojet_PFMet_phi           = bei->book1D("monojet_PFMet_phi", "#phi(PFMET #phi)", 50, -3.14,3.14);
  monojet_PFJetPtOverPFMet    = bei->book1D("monojet_PFJetPtOverPFMet", "Pt of MonoJet/MET (GeV)", 40, 0.0 , 5.);
  monojet_deltaPhiPFJetPFMet  = bei->book1D("monojet_deltaPhiPFJetPFMet", "#Delta#phi(MonoJet, PFMet)", 40, 0., 3.15);
  monojet_PFchef              = bei->book1D("monojet_PFchef", "MonojetJet CHEF", 50, 0.0 , 1.0);
  monojet_PFnhef              = bei->book1D("monojet_PFnhef", "MonojetJet NHEF", 50, 0.0 , 1.0);
  monojet_PFcemf              = bei->book1D("monojet_PFcemf", "MonojetJet CEMF", 50, 0.0 , 1.0);
  monojet_PFnemf              = bei->book1D("monojet_PFnemf", "MonojetJet NEMF", 50, 0.0 , 1.0);
  monojet_PFJetMulti          = bei->book1D("monojet_PFJetMulti", "No. of PFJets", 10, 0., 10.);
  //--- MonoMuon
  bei->setCurrentFolder("Physics/Exotica/MonoMuon");
  monomuon_Muon_pt            = bei->book1D("monomuon_Muon_pt",  "Pt of Monomuon (GeV)", 50, 30.0 , 2000);
  monomuon_Muon_eta           = bei->book1D("monomuon_Muon_eta", "#eta(Monomuon)", 50, -2.5, 2.5);
  monomuon_Muon_phi           = bei->book1D("monomuon_Muon_phi", "#phi(Monomuon)", 50, -3.14,3.14);
  monomuon_Charge             = bei->book1D("monomuon_Charge", "Charge of the MonoMuon", 10, -5., 5.);
  monomuon_PFMet              = bei->book1D("monomuon_PFMet",    "Pt of PFMET (GeV)", 40, 0.0 , 2000);
  monomuon_PFMet_phi          = bei->book1D("monomuon_PFMet_phi", "PFMET #phi", 50, -3.14,3.14);
  monomuon_MuonPtOverPFMet    = bei->book1D("monomuon_MuonPtOverPFMet",  "Pt of Monomuon/PFMet", 40, 0.0 , 5.);
  monomuon_deltaPhiMuonPFMet  = bei->book1D("monomuon_deltaPhiMuonPFMet", "#Delta#phi(Monomuon, PFMet)", 40, 0., 3.15);
  monomuon_TransverseMass     = bei->book1D("monomuon_TransverseMass", "Transverse Mass M_{T} GeV", 40, 200., 3000.);
  monomuon_MuonMulti          = bei->book1D("monomuon_MuonMulti", "No. of Muons", 10, 0., 10.);
  //--- MonoElectron
  bei->setCurrentFolder("Physics/Exotica/MonoElectron");
  monoelectron_Electron_pt            = bei->book1D("monoelectron_Electron_pt",  "Pt of Monoelectron (GeV)", 50, 30.0 , 4000);
  monoelectron_Electron_eta           = bei->book1D("monoelectron_Electron_eta", "#eta(MonoElectron)", 50, -2.5, 2.5);
  monoelectron_Electron_phi           = bei->book1D("monoelectron_Electron_phi", "#phi(MonoElectron)", 50, -3.14,3.14);
  monoelectron_Charge                 = bei->book1D("monoelectron_Charge", "Charge of the MonoElectron", 10, -5., 5.);
  monoelectron_PFMet                  = bei->book1D("monoelectron_PFMet",  "Pt of PFMET (GeV)", 40, 0.0 , 4000);
  monoelectron_PFMet_phi              = bei->book1D("monoelectron_PFMet_phi",  "PFMET #phi", 50, -3.14,3.14);
  monoelectron_ElectronPtOverPFMet    = bei->book1D("monoelectron_ElectronPtOverPFMet",  "Pt of Monoelectron/PFMet", 40, 0.0 , 5.);
  monoelectron_deltaPhiElectronPFMet  = bei->book1D("monoelectron_deltaPhiElectronPFMet", "#Delta#phi(MonoElectron, PFMet)", 40, 0., 3.15);
  monoelectron_TransverseMass         = bei->book1D("monoelectron_TransverseMass", "Transverse Mass M_{T} GeV", 40, 200., 4000.);
  monoelectron_ElectronMulti          = bei->book1D("monoelectron_ElectronMulti", "No. of Electrons", 10, 0., 10.);

  //--- LongLived
  bei->setCurrentFolder("Physics/Exotica/LongLived");
  ll_gammajet_sMajMajPhot         = bei->book1D("ll_gammajet_sMajMajPhot", "sMajMajPhot", 50, 0.0 , 5.0);
  ll_gammajet_sMinMinPhot         = bei->book1D("ll_gammajet_sMinMinPhot", "sMinMinPhot", 50, 0.0 , 5.0);

  //
  //bei->setCurrentFolder("Physics/Exotica/LongLivedTrigger");

  //
  // bei->setCurrentFolder("Physics/Exotica/EIComparison");
  // ei_pfjet1_pt     = bei->book1D("ei_pfjet1_pt",     "Pt of PFJet-1    (EI) (GeV)", 40, 0.0 , 1000);
  // ei_pfmet_pt      = bei->book1D("ei_pfmet_pt",      "Pt of PFMET      (EI) (GeV)", 40, 0.0 , 1000);
  //ei_pfmuon_pt     = bei->book1D("ei_pfmuon_pt",     "Pt of PFMuon     (EI) (GeV)", 40, 0.0 , 1000);
  //ei_pfelectron_pt = bei->book1D("ei_pfelectron_pt", "Pt of PFElectron (EI) (GeV)", 40, 0.0 , 1000);

  bei->cd();
}


//
//  -- Analyze
//
void ExoticaDQM::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){

  bool debbugging = false;
  if (debbugging == true ) printf("New Event getting info \n ");
  // Calo objects
  // Electrons
  bool ValidCaloElectron = iEvent.getByToken(ElectronToken_, ElectronCollection_);
  if(!ValidCaloElectron) return;

  // Muons
  bool ValidCaloMuon = iEvent.getByToken(MuonToken_, MuonCollection_);
  if(!ValidCaloMuon) return;

  if (debbugging == true ) printf("New Event getting: CaloElectrona and CaloMuon OK! \n ");

  // Taus
  // bool ValidCaloTau = iEvent.getByToken(TauToken_, TauCollection_);
  // if(!ValidCaloTau) return;

  // Jets
  // bool ValidCaloJet = iEvent.getByToken(CaloJetToken_, caloJetCollection_);
  // if(!ValidCaloJet) return;
  // calojets = *caloJetCollection_;
  
  if (debbugging == true ) printf("New Event getting: CaloJet Failedf! \n ");

  bool ValidPFJet = iEvent.getByToken(PFJetToken_, pfJetCollection_);
  if(!ValidPFJet) return;
  pfjets = *pfJetCollection_;

  if (debbugging == true ) printf("New Event getting: PFJet OK! \n ");

  // MET
  //bool ValidCaloMET = iEvent.getByToken(CaloMETToken_, caloMETCollection_);
  //if(!ValidCaloMET) return;
  if (debbugging == true ) printf("New Event getting: CaloMET Failed! \n ");

  // PFMETs
  bool ValidPFMET = iEvent.getByToken(PFMETToken_, pfMETCollection_);
  if(!ValidPFMET) return;

  if (debbugging == true ) printf("New Event getting: PFMET OK! \n ");

  // Photons
  bool ValidCaloPhoton = iEvent.getByToken(PhotonToken_, PhotonCollection_);
  if(!ValidCaloPhoton) return;

  if (debbugging == true ) printf("New Event getting: CaloPhoton OK! \n ");


  //#######################################################
  // Jet Correction
  // Define on-the-fly correction Jet
  for(int i=0; i<2; i++){
    PFJetPx[i]     = 0.;
    PFJetPy[i]     = 0.;
    PFJetPt[i]     = 0.;
    PFJetEta[i]    = 0.;
    PFJetPhi[i]    = 0.;
    PFJetNHEF[i]   = 0.;
    PFJetCHEF[i]   = 0.;
    PFJetNEMF[i]   = 0.;
    PFJetCEMF[i]   = 0.;
    //Muons
    MuonPx[i]     = 0.;
    MuonPy[i]     = 0.;
    MuonPt[i]     = 0.;
    MuonEta[i]    = 0.;
    MuonPhi[i]    = 0.;
    MuonCharge[i] = 0.;
    //Electrons
    ElectronPx[i]     = 0.;
    ElectronPy[i]     = 0.;
    ElectronPt[i]     = 0.;
    ElectronEta[i]    = 0.;
    ElectronPhi[i]    = 0.;
    ElectronCharge[i] = 0.;
    

    // CaloJetPx[i]   = 0.;
    // CaloJetPy[i]   = 0.;
    // CaloJetPt[i]   = 0.;
    // CaloJetEta[i]  = 0.;
    // CaloJetPhi[i]  = 0.;
    // CaloJetEMF[i]  = 0.;
    // CaloJetfHPD[i] = 0.;
    // CaloJetn90[i]  = 0.;
  }

  dijet_countPFJet_=0;
  monojet_countPFJet_=0;
  const JetCorrector* pfcorrector = JetCorrector::getJetCorrector(PFJetCorService_,iSetup);
  PFJetCollection::const_iterator pfjet_ = pfjets.begin();
  for(; pfjet_ != pfjets.end(); ++pfjet_){
    double scale = pfcorrector->correction(*pfjet_,iEvent, iSetup);
    if(scale*pfjet_->pt()>PFJetPt[0]){
      PFJetPt[1]   = PFJetPt[0];
      PFJetPx[1]   = PFJetPx[0];
      PFJetPy[1]   = PFJetPy[0];
      PFJetEta[1]  = PFJetEta[0];
      PFJetPhi[1]  = PFJetPhi[0];
      PFJetNHEF[1] = PFJetNHEF[0];
      PFJetCHEF[1] = PFJetCHEF[0];
      PFJetNEMF[1] = PFJetNEMF[0];
      PFJetCEMF[1] = PFJetCEMF[0];
      //
      PFJetPt[0]   = scale*pfjet_->pt();
      PFJetPx[0]   = scale*pfjet_->px();
      PFJetPy[0]   = scale*pfjet_->py();
      PFJetEta[0]  = pfjet_->eta();
      PFJetPhi[0]  = pfjet_->phi();
      PFJetNHEF[0] = pfjet_->neutralHadronEnergyFraction();
      PFJetCHEF[0] = pfjet_->chargedHadronEnergyFraction();
      PFJetNEMF[0] = pfjet_->neutralEmEnergyFraction();
      PFJetCEMF[0] = pfjet_->chargedEmEnergyFraction();
    }
    else if(scale*pfjet_->pt()<PFJetPt[0] && scale*pfjet_->pt()>PFJetPt[1] ){
      PFJetPt[1]   = scale*pfjet_->pt();
      PFJetPx[1]   = scale*pfjet_->px();
      PFJetPy[1]   = scale*pfjet_->py();
      PFJetEta[1]  = pfjet_->eta();
      PFJetPhi[1]  = pfjet_->phi();
      PFJetNHEF[1] = pfjet_->neutralHadronEnergyFraction();
      PFJetCHEF[1] = pfjet_->chargedHadronEnergyFraction();
      PFJetNEMF[1] = pfjet_->neutralEmEnergyFraction();
      PFJetCEMF[1] = pfjet_->chargedEmEnergyFraction();
    }
    else{}
    //    printf("Jet pt %f eta %f, phi %f, %i \n", scale*pfjet_->pt(), pfjet_->eta(),  pfjet_->phi(),monojet_countPFJet_);
    if(scale*pfjet_->pt()>dijet_PFJet1_pt_cut_) dijet_countPFJet_++;
    if(scale*pfjet_->pt()>dijet_PFJet1_pt_cut_) monojet_countPFJet_++;
  }

  dimuon_countMuon_ = 0;
  monomuon_countMuon_ = 0;
  reco::MuonCollection::const_iterator  muon_  = MuonCollection_->begin(); 
  for(; muon_ != MuonCollection_->end(); muon_++){
    if(muon_->pt()>MuonPt[0]){ //  Ensures that the Hightes pt muon is the [0] component and the [1] the second leading muon.
      MuonPt[1]   = MuonPt[0];
      MuonPx[1]   = MuonPx[0];
      MuonPy[1]   = MuonPy[0];
      MuonEta[1]  = MuonEta[0];
      MuonPhi[1]  = MuonPhi[0];
      MuonCharge[1]  = MuonCharge[0];
      //
      MuonPt[0]   = muon_->pt();
      MuonPx[0]   = muon_->px();
      MuonPy[0]   = muon_->py();
      MuonEta[0]  = muon_->eta();
      MuonPhi[0]  = muon_->phi();
      MuonCharge[0]  = muon_->charge();
    }
    //printf("Muon pt %f lepton %f, phi %f , %i \n", muon_->pt(), muon_->eta(), muon_->phi(), monomuon_countMuon_);
    if (muon_->pt() > dimuon_Muon1_pt_cut_) dimuon_countMuon_++;
    if (muon_->pt() > dimuon_Muon1_pt_cut_) monomuon_countMuon_++;

  }

  dielectron_countElectron_ = 0;
  monoelectron_countElectron_ = 0;
  reco::GsfElectronCollection::const_iterator electron_ = ElectronCollection_->begin();
  for(; electron_ != ElectronCollection_->end(); electron_++){ 
    if(electron_->pt()>ElectronPt[0] ){ //  Ensures that the Hightes pt electron is the [0] component and the [1] the second leading electron.
      ElectronPt[1]   = ElectronPt[0];
      ElectronPx[1]   = ElectronPx[0];
      ElectronPy[1]   = ElectronPy[0];
      ElectronEta[1]  = ElectronEta[0];
      ElectronPhi[1]  = ElectronPhi[0];
      ElectronCharge[1]  = ElectronCharge[0];
      //
      ElectronPt[0]   = electron_->pt();
      ElectronPx[0]   = electron_->px();
      ElectronPy[0]   = electron_->py();
      ElectronEta[0]  = electron_->eta();
      ElectronPhi[0]  = electron_->phi();
      ElectronCharge[0]  = electron_->charge();
    }
    //printf("Electron pt %f eta %f phi %f , %i \n", electron_->pt(), electron_->eta(), electron_->phi(), monoelectron_countElectron_);
    if (electron_->pt() > dielectron_Electron1_pt_cut_) dielectron_countElectron_++;
    if (electron_->pt() > dielectron_Electron1_pt_cut_) monoelectron_countElectron_++;
  }
  //printf("============== \n");

  //---------- CaloJet Correction (on-the-fly) ---------- NOT USED------------------------------
  // const JetCorrector* calocorrector = JetCorrector::getJetCorrector(CaloJetCorService_,iSetup);
  // CaloJetCollection::const_iterator calojet_ = calojets.begin();
  // for(; calojet_ != calojets.end(); ++calojet_){
  //   double scale = calocorrector->correction(*calojet_,iEvent, iSetup);
  //   jetID->calculate(iEvent, *calojet_);
  //   //printf("jet 1 pt %f\n", scale*calojet_->pt());
  //   if(scale*calojet_->pt()>CaloJetPt[0]){
  //     CaloJetPt[1]   = CaloJetPt[0];
  //     CaloJetPx[1]   = CaloJetPx[0];
  //     CaloJetPy[1]   = CaloJetPy[0];
  //     CaloJetEta[1]  = CaloJetEta[0];
  //     CaloJetPhi[1]  = CaloJetPhi[0];
  //     CaloJetEMF[1]  = CaloJetEMF[0];
  //     CaloJetfHPD[1] = CaloJetfHPD[0];
  //     CaloJetn90[1]  = CaloJetn90[0];
  //     //
  //     CaloJetPt[0]   = scale*calojet_->pt();
  //     CaloJetPx[0]   = scale*calojet_->px();
  //     CaloJetPy[0]   = scale*calojet_->py();
  //     CaloJetEta[0]  = calojet_->eta();
  //     CaloJetPhi[0]  = calojet_->phi();
  //     CaloJetEMF[0]  = calojet_->emEnergyFraction();
  //     CaloJetfHPD[0] = jetID->fHPD();
  //     CaloJetn90[0]  = jetID->n90Hits();
  //   }
  //   else if(scale*calojet_->pt()<CaloJetPt[0] && scale*calojet_->pt()>CaloJetPt[1] ){
  //     CaloJetPt[1]   = scale*calojet_->pt();
  //     CaloJetPx[1]   = scale*calojet_->px();
  //     CaloJetPy[1]   = scale*calojet_->py();
  //     CaloJetEta[1]  = calojet_->eta();
  //     CaloJetPhi[1]  = calojet_->phi();
  //     CaloJetEMF[1]  = calojet_->emEnergyFraction();
  //     CaloJetfHPD[1] = jetID->fHPD();
  //     CaloJetn90[1]  = jetID->n90Hits();
  //   }
  //   else{}
  // }

  //

  //#######################################################
  // Analyze
  //

  //Resonances
  analyzeDiJets(iEvent);
  analyzeDiMuons(iEvent);
  analyzeDiElectrons(iEvent);

  //MonoSearches
  analyzeMonoJets(iEvent);
  analyzeMonoMuons(iEvent);
  analyzeMonoElectrons(iEvent);

  //analyzeMultiJetsTrigger(iEvent);
  //
  analyzeLongLived(iEvent);
  //analyzeLongLivedTrigger(iEvent);
  //  analyzeEventInterpretation(iEvent, iSetup);
}

void ExoticaDQM::analyzeDiJets(const Event & iEvent){
  if(PFJetPt[0]> dijet_PFJet1_pt_cut_ && PFJetPt[1]> dijet_PFJet2_pt_cut_){
    dijet_PFJet1_pt->Fill(PFJetPt[0]);
    dijet_PFJet1_eta->Fill(PFJetEta[0]);
    dijet_PFJet1_phi->Fill(PFJetPhi[0]);
    dijet_PFJet2_pt->Fill(PFJetPt[1]);
    dijet_PFJet2_eta->Fill(PFJetEta[1]);
    dijet_PFJet2_phi->Fill(PFJetPhi[1]);
    dijet_deltaPhiPFJet1PFJet2->Fill(deltaPhi(PFJetPhi[0],PFJetPhi[1]));
    dijet_deltaEtaPFJet1PFJet2->Fill(PFJetEta[0]-PFJetEta[1]);
    dijet_deltaRPFJet1PFJet2->Fill(deltaR(PFJetEta[0],PFJetPhi[0],PFJetEta[1],PFJetPhi[1]));
    dijet_invMassPFJet1PFJet2->Fill(sqrt(2*PFJetPt[0]*PFJetPt[1]*(cosh(PFJetEta[0]-PFJetEta[1])-cos(PFJetPhi[0]-PFJetPhi[1]))));
    dijet_PFchef->Fill(PFJetCHEF[0]);
    dijet_PFnhef->Fill(PFJetNHEF[0]);
    dijet_PFcemf->Fill(PFJetCEMF[0]);
    dijet_PFnemf->Fill(PFJetNEMF[0]);
    dijet_PFJetMulti->Fill(dijet_countPFJet_);
    //--- PFMET
    // const PFMETCollection *pfmetcol = pfMETCollection_.product();
    // const PFMET pfmet = pfmetcol->front();
    // mj_pfMet_et->Fill(pfmet.et());
    // mj_pfMet_phi->Fill(pfmet.phi());
  }
  //--- MET
  // const CaloMETCollection *calometcol = caloMETCollection_.product();
  // const CaloMET met = calometcol->front();
  // mj_caloMet_et->Fill(met.et());
  // mj_caloMet_phi->Fill(met.phi());
  // mj_caloMet_et->Fill(1.);
  // mj_caloMet_phi->Fill(1.);
    //--- MonoJet
  //bool checkLepton = false;
  //reco::MuonCollection::const_iterator  muon  = MuonCollection_->begin();
  //for(; muon != MuonCollection_->end(); muon++){
  //if(muon->pt()<mj_monojet_ptPFMuon_) continue;
  //checkLepton = true;
  //}
  //reco::GsfElectronCollection::const_iterator electron = ElectronCollection_->begin();
  //for(; electron != ElectronCollection_->end(); electron++){
  //if(electron->pt()<mj_monojet_ptPFElectron_) continue;
  //checkLepton = true;
  //}
  //if(checkLepton==false){
  //intf("jet 1 pt %f, jet 2 pt %f \n", PFJetPt[0], PFJetPt[1]);

  // if (PFJetPt[0] < 325. )return; 
  // if (PFJetPt[1] < 300. )return; 
}
void ExoticaDQM::analyzeDiMuons(const Event & iEvent){
  //  printf("Muon PT 1 %f, Muon PT2 %f, charge %f \n", MuonPt[0], MuonPt[1], MuonCharge[0]*MuonCharge[1]);
  if(MuonPt[0] > dimuon_Muon1_pt_cut_ && MuonPt[1]> dimuon_Muon2_pt_cut_ && MuonCharge[0]*MuonCharge[1] == -1){
    dimuon_Muon1_pt->Fill(MuonPt[0]);
    dimuon_Muon1_eta->Fill(MuonEta[0]);
    dimuon_Muon1_phi->Fill(MuonPhi[0]);
    dimuon_Muon2_pt->Fill(MuonPt[1]);
    dimuon_Muon2_eta->Fill(MuonEta[1]);
    dimuon_Muon2_phi->Fill(MuonPhi[1]);
    dimuon_Charge->Fill(MuonCharge[0]);
    dimuon_Charge->Fill(MuonCharge[1]);
    dimuon_deltaPhiMuon1Muon2->Fill(deltaPhi(MuonPhi[0],MuonPhi[1]));
    dimuon_deltaEtaMuon1Muon2->Fill(MuonEta[0]-MuonEta[1]);
    dimuon_deltaRMuon1Muon2->Fill(deltaR(MuonEta[0],MuonPhi[0],MuonEta[1],MuonPhi[1]));
    dimuon_invMassMuon1Muon2->Fill(sqrt(2*MuonPt[0]*MuonPt[1]*(cosh(MuonEta[0]-MuonEta[1])-cos(MuonPhi[0]-MuonPhi[1]))));
    dimuon_MuonMulti->Fill(dimuon_countMuon_);
  }
}
void ExoticaDQM::analyzeDiElectrons(const Event & iEvent){
  if(ElectronPt[0] > dielectron_Electron1_pt_cut_ && ElectronPt[1]> dielectron_Electron2_pt_cut_ && ElectronCharge[0]*ElectronCharge[1] == -1.){
    dielectron_Electron1_pt->Fill(ElectronPt[0]);
    dielectron_Electron1_eta->Fill(ElectronEta[0]);
    dielectron_Electron1_phi->Fill(ElectronPhi[0]);
    dielectron_Electron2_pt->Fill(ElectronPt[1]);
    dielectron_Electron2_eta->Fill(ElectronEta[1]);
    dielectron_Electron2_phi->Fill(ElectronPhi[1]);
    dielectron_Charge->Fill(ElectronCharge[0]);
    dielectron_Charge->Fill(ElectronCharge[1]);
    dielectron_deltaPhiElectron1Electron2->Fill(deltaPhi(ElectronPhi[0],ElectronPhi[1]));
    dielectron_deltaEtaElectron1Electron2->Fill(ElectronEta[0]-ElectronEta[1]);
    dielectron_deltaRElectron1Electron2->Fill(deltaR(ElectronEta[0],ElectronPhi[0],ElectronEta[1],ElectronPhi[1]));
    dielectron_invMassElectron1Electron2->Fill(sqrt(2*ElectronPt[0]*ElectronPt[1]*(cosh(ElectronEta[0]-ElectronEta[1])-cos(ElectronPhi[0]-ElectronPhi[1]))));
    dielectron_ElectronMulti->Fill(dielectron_countElectron_);
  }
}
void ExoticaDQM::analyzeMonoJets(const Event & iEvent){
  //--- PFMET
  const PFMETCollection *pfmetcol = pfMETCollection_.product();
  const PFMET pfmet = pfmetcol->front();
  if(PFJetPt[0]> monojet_PFJet_pt_cut_ && pfmet.et() > monojet_PFJet_met_cut_){
    monojet_PFJet_pt->Fill(PFJetPt[0]);
    monojet_PFJet_eta->Fill(PFJetEta[0]);
    monojet_PFJet_phi->Fill(PFJetPhi[0]);
    monojet_PFMet->Fill(pfmet.et());
    monojet_PFMet_phi->Fill(pfmet.phi());
    monojet_PFJetPtOverPFMet->Fill(PFJetPt[0]/pfmet.et());
    monojet_deltaPhiPFJetPFMet->Fill(deltaPhi(PFJetPhi[0],pfmet.phi()));
    monojet_PFchef->Fill(PFJetCHEF[0]);
    monojet_PFnhef->Fill(PFJetNHEF[0]);
    monojet_PFcemf->Fill(PFJetCEMF[0]);
    monojet_PFnemf->Fill(PFJetNEMF[0]);
    monojet_PFJetMulti->Fill(monojet_countPFJet_);
  }
}
void ExoticaDQM::analyzeMonoMuons(const Event & iEvent){
  //--- PFMET
  const PFMETCollection *pfmetcol = pfMETCollection_.product();
  const PFMET pfmet = pfmetcol->front();
  if(MuonPt[0]> monomuon_Muon_pt_cut_ && pfmet.et() > monomuon_Muon_met_cut_){
    monomuon_Muon_pt->Fill(MuonPt[0]);
    monomuon_Muon_eta->Fill(MuonEta[0]);
    monomuon_Muon_phi->Fill(MuonPhi[0]);
    monomuon_Charge->Fill(MuonCharge[0]);
    monomuon_PFMet->Fill(pfmet.et());
    monomuon_PFMet_phi->Fill(pfmet.phi());
    monomuon_MuonPtOverPFMet->Fill(MuonPt[0]/pfmet.et());
    monomuon_deltaPhiMuonPFMet->Fill(deltaPhi(MuonPhi[0],pfmet.phi()));
    double mt = sqrt(2*MuonPt[0]*pfmet.et()*(1-cos(deltaPhi(MuonPhi[0],pfmet.phi()))));
    //printf("Transverse mass %f muon pt %f, Met %f \n", mt, MuonPt[0], pfmet.et());
    monomuon_TransverseMass->Fill(mt);
    monomuon_MuonMulti->Fill(monomuon_countMuon_);
  }
}
void ExoticaDQM::analyzeMonoElectrons(const Event & iEvent){
  //--- PFMET
  const PFMETCollection *pfmetcol = pfMETCollection_.product();
  const PFMET pfmet = pfmetcol->front();
  if(ElectronPt[0]> monoelectron_Electron_pt_cut_ && pfmet.et() > monoelectron_Electron_met_cut_){
    monoelectron_Electron_pt->Fill(ElectronPt[0]);
    monoelectron_Electron_eta->Fill(ElectronEta[0]);
    monoelectron_Electron_phi->Fill(ElectronPhi[0]);
    monoelectron_Charge->Fill(ElectronCharge[0]);
    monoelectron_PFMet->Fill(pfmet.et());
    monoelectron_PFMet_phi->Fill(pfmet.phi());
    monoelectron_ElectronPtOverPFMet->Fill(ElectronPt[0]/pfmet.et());
    monoelectron_deltaPhiElectronPFMet->Fill(deltaPhi(ElectronPhi[0],pfmet.phi()));
    double mt = sqrt(2*ElectronPt[0]*pfmet.et()*(1-cos(deltaPhi(ElectronPhi[0],pfmet.phi()))));
    //    printf("Transverse mass %f electron \n", mt);
    monoelectron_TransverseMass->Fill(mt);
    monoelectron_ElectronMulti->Fill(monoelectron_countElectron_);
  }
}
void ExoticaDQM::analyzeMultiJetsTrigger(const Event & iEvent){
}

void ExoticaDQM::analyzeLongLived(const Event & iEvent){
  // SMajMajPho, SMinMinPho
  // get ECAL reco hits
  Handle<EBRecHitCollection> ecalhitseb;
  const EBRecHitCollection* rhitseb=0;
  iEvent.getByToken(ecalBarrelRecHitToken_, ecalhitseb);
  rhitseb = ecalhitseb.product(); // get a ptr to the product
  //
  Handle<EERecHitCollection> ecalhitsee;
  const EERecHitCollection* rhitsee=0;
  iEvent.getByToken(ecalEndcapRecHitToken_, ecalhitsee);
  rhitsee = ecalhitsee.product(); // get a ptr to the product
  //
  int nPhot = 0;
  reco::PhotonCollection::const_iterator photon = PhotonCollection_->begin();
  for(; photon != PhotonCollection_->end(); ++photon){
    if(photon->energy()<3.) continue;
    if(nPhot>=40) continue;

    const Ptr<CaloCluster> theSeed = photon->superCluster()->seed();
    const EcalRecHitCollection* rechits = ( photon->isEB()) ? rhitseb : rhitsee;
    CaloClusterPtr SCseed = photon->superCluster()->seed();

    std::pair<DetId, float> maxRH = EcalClusterTools::getMaximum( *theSeed, &(*rechits) );

    if(maxRH.second) {
      Cluster2ndMoments moments = EcalClusterTools::cluster2ndMoments(*SCseed, *rechits);
      //std::vector<float> etaphimoments = EcalClusterTools::localCovariances(*SCseed, &(*rechits), &(*topology));
      ll_gammajet_sMajMajPhot->Fill(moments.sMaj);
      ll_gammajet_sMinMinPhot->Fill(moments.sMin);
    }
    else{
      ll_gammajet_sMajMajPhot->Fill(-100.);
      ll_gammajet_sMinMinPhot->Fill(-100.);
    }
    ++nPhot;
  }

}
// OLD Stuff Alberto.
// void ExoticaDQM::analyzeMuons(const Event & iEvent){
//   // if (MuonPt[0] < 325. )return; 
//   // if (MuonPt[1] < 300. )return; 
//   if(MuonPt[0]>0.){
//     hpt_Muon1_pt->Fill(MuonPt[0]);
//     hpt_Muon1_eta->Fill(MuonEta[0]);
//     hpt_Muon1_phi->Fill(MuonPhi[0]);
//   }
//   if(MuonPt[1]>0.){
//     hpt_Muon2_pt->Fill(MuonPt[1]);
//     hpt_Muon2_eta->Fill(MuonEta[1]);
//     hpt_Muon2_phi->Fill(MuonPhi[1]);
//     hpt_Muon_deltaPhiMuon1Muon2->Fill(deltaPhi(MuonPhi[0],MuonPhi[1]));
//     hpt_Muon_deltaEtaMuon1Muon2->Fill(MuonEta[0]-MuonEta[1]);
//     hpt_Muon_deltaRMuon1Muon2->Fill(deltaR(MuonEta[0],MuonPhi[0],
//      					   MuonEta[1],MuonPhi[1]));
//     hpt_Muon_invMassMuon1Muon2->Fill(sqrt(2*MuonPt[0]*MuonPt[1]*(cosh(MuonEta[0]-MuonEta[1])-cos(MuonPhi[0]-MuonPhi[1]))));
//     //    printf("MuonPt[0] %f, MuonPt[1] %f, deltaEtaMuon1Muon2 %f, deltaPhiMuon1Muon2 %f, invMassMuon1Muon2 %f \n", 
//     //	   MuonPt[0], MuonPt[1], MuonEta[0]-MuonEta[1], deltaPhi(MuonPhi[0],MuonPhi[1]), sqrt(2*MuonPt[0]*MuonPt[1]*(cosh(MuonEta[0]-MuonEta[1])-cos(MuonPhi[0]-MuonPhi[1]))));    
//   }
//   hpt_Muon_multiplicity->Fill(mj_dilepton_countMuons);

// }


void ExoticaDQM::analyzeLongLivedTrigger(const Event & iEvent){
}

void ExoticaDQM::analyzeEventInterpretation(const Event & iEvent, const edm::EventSetup& iSetup){

  // EI
  // PFElectrons
  bool ValidPFElectronEI = iEvent.getByToken(PFElectronTokenEI_, pfElectronCollectionEI_);
  if(!ValidPFElectronEI) return;
  pfelectronsEI = *pfElectronCollectionEI_;

  // PFMuons
  bool ValidPFMuonEI = iEvent.getByToken(PFMuonTokenEI_, pfMuonCollectionEI_);
  if(!ValidPFMuonEI) return;
  pfmuonsEI = *pfMuonCollectionEI_;

  // PFJets
  bool ValidPFJetEI = iEvent.getByToken(PFJetTokenEI_, pfJetCollectionEI_);
  if(!ValidPFJetEI) return;
  pfjetsEI = *pfJetCollectionEI_;

  // PFMETs
  bool ValidPFMETEI = iEvent.getByToken(PFMETTokenEI_, pfMETCollectionEI_);
  if(!ValidPFMETEI) return;

  // Jet Correction
  int countJet = 0;
  PFJetEIPt    = -99.;
  const JetCorrector* pfcorrectorEI = JetCorrector::getJetCorrector(PFJetCorService_,iSetup);
  PFJetCollection::const_iterator pfjet_ = pfjetsEI.begin();
  for(; pfjet_ != pfjetsEI.end(); ++pfjet_){
    double scale = pfcorrectorEI->correction(*pfjet_,iEvent, iSetup);
    if(scale*pfjet_->pt()<PFJetEIPt) continue;
    PFJetEIPt   = scale*pfjet_->pt();
    PFJetEIPx   = scale*pfjet_->px();
    PFJetEIPy   = scale*pfjet_->py();
    PFJetEIEta  = pfjet_->eta();
    PFJetEIPhi  = pfjet_->phi();
    PFJetEINHEF = pfjet_->neutralHadronEnergyFraction();
    PFJetEICHEF = pfjet_->chargedHadronEnergyFraction();
    PFJetEINEMF = pfjet_->neutralEmEnergyFraction();
    PFJetEICEMF = pfjet_->chargedEmEnergyFraction();
    countJet++;
  }
  if(countJet>0){
    ei_pfjet1_pt->Fill(PFJetEIPt);
  }

  const PFMETCollection *pfmetcolEI = pfMETCollectionEI_.product();
  const PFMET pfmetEI = pfmetcolEI->front();
  ei_pfmet_pt->Fill(pfmetEI.et());


}

//
// -- End Luminosity Block
//
void ExoticaDQM::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) {
  //edm::LogInfo ("ExoticaDQM") <<"[ExoticaDQM]: End of LS transition, performing the DQM client operation";
  nLumiSecs_++;
  //edm::LogInfo("ExoticaDQM") << "============================================ "
  //<< endl << " ===> Iteration # " << nLumiSecs_ << " " << lumiSeg.luminosityBlock()
  //<< endl  << "============================================ " << endl;
}


//
// -- End Run
//
void ExoticaDQM::endRun(edm::Run const& run, edm::EventSetup const& eSetup){
}


//
// -- End Job
//
void ExoticaDQM::endJob(){
  //edm::LogInfo("ExoticaDQM") <<"[ExoticaDQM]: endjob called!";
}

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:

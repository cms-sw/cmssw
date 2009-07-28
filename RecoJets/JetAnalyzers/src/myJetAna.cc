// myJetAna.cc
// Description:  Access Cruzet Data
// Author: Frank Chlebana
// Date:  24 - July - 2008
// 
#include "RecoJets/JetAnalyzers/interface/myJetAna.h"
#include "RecoJets/JetAlgorithms/interface/JetAlgoHelper.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"


#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// #include "DataFormats/PhotonReco/interface/PhotonFwd.h"
// #include "DataFormats/PhotonReco/interface/Photon.h"


#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
// #include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
// #include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
// #include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include <TROOT.h>
#include <TSystem.h>
#include <TFile.h>
#include <TCanvas.h>
#include <cmath>

using namespace edm;
using namespace reco;
using namespace std;

#define DEBUG 1
#define MAXJETS 100

typedef struct RBX_struct {
  double et;
  double hadEnergy;
  double emEnergy;
  float  hcalTime;
  float  ecalTime;
  int    nTowers;
} RBX ;

typedef struct HPD_struct {
  double et;
  double hadEnergy;
  double emEnergy;
  double time;
  float  hcalTime;
  float  ecalTime;
  int    nTowers;
} HPD ;


RBX RBXColl[36];
HPD HPDColl[144];

// ************************
// ************************

// Get the algorithm of the jet collections we will read from the .cfg file 
// which defines the value of the strings CaloJetAlgorithm and GenJetAlgorithm.

myJetAna::myJetAna( const ParameterSet & cfg ) :
  CaloJetAlgorithm( cfg.getParameter<string>( "CaloJetAlgorithm" ) ), 
  GenJetAlgorithm( cfg.getParameter<string>( "GenJetAlgorithm" ) )  
{
  theTriggerResultsLabel = cfg.getParameter<edm::InputTag>("TriggerResultsLabel");
}


// ************************
// ************************

void myJetAna::beginJob( const EventSetup & ) {

  edm::Service<TFileService> fs;


  // --- passed selection cuts 
  h_pt     = fs->make<TH1F>( "pt",  "Jet p_{T}", 100, 0, 1000 );
  h_ptRBX  = fs->make<TH1F>( "ptRBX",  "RBX: Jet p_{T}", 100, 0, 1000 );
  h_ptHPD  = fs->make<TH1F>( "ptHPD",  "HPD: Jet p_{T}", 100, 0, 1000 );
  h_ptTower  = fs->make<TH1F>( "ptTower",  "Jet p_{T}", 100, 0, 1000 );
  h_et     = fs->make<TH1F>( "et",  "Jet E_{T}", 100, 0, 1000 );
  h_eta    = fs->make<TH1F>( "eta", "Jet #eta", 100, -4, 4 );
  h_phi    = fs->make<TH1F>( "phi", "Jet #phi", 50, -M_PI, M_PI );
  // ---

  dijetMass  =  fs->make<TH1F>("dijetMass","DiJet Mass",100,0,4000);

  totEneLeadJetEta1 = fs->make<TH1F>("totEneLeadJetEta1","Total Energy Lead Jet Eta1 1",100,0,1500);
  totEneLeadJetEta2 = fs->make<TH1F>("totEneLeadJetEta2","Total Energy Lead Jet Eta2 1",100,0,1500);
  totEneLeadJetEta3 = fs->make<TH1F>("totEneLeadJetEta3","Total Energy Lead Jet Eta3 1",100,0,1500);
  hadEneLeadJetEta1 = fs->make<TH1F>("hadEneLeadJetEta1","Hadronic Energy Lead Jet Eta1 1",100,0,1500);
  hadEneLeadJetEta2 = fs->make<TH1F>("hadEneLeadJetEta2","Hadronic Energy Lead Jet Eta2 1",100,0,1500);
  hadEneLeadJetEta3 = fs->make<TH1F>("hadEneLeadJetEta3","Hadronic Energy Lead Jet Eta3 1",100,0,1500);
  emEneLeadJetEta1  = fs->make<TH1F>("emEneLeadJetEta1","EM Energy Lead Jet Eta1 1",100,0,1500);
  emEneLeadJetEta2  = fs->make<TH1F>("emEneLeadJetEta2","EM Energy Lead Jet Eta2 1",100,0,1500);
  emEneLeadJetEta3  = fs->make<TH1F>("emEneLeadJetEta3","EM Energy Lead Jet Eta3 1",100,0,1500);


  hadFracEta1 = fs->make<TH1F>("hadFracEta11","Hadronic Fraction Eta1 Jet 1",100,0,1);
  hadFracEta2 = fs->make<TH1F>("hadFracEta21","Hadronic Fraction Eta2 Jet 1",100,0,1);
  hadFracEta3 = fs->make<TH1F>("hadFracEta31","Hadronic Fraction Eta3 Jet 1",100,0,1);

  SumEt  = fs->make<TH1F>("SumEt","SumEt",100,0,1000);
  MET    = fs->make<TH1F>("MET",  "MET",100,0,1000);
  MET_RBX    = fs->make<TH1F>("MET_RBX",  "MET",100,0,1000);
  MET_HPD    = fs->make<TH1F>("MET_HPD",  "MET",100,0,1000);
  MET_Tower  = fs->make<TH1F>("MET_Tower",  "MET",100,0,1000);
  METSig = fs->make<TH1F>("METSig",  "METSig",100,0,200);
  MEx    = fs->make<TH1F>("MEx",  "MEx",100,-100,100);
  MEy    = fs->make<TH1F>("MEy",  "MEy",100,-100,100);
  METPhi = fs->make<TH1F>("METPhi",  "METPhi",100,0,4);

  h_Vx     = fs->make<TH1F>("Vx",  "Vx",100,-0.5,0.5);
  h_Vy     = fs->make<TH1F>("Vy",  "Vy",100,-0.5,0.5);
  h_Vz     = fs->make<TH1F>("Vz",  "Vz",100,-20,20);
  h_VNTrks = fs->make<TH1F>("VNTrks",  "VNTrks",10,1,100);

  h_Trk_pt   = fs->make<TH1F>("Trk_pt",  "Trk_pt",100,1,100);
  h_Trk_NTrk = fs->make<TH1F>("Trk_NTrkt",  "Trk_NTrk",100,1,100);

  hf_sumTowerAllEx = fs->make<TH1F>("sumTowerAllEx","Tower Ex",100,-1000,1000);
  hf_sumTowerAllEy = fs->make<TH1F>("sumTowerAllEy","Tower Ey",100,-1000,1000);

  hf_TowerJetEt   = fs->make<TH1F>("TowerJetEt","Tower/Jet Et 1",50,0,1);

  ETime = fs->make<TH1F>("ETime","Ecal Time",200,-200,200);
  HTime = fs->make<TH1F>("HTime","Hcal Time",200,-200,200);

  h_towerHadEn  = fs->make<TH1F>("h_towerHadEn" ,"Hadronic Energy in Calo Tower",12000,-20,100);
  h_towerEmEn  	= fs->make<TH1F>("h_towerEmEn"  ,"EM Energy in Calo Tower",12000,-20,100);
  h_towerEmFrac	= fs->make<TH1F>("h_towerEmFrac","EM Fraction of Energy in Calo Tower",1000,0.,1.0);

  RBX_et        = fs->make<TH1F>("RBX_et","ET in RBX",1000,-20,100);
  RBX_hadEnergy = fs->make<TH1F>("RBX_hadEnergy","Hcal Energy in RBX",1000,-20,100);
  RBX_hcalTime  = fs->make<TH1F>("RBX_hcalTime","Hcal Time in RBX",200,-200,200);
  RBX_nTowers   = fs->make<TH1F>("RBX_nTowers","Number of Towers in RBX",75,0,75);
  RBX_N         = fs->make<TH1F>("RBX_N","Number of RBX",10,0,10);

  HPD_et        = fs->make<TH1F>("HPD_et","ET in HPD",1000,-20,100);
  HPD_hadEnergy = fs->make<TH1F>("HPD_hadEnergy","Hcal Energy in HPD",1000,-20,100);
  HPD_hcalTime  = fs->make<TH1F>("HPD_hcalTime","Hcal Time in HPD",200,-200,200);
  HPD_nTowers   = fs->make<TH1F>("HPD_nTowers","Number of Towers in HPD",20,0,20);
  HPD_N         = fs->make<TH1F>("HPD_N","Number of HPD",10,0,10);
  
  nTowers1  = fs->make<TH1F>("nTowers1","Number of Towers pt 0.5",100,0,200);
  nTowers2  = fs->make<TH1F>("nTowers2","Number of Towers pt 1.0",100,0,200);
  nTowers3  = fs->make<TH1F>("nTowers3","Number of Towers pt 1.5",100,0,200);
  nTowers4  = fs->make<TH1F>("nTowers4","Number of Towers pt 2.0",100,0,200);

  nTowersLeadJetPt1  = fs->make<TH1F>("nTowersLeadJetPt1","Number of Towers in Lead Jet pt 0.5",100,0,100);
  nTowersLeadJetPt2  = fs->make<TH1F>("nTowersLeadJetPt2","Number of Towers in Lead Jet pt 1.0",100,0,100);
  nTowersLeadJetPt3  = fs->make<TH1F>("nTowersLeadJetPt3","Number of Towers in Lead Jet pt 1.5",100,0,100);
  nTowersLeadJetPt4  = fs->make<TH1F>("nTowersLeadJetPt4","Number of Towers in Lead Jet pt 2.0",100,0,100);

  h_nCalJets  =  fs->make<TH1F>( "nCalJets",  "Number of CalJets", 20, 0, 20 );


  HBEne     = fs->make<TH1F>( "HBEne",  "HBEne", 12000, -20, 100 );
  HBTime    = fs->make<TH1F>( "HBTime", "HBTime", 200, -200, 200 );
  HEEne     = fs->make<TH1F>( "HEEne",  "HEEne", 12000, -20, 100 );
  HEposEne  = fs->make<TH1F>( "HEposEne",  "HEposEne", 12000, -20, 100 );
  HEnegEne  = fs->make<TH1F>( "HEnegEne",  "HEnegEne", 12000, -20, 100 );
  HETime    = fs->make<TH1F>( "HETime", "HETime", 200, -200, 200 );
  HEposTime = fs->make<TH1F>( "HEposTime",  "HEposTime", 12000, -20, 100 );
  HEnegTime = fs->make<TH1F>( "HEnegTime",  "HEnegTime", 12000, -20, 100 );
  HOEne     = fs->make<TH1F>( "HOEne",  "HOEne", 12000, -20, 100 );
  HOTime    = fs->make<TH1F>( "HOTime", "HOTime", 200, -200, 200 );

  // Histos for separating SiPMs and HPDs in HO:
  HOSEne     = fs->make<TH1F>( "HOSEne",  "HOSEne", 12000, -20, 100 );
  HOSTime    = fs->make<TH1F>( "HOSTime", "HOSTime", 200, -200, 200 );
  HOHEne     = fs->make<TH1F>( "HOHEne",  "HOHEne", 12000, -20, 100 );
  HOHTime    = fs->make<TH1F>( "HOHTime", "HOHTime", 200, -200, 200 );

  HOHr0Ene      = fs->make<TH1F>( "HOHr0Ene"  ,   "HOHr0Ene", 12000, -20 , 100 );
  HOHr0Time     = fs->make<TH1F>( "HOHr0Time" ,  "HOHr0Time",   200, -200, 200 );
  HOHrm1Ene     = fs->make<TH1F>( "HOHrm1Ene" ,  "HOHrm1Ene", 12000, -20 , 100 );
  HOHrm1Time    = fs->make<TH1F>( "HOHrm1Time", "HOHrm1Time",   200, -200, 200 );
  HOHrm2Ene     = fs->make<TH1F>( "HOHrm2Ene" ,  "HOHrm2Ene", 12000, -20 , 100 );
  HOHrm2Time    = fs->make<TH1F>( "HOHrm2Time", "HOHrm2Time",   200, -200, 200 );
  HOHrp1Ene     = fs->make<TH1F>( "HOHrp1Ene" ,  "HOHrp1Ene", 12000, -20 , 100 );
  HOHrp1Time    = fs->make<TH1F>( "HOHrp1Time", "HOHrp1Time",   200, -200, 200 );
  HOHrp2Ene     = fs->make<TH1F>( "HOHrp2Ene" ,  "HOHrp2Ene", 12000, -20 , 100 );
  HOHrp2Time    = fs->make<TH1F>( "HOHrp2Time", "HOHrp2Time",   200, -200, 200 );

  HOocc    = fs->make<TH2F>( "HOocc", "HOocc",81,-40.5,40.5,70,0.5,70.5);
  HBocc    = fs->make<TH2F>( "HBocc", "HBocc",81,-40.5,40.5,70,0.5,70.5);
  HEocc    = fs->make<TH2F>( "HEocc", "HEocc",81,-40.5,40.5,70,0.5,70.5);
  HFocc    = fs->make<TH2F>( "HFocc", "HFocc",81,-40.5,40.5,70,0.5,70.5);

  HFEne     = fs->make<TH1F>( "HFEne",  "HFEne", 12000, -20, 100 );
  HFTime    = fs->make<TH1F>( "HFTime", "HFTime", 200, -200, 200 );

  // Histos for separating HF long/short fibers:
  HFLEne     = fs->make<TH1F>( "HFLEne",  "HFSEne", 12000, -20, 100 );
  HFLTime    = fs->make<TH1F>( "HFLTime", "HFSTime", 200, -200, 200 );
  HFSEne     = fs->make<TH1F>( "HFSEne",  "HFSEne", 12000, -20, 100 );
  HFSTime    = fs->make<TH1F>( "HFSTime", "HFSTime", 200, -200, 200 );

  EBEne     = fs->make<TH1F>( "EBEne",  "EBEne", 12000, -20, 100 );
  EBTime    = fs->make<TH1F>( "EBTime", "EBTime", 200, -200, 200 );
  EEEne     = fs->make<TH1F>( "EEEne",  "EEEne", 12000, -20, 100 );
  EEnegEne  = fs->make<TH1F>( "EEnegEne",  "EEnegEne", 12000, -20, 100 );
  EEposEne  = fs->make<TH1F>( "EEposEne",  "EEposEne", 12000, -20, 100 );
  EETime    = fs->make<TH1F>( "EETime", "EETime", 200, -200, 200 );
  EEnegTime = fs->make<TH1F>( "EEnegTime", "EEnegTime", 200, -200, 200 );
  EEposTime = fs->make<TH1F>( "EEposTime", "EEposTime", 200, -200, 200 );

  h_ptCal     = fs->make<TH1F>( "ptCal",  "p_{T} of CalJet", 50, 0, 1000 );
  h_etaCal    = fs->make<TH1F>( "etaCal", "#eta of  CalJet", 100, -4, 4 );
  h_phiCal    = fs->make<TH1F>( "phiCal", "#phi of  CalJet", 50, -M_PI, M_PI );

  h_nGenJets  =  fs->make<TH1F>( "nGenJets",  "Number of GenJets", 20, 0, 20 );

  h_ptGen     =  fs->make<TH1F>( "ptGen",  "p_{T} of GenJet", 50, 0, 1000 );
  h_etaGen    =  fs->make<TH1F>( "etaGen", "#eta of GenJet", 100, -4, 4 );
  h_phiGen    =  fs->make<TH1F>( "phiGen", "#phi of GenJet", 50, -M_PI, M_PI );

  h_ptGenL    =  fs->make<TH1F>( "ptGenL",  "p_{T} of GenJetL", 50, 0, 300 );
  h_etaGenL   =  fs->make<TH1F>( "etaGenL", "#eta of GenJetL", 100, -4, 4 );
  h_phiGenL   =  fs->make<TH1F>( "phiGenL", "#phi of GenJetL", 50, -M_PI, M_PI );

  h_jetEt      = fs->make<TH1F>( "jetEt", "Total Jet Et", 100, 0, 3000 );

  h_jet1Pt       = fs->make<TH1F>( "jet1Pt", "Jet1 Pt", 100, 0, 1000 );
  h_jet2Pt       = fs->make<TH1F>( "jet2Pt", "Jet2 Pt", 100, 0, 1000 );
  h_jet1PtHLT    = fs->make<TH1F>( "jet1PtHLT", "Jet1 Pt HLT", 100, 0, 1000 );

  h_TotalUnclusteredEt = fs->make<TH1F>( "TotalUnclusteredEt", "Total Unclustered Et", 100, 0, 500 );
  h_UnclusteredEt      = fs->make<TH1F>( "UnclusteredEt", "Unclustered Et", 100, 0, 50 );
  h_UnclusteredEts     = fs->make<TH1F>( "UnclusteredEts", "Unclustered Et", 100, 0, 2 );

  h_ClusteredE         = fs->make<TH1F>( "ClusteredE", "Clustered E", 200, 0, 20 );
  h_TotalClusteredE    = fs->make<TH1F>( "TotalClusteredE", "Total Clustered E", 200, 0, 100 );
  h_UnclusteredE       = fs->make<TH1F>( "UnclusteredE", "Unclustered E", 200, 0, 20 );
  h_TotalUnclusteredE  = fs->make<TH1F>( "TotalUnclusteredE", "Total Unclustered E", 200, 0, 100 );

  EMFraction           = fs->make<TH1F>( "EMFraction", "Jet EM Fraction", 100, 0, 1 );
  NTowers              = fs->make<TH1F>( "NTowers", "Number of Towers", 100, 0, 100 );


  h_EmEnergy   = fs->make<TH2F>( "EmEnergy",  "Em Energy",  90, -45, 45, 73, 0, 73 );
  h_HadEnergy  = fs->make<TH2F>( "HadEnergy", "Had Energy", 90, -45, 45, 73, 0, 73 );

  st_Pt            = fs->make<TH1F>( "st_Pt", "Pt", 200, 0, 200 );
  st_Constituents  = fs->make<TH1F>( "st_Constituents", "Constituents", 200, 0, 200 );
  st_Energy        = fs->make<TH1F>( "st_Energy", "Tower Energy", 200, 0, 200 );
  st_EmEnergy      = fs->make<TH1F>( "st_EmEnergy", "Tower EmEnergy", 200, 0, 200 );
  st_HadEnergy     = fs->make<TH1F>( "st_HadEnergy", "Tower HadEnergy", 200, 0, 200 );
  st_OuterEnergy   = fs->make<TH1F>( "st_OuterEnergy", "Tower OuterEnergy", 200, 0, 200 );
  st_Eta           = fs->make<TH1F>( "st_Eta", "Eta", 100, -4, 4 );
  st_Phi           = fs->make<TH1F>( "st_Phi", "Phi", 50, -M_PI, M_PI );
  st_iEta          = fs->make<TH1F>( "st_iEta", "iEta", 60, -30, 30 );
  st_iPhi          = fs->make<TH1F>( "st_iPhi", "iPhi", 80, 0, 80 );
  st_Frac          = fs->make<TH1F>( "st_Frac", "Frac", 100, 0, 1 );

}

// ************************
// ************************
void myJetAna::analyze( const edm::Event& evt, const edm::EventSetup& es ) {
 
  using namespace edm;

  int EtaOk10, EtaOk13, EtaOk40;

  double LeadMass;

  float pt1;

  float minJetPt = 30.;
  float minJetPt10 = 10.;
  int jetInd, allJetInd;
  LeadMass = -1;

  math::XYZTLorentzVector p4tmp[2], p4cortmp[2];

  // --------------------------------------------------------------
  // --------------------------------------------------------------
  
  // *********************
  // *** Classify Event
  // *********************
  int evtType = 0;

  Handle<CaloTowerCollection> caloTowers;
  evt.getByLabel( "towerMaker", caloTowers );

  for (int i=0;i<36;i++) {
    RBXColl[i].et        = 0;
    RBXColl[i].hadEnergy = 0;
    RBXColl[i].emEnergy  = 0;
    RBXColl[i].hcalTime  = 0;
    RBXColl[i].ecalTime  = 0;
    RBXColl[i].nTowers   = 0;
  }  
  for (int i=0;i<144;i++) {
    HPDColl[i].et        = 0;
    HPDColl[i].hadEnergy = 0;
    HPDColl[i].emEnergy  = 0;
    HPDColl[i].hcalTime  = 0;
    HPDColl[i].ecalTime  = 0;
    HPDColl[i].nTowers   = 0;
  }  

  for (CaloTowerCollection::const_iterator tower = caloTowers->begin();
       tower != caloTowers->end(); tower++) {

    // Raw tower energy without grouping or thresholds
    h_towerHadEn->Fill(tower->hadEnergy());
    h_towerEmEn->Fill(tower->emEnergy());
    h_towerEmFrac->Fill(tower->emEnergy()/(tower->emEnergy()+tower->hadEnergy()));

    if ((tower->hadEnergy() + tower->emEnergy()) > 2.0) {

      int iRBX = tower->iphi();
      iRBX = iRBX-2;
      if (iRBX == 0)  iRBX = 17;
      if (iRBX == -1) iRBX = 18;
      iRBX = (iRBX-1)/4;

      if (tower->ieta() < 0) iRBX += 18;
      if (iRBX < 36) {
	RBXColl[iRBX].et        += tower->et(); 
	RBXColl[iRBX].hadEnergy += tower->hadEnergy(); 
	RBXColl[iRBX].emEnergy  += tower->emEnergy(); 
	RBXColl[iRBX].hcalTime  += tower->hcalTime(); 
	RBXColl[iRBX].ecalTime  += tower->ecalTime(); 
	RBXColl[iRBX].nTowers++;
      }
      /***
      std::cout << "iRBX = " << iRBX << " " 	
		<< "ieta/iphi = " <<  tower->ieta() << " / "  << tower->iphi() 
		<< " et = " << tower->et()
		<< std::endl;
      ***/
      int iHPD = tower->iphi();
      if (tower->ieta() < 0) iHPD = iHPD + 72;
      if (iHPD < 144) {
	HPDColl[iHPD].et        += tower->et(); 
	HPDColl[iHPD].hadEnergy += tower->hadEnergy(); 
	HPDColl[iHPD].emEnergy  += tower->emEnergy(); 
	HPDColl[iHPD].hcalTime  += tower->hcalTime(); 
	HPDColl[iHPD].ecalTime  += tower->ecalTime(); 
	HPDColl[iHPD].nTowers++;
      }
      /***
      std::cout << "iHPD = " << iHPD << " " 	
		<< "ieta/iphi = " <<  tower->ieta() << " / "  << tower->iphi() 
		<< " et = " << tower->et()
		<< std::endl;
      ***/

    }

  }


  // Loop over the RBX Collection
  int nRBX = 0;
  int nTowers = 0;
  for (int i=0;i<36;i++) {
    RBX_et->Fill(RBXColl[i].et);
    RBX_hadEnergy->Fill(RBXColl[i].hadEnergy);
    RBX_hcalTime->Fill(RBXColl[i].hcalTime / RBXColl[i].nTowers);
    RBX_nTowers->Fill(RBXColl[i].nTowers);
    if (RBXColl[i].hadEnergy > 3.0) {
      nRBX++;
      nTowers = RBXColl[i].nTowers;
    }
  }
  RBX_N->Fill(nRBX);
  if ( (nRBX == 1) && (nTowers > 24) ) {
    evtType = 1;
  }

  // Loop over the HPD Collection
  int nHPD = 0;
  for (int i=0;i<144;i++) {
    HPD_et->Fill(HPDColl[i].et);
    HPD_hadEnergy->Fill(HPDColl[i].hadEnergy);
    HPD_hcalTime->Fill(HPDColl[i].hcalTime / HPDColl[i].nTowers);
    HPD_nTowers->Fill(HPDColl[i].nTowers);
    if (HPDColl[i].hadEnergy > 3.0) {
      nHPD++;
      nTowers = HPDColl[i].nTowers;
    }
  }
  HPD_N->Fill(nHPD);
  if ( (nHPD == 1) && (nTowers > 6) ) {
    evtType = 2;
    cout << " nHPD = "   << nHPD 
	 << " Towers = " << nTowers
	 << " Type = "   << evtType 
	 << endl; 
  }
 
  // **************************************************************
  // ** Access Trigger Information
  // **************************************************************

  // **** Get the TriggerResults container
  Handle<TriggerResults> triggerResults;
  evt.getByLabel(theTriggerResultsLabel, triggerResults);

  Int_t JetLoPass = 0;
  
  if (triggerResults.isValid()) {
    if (DEBUG) std::cout << "trigger valid " << std::endl;
    edm::TriggerNames triggerNames;    // TriggerNames class
    triggerNames.init(*triggerResults);
    unsigned int n = triggerResults->size();
    for (unsigned int i=0; i!=n; i++) {

      //      std::cout << ">>> Trigger Name = " << triggerNames.triggerName(i)
      //                << " Accept = " << triggerResults.accept(i)
      //                << std::endl;
      //      if (DEBUG) std::cout <<  triggerNames.triggerName(i) << std::endl;

      if ( triggerNames.triggerName(i) == "HLT_Jet30" ) {
        JetLoPass =  triggerResults->accept(i);
        if (DEBUG) std::cout << "Found  HLT_Jet30 " 
			     << JetLoPass
			     << std::endl;
      }

    }
      
  } else {

    edm::Handle<TriggerResults> *tr = new edm::Handle<TriggerResults>;
    triggerResults = (*tr);

    //     std::cout << "triggerResults is not valid" << std::endl;
    //     std::cout << triggerResults << std::endl;
    //     std::cout << triggerResults.isValid() << std::endl;
    
    if (DEBUG) std::cout << "trigger not valid " << std::endl;
    edm::LogInfo("myJetAna") << "TriggerResults::HLT not found, "
      "automatically select events";
    //return;
  }

  /****
  Handle <L1GlobalTriggerReadoutRecord> gtRecord_h;
  evt.getByType (gtRecord_h); // assume only one L1 trigger record here
  const L1GlobalTriggerReadoutRecord* gtRecord = gtRecord_h.failedToGet () ? 0 : &*gtRecord_h;
  
  if (gtRecord) { // object is available
    for (int l1bit = 0; l1bit < 128; ++l1bit) {
      if (gtRecord->decisionWord() [l1bit]) h_L1TrigBit->Fill (l1bit);
    }
  }
  ****/




  // **************************************************************
  // ** Loop over the two leading CaloJets and fill some histograms
  // **************************************************************
  Handle<CaloJetCollection> caloJets;
  evt.getByLabel( CaloJetAlgorithm, caloJets );


  jetInd    = 0;
  allJetInd = 0;

  EtaOk10 = 0;
  EtaOk13 = 0;
  EtaOk40 = 0;

  //  const JetCorrector* corrector = 
  //    JetCorrector::getJetCorrector (JetCorrectionService, es);

  double highestPt;
  double nextPt;

  highestPt = 0.0;
  nextPt    = 0.0;
  
  for( CaloJetCollection::const_iterator cal = caloJets->begin(); cal != caloJets->end(); ++ cal ) {
    
    //    double scale = corrector->correction (*cal);
    double scale = 1.0;
    double corPt = scale*cal->pt();
    //    double corPt = cal->pt();
    //    cout << "Pt = " << cal->pt() << endl;
    
    if (corPt>highestPt) {
      nextPt      = highestPt;
      p4cortmp[1] = p4cortmp[0]; 
      highestPt   = corPt;
      p4cortmp[0] = scale*cal->p4();
    } else if (corPt>nextPt) {
      nextPt      = corPt;
      p4cortmp[1] = scale*cal->p4();
    }

    allJetInd++;
    if (allJetInd == 1) {
      h_jet1Pt->Fill( cal->pt() );
      if (JetLoPass != 0) h_jet1PtHLT->Fill( cal->pt() );
      pt1 = cal->pt();
      p4tmp[0] = cal->p4();
      if ( fabs(cal->eta()) < 1.0) EtaOk10++;
      if ( fabs(cal->eta()) < 1.3) EtaOk13++;
      if ( fabs(cal->eta()) < 4.0) EtaOk40++;
      
      
  }
    if (allJetInd == 2) {
      h_jet2Pt->Fill( cal->pt() );
      p4tmp[1] = cal->p4();
      if ( fabs(cal->eta()) < 1.0) EtaOk10++;
      if ( fabs(cal->eta()) < 1.3) EtaOk13++;
      if ( fabs(cal->eta()) < 4.0) EtaOk40++;
    }

    if ( cal->pt() > minJetPt) {
      h_ptCal->Fill( cal->pt() );   
      h_etaCal->Fill( cal->eta() );
      h_phiCal->Fill( cal->phi() );
      jetInd++;
    }
  }

  h_nCalJets->Fill( jetInd ); 

  if (jetInd > 1) {
    LeadMass = (p4tmp[0]+p4tmp[1]).mass();
    dijetMass->Fill( LeadMass );    
  }


  // ******************
  // *** Jet Properties
  // ******************

  int nTow1, nTow2, nTow3, nTow4;
  //  Handle<CaloJetCollection> jets;
  //  evt.getByLabel( CaloJetAlgorithm, jets );

  // *********************************************************
  // --- Loop over jets and make a list of all the used towers
  int jjet = 0;
  for ( CaloJetCollection::const_iterator ijet=caloJets->begin(); ijet!=caloJets->end(); ijet++) {
    jjet++;

    float hadEne  = ijet->hadEnergyInHB() + ijet->hadEnergyInHO() + 
                    ijet->hadEnergyInHE() + ijet->hadEnergyInHF();                   
    float emEne   = ijet->emEnergyInEB() + ijet->emEnergyInEE() + ijet->emEnergyInHF();
    float had     = ijet->energyFractionHadronic();    

    EMFraction->Fill(ijet->emEnergyFraction());    

    float j_et = ijet->et();

    // *** Barrel
    if (fabs(ijet->eta()) < 1.3) {
      totEneLeadJetEta1->Fill(hadEne+emEne); 
      hadEneLeadJetEta1->Fill(hadEne); 
      emEneLeadJetEta1->Fill(emEne);       

      if (ijet->pt() > minJetPt10) 
	hadFracEta1->Fill(had);
    }

    // *** EndCap
    if ((fabs(ijet->eta()) > 1.3) && (fabs(ijet->eta()) < 3.) ) {

      totEneLeadJetEta2->Fill(hadEne+emEne); 
      hadEneLeadJetEta2->Fill(hadEne); 
      emEneLeadJetEta2->Fill(emEne);   
    
      if (ijet->pt() > minJetPt10) 
	hadFracEta2->Fill(had);
    }

    // *** Forward
    if (fabs(ijet->eta()) > 3.) {

      totEneLeadJetEta3->Fill(hadEne+emEne); 
      hadEneLeadJetEta3->Fill(hadEne); 
      emEneLeadJetEta3->Fill(emEne); 

      if (ijet->pt() > minJetPt10) 
	hadFracEta3->Fill(had);
    }

    // *** CaloTowers in Jet
    const std::vector<CaloTowerPtr> jetCaloRefs = ijet->getCaloConstituents();
    int nConstituents = jetCaloRefs.size();
    NTowers->Fill(nConstituents);

    if (jjet == 1) {

      nTow1 = nTow2 = nTow3 = nTow4 = 0;
      for (int i = 0; i <nConstituents ; i++){

	float et  = jetCaloRefs[i]->et();

	if (et > 0.5) nTow1++;
	if (et > 1.0) nTow2++;
	if (et > 1.5) nTow3++;
	if (et > 2.0) nTow4++;
	
	hf_TowerJetEt->Fill(et/j_et);

      }

      nTowersLeadJetPt1->Fill(nTow1);
      nTowersLeadJetPt2->Fill(nTow2);
      nTowersLeadJetPt3->Fill(nTow3);
      nTowersLeadJetPt4->Fill(nTow4);

    }

  }


  // **********************
  // *** Unclustered Energy
  // **********************

  double SumPtJet(0);

  double SumEtNotJets(0);
  double SumEtJets(0);
  double SumEtTowers(0);
  double TotalClusteredE(0);
  double TotalUnclusteredE(0);

  double sumJetPx(0);
  double sumJetPy(0);

  double sumTowerAllPx(0);
  double sumTowerAllPy(0);

  double sumTowerAllEx(0);
  double sumTowerAllEy(0);

  std::vector<CaloTowerPtr>   UsedTowerList;
  std::vector<CaloTower>      TowerUsedInJets;
  std::vector<CaloTower>      TowerNotUsedInJets;


  // *********************
  // *** Hcal recHits
  // *********************

  edm::Handle<HcalSourcePositionData> spd;

  try {
    std::vector<edm::Handle<HBHERecHitCollection> > colls;
    evt.getManyByType(colls);
    std::vector<edm::Handle<HBHERecHitCollection> >::iterator i;
    for (i=colls.begin(); i!=colls.end(); i++) {
      for (HBHERecHitCollection::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++) {
        //      std::cout << *j << std::endl;
        if (j->id().subdet() == HcalBarrel) {
	  HBocc->Fill(j->id().ieta(),j->id().iphi());
	  HBEne->Fill(j->energy()); 
	  HBTime->Fill(j->time()); 
        }
        if (j->id().subdet() == HcalEndcap) {
	  HEocc->Fill(j->id().ieta(),j->id().iphi());
	  HEEne->Fill(j->energy()); 
	  HETime->Fill(j->time()); 
	  // Fill +-HE separately
	  if (j->id().ieta()<0) {
	    HEnegEne->Fill(j->energy()); 
	    HEnegTime->Fill(j->time()); 
	  } else {
	    HEposEne->Fill(j->energy()); 
	    HEposTime->Fill(j->time()); 
	  }
	  
        }

        /***
        std::cout << j->id()     << " "
                  << j->id().subdet() << " "
                  << j->id().ieta()   << " "
                  << j->id().iphi()   << " "
                  << j->id().depth()  << " "
                  << j->energy() << " "
                  << j->time()   << std::endl;
        ****/
      }
    }
  } catch (...) {
    cout << "No HB/HE RecHits." << endl;
  }


  try {
    std::vector<edm::Handle<HFRecHitCollection> > colls;
    evt.getManyByType(colls);
    std::vector<edm::Handle<HFRecHitCollection> >::iterator i;
    for (i=colls.begin(); i!=colls.end(); i++) {
      for (HFRecHitCollection::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++) {
	//	std::cout << *j << std::endl;
        if (j->id().subdet() == HcalForward) {
	  HFocc->Fill(j->id().ieta(),j->id().iphi());
	  HFEne->Fill(j->energy()); 
	  HFTime->Fill(j->time()); 
	  // Long and short fibers
	  if (j->id().depth() == 1){
	    HFLEne->Fill(j->energy()); 
	    HFLTime->Fill(j->time());
	  } else {
	    HFSEne->Fill(j->energy()); 
	    HFSTime->Fill(j->time());
	  }
        }
      }
    }
  } catch (...) {
    cout << "No HF RecHits." << endl;
  }


  try {
    std::vector<edm::Handle<HORecHitCollection> > colls;
    evt.getManyByType(colls);
    std::vector<edm::Handle<HORecHitCollection> >::iterator i;
    for (i=colls.begin(); i!=colls.end(); i++) {
      for (HORecHitCollection::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++) {
        if (j->id().subdet() == HcalOuter) {
	  HOEne->Fill(j->energy()); 
	  HOTime->Fill(j->time());
	  HOocc->Fill(j->id().ieta(),j->id().iphi());
	  // Separate SiPMs and HPDs:
	  if (((j->id().iphi()>=59 && j->id().iphi()<=70 && 
		j->id().ieta()>=11 && j->id().ieta()<=15) || 
	       (j->id().iphi()>=47 && j->id().iphi()<=58 && 
		j->id().ieta()>=5 && j->id().ieta()<=10)))
	  {  
	    HOSEne->Fill(j->energy());
	    HOSTime->Fill(j->time());
	  } else if ((j->id().iphi()<59 || j->id().iphi()>70 || 
		      j->id().ieta()<11 || j->id().ieta()>15) && 
		     (j->id().iphi()<47 || j->id().iphi()>58 ||
		      j->id().ieta()<5  || j->id().ieta()>10))
	  {
	    HOHEne->Fill(j->energy());
	    HOHTime->Fill(j->time());
	    // Separate rings -1,-2,0,1,2 in HPDs:
	    if (j->id().ieta()<= -11){
	      HOHrm2Ene->Fill(j->energy());
	      HOHrm2Time->Fill(j->time());
	    } else if (j->id().ieta()>= -10 && j->id().ieta() <= -5) {
	      HOHrm1Ene->Fill(j->energy());
	      HOHrm1Time->Fill(j->time());
	    } else if (j->id().ieta()>= -4 && j->id().ieta() <= 4) {
	      HOHr0Ene->Fill(j->energy());
	      HOHr0Time->Fill(j->time());
	    } else if (j->id().ieta()>= 5 && j->id().ieta() <= 10) {
	      HOHrp1Ene->Fill(j->energy());
	      HOHrp1Time->Fill(j->time());
	    } else if (j->id().ieta()>= 11) {
	      HOHrp2Ene->Fill(j->energy());
	      HOHrp2Time->Fill(j->time());
	    } else {
	      std::cout << "Finding events that are in no ring !?!" << std::endl;
	      std::cout << "eta = " << j->id().ieta() << std::endl;
	      
	    }
	  } else {
	    std::cout << "Finding events that are neither SiPM nor HPD!?" << std::endl;	    
	  }

	  

        }
        //      std::cout << *j << std::endl;
      }
    }
  } catch (...) {
    cout << "No HO RecHits." << endl;
  }

  try {
    std::vector<edm::Handle<EBRecHitCollection> > colls;
    evt.getManyByType(colls);
    std::vector<edm::Handle<EBRecHitCollection> >::iterator i;

    for (i=colls.begin(); i!=colls.end(); i++) {
      for (EBRecHitCollection::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++) {
	if (j->id().subdetId() == EcalBarrel) {
	  EBEne->Fill(j->energy()); 
	  EBTime->Fill(j->time()); 
	}
	//	std::cout << *j << std::endl;
	//	std::cout << j->id() << std::endl;
      }
    }
  } catch (...) {
    cout << "No EB RecHits." << endl;
  }

  try {
    std::vector<edm::Handle<EERecHitCollection> > colls;
    evt.getManyByType(colls);
    std::vector<edm::Handle<EERecHitCollection> >::iterator i;
    for (i=colls.begin(); i!=colls.end(); i++) {
      for (EERecHitCollection::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++) {
	if (j->id().subdetId() == EcalEndcap) {
	  EEEne->Fill(j->energy()); 
	  EETime->Fill(j->time());
	  // Separate +-EE;
	  EEDetId EEid = EEDetId(j->id());
	  if (!EEid.positiveZ()) 
	  {
	    EEnegEne->Fill(j->energy()); 
	    EEnegTime->Fill(j->time()); 
	  }else{
	    EEposEne->Fill(j->energy()); 
	    EEposTime->Fill(j->time()); 
	  }
	}
	//	std::cout << *j << std::endl;
      }
    }
  } catch (...) {
    cout << "No EE RecHits." << endl;
  }



  // *********************
  // *** CaloTowers
  // *********************
  //  Handle<CaloTowerCollection> caloTowers;
  //  evt.getByLabel( "towerMaker", caloTowers );

  nTow1 = nTow2 = nTow3 = nTow4 = 0;

  double sum_et = 0.0;
  double sum_ex = 0.0;
  double sum_ey = 0.0;
  //  double sum_ez = 0.0;


  //  std::cout<<">>>> Run " << evt.id().run() << " Event " << evt.id().event() << std::endl;
  // --- Loop over towers and make a lists of used and unused towers
  for (CaloTowerCollection::const_iterator tower = caloTowers->begin();
       tower != caloTowers->end(); tower++) {

    Double_t  et = tower->et();
    
    if (et > 0.5) nTow1++;
    if (et > 1.0) nTow2++;
    if (et > 1.5) nTow3++;
    if (et > 2.0) nTow4++;

    //    if ( (fabs(tower->ieta() > 42)) ||  (fabs(tower->iphi()) > 72) ) {
    //      std::cout << "ieta/iphi = " <<  tower->ieta() << " / "  << tower->iphi() << std::endl;
    //    }

    if (tower->emEnergy() > 2.0) {
      h_EmEnergy->Fill (tower->ieta(), tower->iphi(), tower->emEnergy());
    }
    if (tower->hadEnergy() > 2.0) {
      h_HadEnergy->Fill (tower->ieta(), tower->iphi(), tower->hadEnergy());
    }

    if (et>0.5) {

      ETime->Fill(tower->ecalTime());
      HTime->Fill(tower->hcalTime());

      // ********
      double phix   = tower->phi();
      //      double theta = tower->theta();
      //      double e     = tower->energy();
      //      double et    = e*sin(theta);
      //      double et    = tower->emEt() + tower->hadEt();
      double et    = tower->et();

      //      sum_ez += e*cos(theta);
      sum_et += et;
      sum_ex += et*cos(phix);
      sum_ey += et*sin(phix);
      // ********

      Double_t phi = tower->phi();
      SumEtTowers += tower->et();

      sumTowerAllEx += et*cos(phi);
      sumTowerAllEy += et*sin(phi);

    }

  }

  //  SumEt->Fill(sum_et);
  //  MET->Fill(sqrt( sum_ex*sum_ex + sum_ey*sum_ey));

  hf_sumTowerAllEx->Fill(sumTowerAllEx);
  hf_sumTowerAllEy->Fill(sumTowerAllEy);

  nTowers1->Fill(nTow1);
  nTowers2->Fill(nTow2);
  nTowers3->Fill(nTow3);
  nTowers4->Fill(nTow4);


  // *********************
  // *********************

  UsedTowerList.clear();
  TowerUsedInJets.clear();
  TowerNotUsedInJets.clear();

  // --- Loop over jets and make a list of all the used towers
  //  evt.getByLabel( CaloJetAlgorithm, jets );
  for ( CaloJetCollection::const_iterator ijet=caloJets->begin(); ijet!=caloJets->end(); ijet++) {

    Double_t jetPt  = ijet->pt();
    Double_t jetEta = ijet->eta();
    Double_t jetPhi = ijet->phi();

    //    if (jetPt>5.0) {

      Double_t jetPx = jetPt*cos(jetPhi);
      Double_t jetPy = jetPt*sin(jetPhi);

      sumJetPx +=jetPx;
      sumJetPy +=jetPy;

      const std::vector<CaloTowerPtr> jetCaloRefs = ijet->getCaloConstituents();
      int nConstituents = jetCaloRefs.size();
      for (int i = 0; i <nConstituents ; i++){
	
        UsedTowerList.push_back(jetCaloRefs[i]);
      }
      
      SumPtJet +=jetPt;

    //    }

      if ( (jetPt>80.0) && (fabs(jetEta) < 1.3) ){
	st_Pt->Fill( jetPt );
	int nConstituents = ijet->getCaloConstituents().size();
	st_Constituents->Fill( nConstituents );
	
	float maxEne = 0.;
	float totEne = 0.;
	  
	for(unsigned twr=0; twr<ijet->getCaloConstituents().size(); ++twr){
	  CaloTowerPtr tower = (ijet->getCaloConstituents())[twr];
	  //	  CaloTowerDetId id = tower->id();     
	  if( tower->et()>0. ){

	    if (tower->energy() > maxEne) maxEne = tower->energy();
	    totEne += tower->energy();

	    st_Energy->Fill( tower->energy() );
	    st_EmEnergy->Fill( tower->emEnergy() );
	    st_HadEnergy->Fill( tower->hadEnergy() );
	    st_OuterEnergy->Fill( tower->outerEnergy() );

	    st_Eta->Fill( tower->eta() );
	    st_Phi->Fill( tower->phi() );

	    st_iEta->Fill( tower->ieta() );
	    st_iPhi->Fill( tower->iphi() );

	    std::cout << ">>> Towers :  " 
		      << " " << tower->energy() 
		      << " " << tower->emEnergy()
		      << " " << tower->hadEnergy()
		      << " " << tower->outerEnergy()
		      << " " << tower->et()
		      << " " << tower->emEt() 
		      << " " << tower->hadEt() 
		      << " " << tower->outerEt() 
		      << " " << tower->eta() 
		      << " " << tower->phi() 	    
		      << std::endl;
	  
	  }
	}
	st_Frac->Fill( maxEne / totEne );

      }

  }

  int NTowersUsed = UsedTowerList.size();

  // --- Loop over towers and make a lists of used and unused towers
  for (CaloTowerCollection::const_iterator tower = caloTowers->begin();
       tower != caloTowers->end(); tower++) {

    CaloTower  t = *tower;
    Double_t  et = tower->et();

    if(et>0) {

      Double_t phi = tower->phi();
      SumEtTowers += tower->et();

      sumTowerAllPx += et*cos(phi);
      sumTowerAllPy += et*sin(phi);

      bool used = false;

      for(int i=0; i<NTowersUsed; i++){
        if(tower->id() == UsedTowerList[i]->id()){
          used=true;
          break;
        }
      }

      if (used) {
        TowerUsedInJets.push_back(t);
      } else {
        TowerNotUsedInJets.push_back(t);
      }
    }
  }

  int nUsed    = TowerUsedInJets.size();
  int nNotUsed = TowerNotUsedInJets.size();

  SumEtJets    = 0;
  SumEtNotJets = 0;
  TotalClusteredE   = 0;
  TotalUnclusteredE = 0;

  for(int i=0;i<nUsed;i++){
    SumEtJets += TowerUsedInJets[i].et();
    h_ClusteredE->Fill(TowerUsedInJets[i].energy());
    if (TowerUsedInJets[i].energy() > 1.0) 
      TotalClusteredE += TowerUsedInJets[i].energy();
  }
  h_jetEt->Fill(SumEtJets);

  for(int i=0;i<nNotUsed;i++){
    if (TowerNotUsedInJets[i].et() > 0.5)
      SumEtNotJets += TowerNotUsedInJets[i].et();
    h_UnclusteredEt->Fill(TowerNotUsedInJets[i].et());
    h_UnclusteredEts->Fill(TowerNotUsedInJets[i].et());
    h_UnclusteredE->Fill(TowerNotUsedInJets[i].energy());
    if (TowerNotUsedInJets[i].energy() > 1.0)  
      TotalUnclusteredE += TowerNotUsedInJets[i].energy();
  }

  h_TotalClusteredE->Fill(TotalClusteredE);
  h_TotalUnclusteredE->Fill(TotalUnclusteredE);
  h_TotalUnclusteredEt->Fill(SumEtNotJets);

  // ********************************
  // *** CaloMET
  // ********************************

  edm::Handle<reco::CaloMETCollection> calometcoll;
  evt.getByLabel("met", calometcoll);
  if (calometcoll.isValid()) {
    const CaloMETCollection *calometcol = calometcoll.product();
    const CaloMET *calomet;
    calomet = &(calometcol->front());

    double caloSumET  = calomet->sumEt();
    double caloMET    = calomet->pt();
    double caloMETSig = calomet->mEtSig();
    double caloMEx    = calomet->px();
    double caloMEy    = calomet->py();
    double caloMETPhi = calomet->phi();

    SumEt->Fill(caloSumET);
    MET->Fill(caloMET);
    if (evtType == 0) MET_Tower->Fill(caloMET);
    if (evtType == 1) MET_RBX->Fill(caloMET);
    if (evtType == 2) MET_HPD->Fill(caloMET);
    METSig->Fill(caloMETSig);
    MEx->Fill(caloMEx);
    MEy->Fill(caloMEy);
    METPhi->Fill(caloMETPhi);

    /***
    double caloEz     = calomet->e_longitudinal();

    double caloMaxEtInEMTowers    = calomet->maxEtInEmTowers();
    double caloMaxEtInHadTowers   = calomet->maxEtInHadTowers();
    double caloEtFractionHadronic = calomet->etFractionHadronic();
    double caloEmEtFraction       = calomet->emEtFraction();

    double caloHadEtInHB = calomet->hadEtInHB();
    double caloHadEtInHO = calomet->hadEtInHO();
    double caloHadEtInHE = calomet->hadEtInHE();
    double caloHadEtInHF = calomet->hadEtInHF();
    double caloEmEtInEB  = calomet->emEtInEB();
    double caloEmEtInEE  = calomet->emEtInEE();
    double caloEmEtInHF  = calomet->emEtInHF();
    ****/
  }

  // ********************************
  // *** Vertex
  // ********************************
  /****
  edm::Handle<reco::VertexCollection> vertexCollection;
  evt.getByLabel("offlinePrimaryVertices", vertexCollection);
  const reco::VertexCollection vC = *(vertexCollection.product());

  std::cout << "Reconstructed "<< vC.size() << " vertices" << std::endl ;
  for (reco::VertexCollection::const_iterator vertex=vC.begin(); vertex!=vC.end(); vertex++){

    h_Vx->Fill(vertex->x());
    h_Vy->Fill(vertex->y());
    h_Vz->Fill(vertex->z());
    h_VNTrks->Fill(vertex->tracksSize());

  }
  ****/

  // ********************************
  // *** Tracks
  // ********************************
  /***
  edm::Handle<reco::TrackCollection> trackCollection;
  //  evt.getByLabel("ctfWithMaterialTracks", trackCollection);
  evt.getByLabel("generalTracks", trackCollection);

  const reco::TrackCollection tC = *(trackCollection.product());
  std::cout << "Reconstructed "<< tC.size() << " tracks" << std::endl ;
  h_Trk_NTrk->Fill(tC.size());
  for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++){

    h_Trk_pt->Fill(track->pt());

  }
  ****/

    /****
    std::cout << "Track number "<< i << std::endl ;
    std::cout << "\tmomentum: " << track->momentum()<< std::endl;
    std::cout << "\tPT: " << track->pt()<< std::endl;
    std::cout << "\tvertex: " << track->vertex()<< std::endl;
    std::cout << "\timpact parameter: " << track->d0()<< std::endl;
    std::cout << "\tcharge: " << track->charge()<< std::endl;
    std::cout << "\tnormalizedChi2: " << track->normalizedChi2()<< std::endl;

    cout<<"\tFrom EXTRA : "<<endl;
    cout<<"\t\touter PT "<< track->outerPt()<<endl;
    std::cout << "\t direction: " << track->seedDirection() << std::endl;
    ****/

  // ********************************
  // *** Photons
  // ********************************
  /***
  edm::Handle<reco::PhotonCollection> photonCollection;
  evt.getByLabel("photons", photonCollection);
  const reco::PhotonCollection pC = *(photonCollection.product());

  std::cout << "Reconstructed "<< pC.size() << " photons" << std::endl ;
  for (reco::PhotonCollection::const_iterator photon=pC.begin(); photon!=pC.end(); photon++){
  }
  ***/

  // ********************************
  // *** Muons
  // ********************************
  /***
  edm::Handle<reco::MuonCollection> muonCollection;
  evt.getByLabel("muons", muonCollection);

  const reco::MuonCollection mC = *(muonCollection.product());

  std::cout << "Reconstructed "<< mC.size() << " muons" << std::endl ;
  for (reco::MuonCollection::const_iterator muon=mC.begin(); muon!=mC.end(); muon++){
  }
  ***/




  // ********************************
  // *** Events passing seletion cuts
  // ********************************

  // --- Cosmic Cleanup
  // --- Vertex
  // --- Eta 

  int iJet; 
  iJet = 0;
  for( CaloJetCollection::const_iterator ijet = caloJets->begin(); ijet != caloJets->end(); ++ ijet ) {
    
    //    if ( (fabs(ijet->eta()) < 1.3) && 
    //	 (fabs(ijet->pt())  > 20.) ) {

	 //	 (ijet->emEnergyFraction() > 0.01) &&
	 //	 (ijet->emEnergyFraction() > 0.99) ) {

    iJet++; 
    if (iJet == 1) {
      cout << " Event Type = "   << evtType 
	   << " pt = " << ijet->pt()
	   << endl; 
    }
      h_pt->Fill(ijet->pt());
      if (evtType == 0) h_ptTower->Fill(ijet->pt());
      if (evtType == 1) h_ptRBX->Fill(ijet->pt());
      if (evtType == 2) h_ptHPD->Fill(ijet->pt());
      h_et->Fill(ijet->et());
      h_eta->Fill(ijet->eta());
      h_phi->Fill(ijet->phi());
      
      //    }    
  }



  //*****************************
  //*** Get the GenJet collection
  //*****************************

      /**************
  Handle<GenJetCollection> genJets;
  evt.getByLabel( GenJetAlgorithm, genJets );

  //Loop over the two leading GenJets and fill some histograms
  jetInd    = 0;
  allJetInd = 0;
  for( GenJetCollection::const_iterator gen = genJets->begin(); gen != genJets->end(); ++ gen ) {
    allJetInd++;
    if (allJetInd == 1) {
      p4tmp[0] = gen->p4();
    }
    if (allJetInd == 2) {
      p4tmp[1] = gen->p4();
    }

    if ( (allJetInd == 1) || (allJetInd == 2) ) {
      h_ptGenL->Fill( gen->pt() );
      h_etaGenL->Fill( gen->eta() );
      h_phiGenL->Fill( gen->phi() );
    }

    if ( gen->pt() > minJetPt) {
      // std::cout << "GEN JET1 #" << jetInd << std::endl << gen->print() << std::endl;
      h_ptGen->Fill( gen->pt() );
      h_etaGen->Fill( gen->eta() );
      h_phiGen->Fill( gen->phi() );
      jetInd++;
    }
  }

  h_nGenJets->Fill( jetInd );
      *******/


}

// ***********************************
// ***********************************
void myJetAna::endJob() {

}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(myJetAna);

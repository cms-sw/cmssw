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
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

// #include "DataFormats/Scalers/interface/DcsStatus.h"


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


#define INVALID 9999.
#define DEBUG false
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



float totBNC, nBNC[4000];

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

void myJetAna::beginJob( ) {



  edm::Service<TFileService> fs;

  // --- passed selection cuts 
  h_pt     = fs->make<TH1F>( "pt",  "Jet p_{T}", 100, 0, 50 );
  h_ptRBX  = fs->make<TH1F>( "ptRBX",  "RBX: Jet p_{T}", 100, 0, 50 );
  h_ptHPD  = fs->make<TH1F>( "ptHPD",  "HPD: Jet p_{T}", 100, 0, 50 );
  h_ptTower  = fs->make<TH1F>( "ptTower",  "Jet p_{T}", 100, 0, 50 );
  h_et     = fs->make<TH1F>( "et",  "Jet E_{T}", 100, 0, 50 );
  h_eta    = fs->make<TH1F>( "eta", "Jet #eta", 100, -4, 4 );
  h_phi    = fs->make<TH1F>( "phi", "Jet #phi", 50, -M_PI, M_PI );
  // ---

  hitEtaEt  = fs->make<TH1F>( "hitEtaEt", "RecHit #eta", 90, -45, 45 );
  hitEta    = fs->make<TH1F>( "hitEta", "RecHit #eta", 90, -45, 45 );
  hitPhi    = fs->make<TH1F>( "hitPhi", "RecHit #phi", 73, 0, 73 );

  caloEtaEt  = fs->make<TH1F>( "caloEtaEt", "CaloTower #eta", 100, -4, 4 );
  caloEta    = fs->make<TH1F>( "caloEta", "CaloTower #eta", 100, -4, 4 );
  caloPhi    = fs->make<TH1F>( "caloPhi", "CaloTower #phi", 50, -M_PI, M_PI );

  dijetMass  =  fs->make<TH1F>("dijetMass","DiJet Mass",100,0,100);

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

  SumEt  = fs->make<TH1F>("SumEt","SumEt",100,0,100);
  MET    = fs->make<TH1F>("MET",  "MET",100,0,50);
  METSig = fs->make<TH1F>("METSig",  "METSig",100,0,50);
  MEx    = fs->make<TH1F>("MEx",  "MEx",100,-20,20);
  MEy    = fs->make<TH1F>("MEy",  "MEy",100,-20,20);
  METPhi = fs->make<TH1F>("METPhi",  "METPhi",315,0,3.15);
  MET_RBX    = fs->make<TH1F>("MET_RBX",  "MET",100,0,1000);
  MET_HPD    = fs->make<TH1F>("MET_HPD",  "MET",100,0,1000);
  MET_Tower  = fs->make<TH1F>("MET_Tower",  "MET",100,0,1000);


  h_Vx     = fs->make<TH1F>("Vx",  "Vx",100,-0.5,0.5);
  h_Vy     = fs->make<TH1F>("Vy",  "Vy",100,-0.5,0.5);
  h_Vz     = fs->make<TH1F>("Vz",  "Vz",100,-20,20);
  h_VNTrks = fs->make<TH1F>("VNTrks",  "VNTrks",10,1,100);

  h_Trk_pt   = fs->make<TH1F>("Trk_pt",  "Trk_pt",100,0,20);
  h_Trk_NTrk = fs->make<TH1F>("Trk_NTrk",  "Trk_NTrk",20,0,20);

  hf_sumTowerAllEx = fs->make<TH1F>("sumTowerAllEx","Tower Ex",100,-1000,1000);
  hf_sumTowerAllEy = fs->make<TH1F>("sumTowerAllEy","Tower Ey",100,-1000,1000);

  hf_TowerJetEt   = fs->make<TH1F>("TowerJetEt","Tower/Jet Et 1",50,0,1);

  ETime = fs->make<TH1F>("ETime","Ecal Time",200,-200,200);
  HTime = fs->make<TH1F>("HTime","Hcal Time",200,-200,200);

  towerHadEn    = fs->make<TH1F>("towerHadEn" ,"Hadronic Energy in Calo Tower",2000,-100,100);
  towerEmEn  	= fs->make<TH1F>("towerEmEn"  ,"EM Energy in Calo Tower",2000,-100,100);
  towerOuterEn  = fs->make<TH1F>("towerOuterEn"  ,"HO Energy in Calo Tower",2000,-100,100);

  towerEmFrac	= fs->make<TH1F>("towerEmFrac","EM Fraction of Energy in Calo Tower",100,-1.,1.);

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

  HBEneOOT  = fs->make<TH1F>( "HBEneOOT",  "HBEneOOT", 200, -5, 10 );
  HEEneOOT  = fs->make<TH1F>( "HEEneOOT",  "HEEneOOT", 200, -5, 10 );
  HFEneOOT  = fs->make<TH1F>( "HFEneOOT",  "HFEneOOT", 200, -5, 10 );
  HOEneOOT  = fs->make<TH1F>( "HOEneOOT",  "HOEneOOT", 200, -5, 10 );

  HBEne     = fs->make<TH1F>( "HBEne",  "HBEne", 200, -5, 10 );
  HBEneTh   = fs->make<TH1F>( "HBEneTh",  "HBEneTh", 200, -5, 10 );
  HBEneX    = fs->make<TH1F>( "HBEneX",  "HBEneX", 200, -5, 10 );
  HBEneY    = fs->make<TH1F>( "HBEneY",  "HBEnedY", 200, -5, 10 );
  HBTime    = fs->make<TH1F>( "HBTime", "HBTime", 200, -100, 100 );
  HBTimeTh  = fs->make<TH1F>( "HBTimeTh", "HBTimeTh", 200, -100, 100 );
  HBTimeX   = fs->make<TH1F>( "HBTimeX", "HBTimeX", 200, -100, 100 );
  HBTimeY   = fs->make<TH1F>( "HBTimeY", "HBTimeY", 200, -100, 100 );
  HEEne     = fs->make<TH1F>( "HEEne",  "HEEne", 200, -5, 10 );
  HEEneTh   = fs->make<TH1F>( "HEEneTh",  "HEEneTh", 200, -5, 10 );
  HEEneX    = fs->make<TH1F>( "HEEneX",  "HEEneX", 200, -5, 10 );
  HEEneY    = fs->make<TH1F>( "HEEneY",  "HEEneY", 200, -5, 10 );
  HEposEne  = fs->make<TH1F>( "HEposEne",  "HEposEne", 200, -5, 10 );
  HEnegEne  = fs->make<TH1F>( "HEnegEne",  "HEnegEne", 200, -5, 10 );
  HETime    = fs->make<TH1F>( "HETime", "HETime", 200, -100, 100 );
  HETimeTh  = fs->make<TH1F>( "HETimeTh", "HETimeTh", 200, -100, 100 );
  HETimeX   = fs->make<TH1F>( "HETimeX", "HETimeX", 200, -100, 100 );
  HETimeY   = fs->make<TH1F>( "HETimeY", "HETimeY", 200, -100, 100 );
  HEposTime = fs->make<TH1F>( "HEposTime",  "HEposTime", 200, -100, 100 );
  HEnegTime = fs->make<TH1F>( "HEnegTime",  "HEnegTime", 200, -100, 100 );
  HOEne     = fs->make<TH1F>( "HOEne",  "HOEne", 200, -5, 10 );
  HOEneTh   = fs->make<TH1F>( "HOEneTh",  "HOEneTh", 200, -5, 10 );
  HOTime    = fs->make<TH1F>( "HOTime", "HOTime", 200, -100, 100 );
  HOTimeTh  = fs->make<TH1F>( "HOTimeTh", "HOTimeTh", 200, -100, 100 );

  // Histos for separating SiPMs and HPDs in HO:
  HOSEne     = fs->make<TH1F>( "HOSEne",  "HOSEne", 12000, -20, 100 );
  HOSTime    = fs->make<TH1F>( "HOSTime", "HOSTime", 200, -100, 100 );
  HOHEne     = fs->make<TH1F>( "HOHEne",  "HOHEne", 12000, -20, 100 );
  HOHTime    = fs->make<TH1F>( "HOHTime", "HOHTime", 200, -100, 100 );

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

  HBTvsE    = fs->make<TH2F>( "HBTvsE", "HBTvsE",100, -5, 50, 100, -100, 100);
  HETvsE    = fs->make<TH2F>( "HETvsE", "HETvsE",100, -5, 50, 100, -100, 100);
  HFTvsE    = fs->make<TH2F>( "HFTvsE", "HFTvsE",100, -5, 50, 100, -100, 100);
  HOTvsE    = fs->make<TH2F>( "HOTvsE", "HOTvsE",100, -5, 50, 100, -100, 100);

  HFvsZ    = fs->make<TH2F>( "HFvsZ", "HFvsZ",100,-50,50,100,-50,50);

  HOocc    = fs->make<TH2F>( "HOocc", "HOocc",81,-40.5,40.5,70,0.5,70.5);
  HBocc    = fs->make<TH2F>( "HBocc", "HBocc",81,-40.5,40.5,70,0.5,70.5);
  HEocc    = fs->make<TH2F>( "HEocc", "HEocc",81,-40.5,40.5,70,0.5,70.5);
  HFocc    = fs->make<TH2F>( "HFocc", "HFocc",81,-40.5,40.5,70,0.5,70.5);

  HFEne     = fs->make<TH1F>( "HFEne",  "HFEne", 210, -10, 200 );
  HFEneTh   = fs->make<TH1F>( "HFEneTh",  "HFEneTh", 210, -10, 200 );
  HFEneP    = fs->make<TH1F>( "HFEneP",  "HFEneP", 200, -5, 10 );
  HFEneM    = fs->make<TH1F>( "HFEneM",  "HFEneM", 200, -5, 10 );
  HFTime    = fs->make<TH1F>( "HFTime", "HFTime", 200, -100, 100 );
  HFTimeTh  = fs->make<TH1F>( "HFTimeTh", "HFTimeTh", 200, -100, 100 );
  HFTimeP   = fs->make<TH1F>( "HFTimeP", "HFTimeP", 100, -100, 50 );
  HFTimeM   = fs->make<TH1F>( "HFTimeM", "HFTimeM", 100, -100, 50 );
  HFTimePMa = fs->make<TH1F>( "HFTimePMa", "HFTimePMa", 100, -100, 100 );
  HFTimePM  = fs->make<TH1F>( "HFTimePM", "HFTimePM", 100, -100, 100 );

  // Histos for separating HF long/short fibers:
  HFLEne     = fs->make<TH1F>( "HFLEne",  "HFLEne", 200, -5, 10 );
  HFLTime    = fs->make<TH1F>( "HFLTime", "HFLTime", 200, -100, 100 );
  HFSEne     = fs->make<TH1F>( "HFSEne",  "HFSEne", 200, -5, 10 );
  HFSTime    = fs->make<TH1F>( "HFSTime", "HFSTime", 200, -100, 100 );

  HFLvsS     = fs->make<TH2F>( "HFLvsS", "HFLvsS",220,-20,200,220,-20,200);


  EBEne     = fs->make<TH1F>( "EBEne",  "EBEne", 200, -5, 10 );
  EBEneTh   = fs->make<TH1F>( "EBEneTh",  "EBEneTh", 200, -5, 10 );
  EBEneX    = fs->make<TH1F>( "EBEneX",  "EBEneX", 200, -5, 10 );
  EBEneY    = fs->make<TH1F>( "EBEneY",  "EBEneY", 200, -5, 10 );
  EBTime    = fs->make<TH1F>( "EBTime", "EBTime", 200, -100, 100 );
  EBTimeTh  = fs->make<TH1F>( "EBTimeTh", "EBTimeTh", 200, -100, 100 );
  EBTimeX   = fs->make<TH1F>( "EBTimeX", "EBTimeX", 200, -100, 100 );
  EBTimeY   = fs->make<TH1F>( "EBTimeY", "EBTimeY", 200, -100, 100 );
  EEEne     = fs->make<TH1F>( "EEEne",  "EEEne", 200, -5, 10 );
  EEEneTh   = fs->make<TH1F>( "EEEneTh",  "EEEneTh", 200, -5, 10 );
  EEEneX    = fs->make<TH1F>( "EEEneX",  "EEEneX", 200, -5, 10 );
  EEEneY    = fs->make<TH1F>( "EEEneY",  "EEEneY", 200, -5, 10 );
  EEnegEne  = fs->make<TH1F>( "EEnegEne",  "EEnegEne", 200, -5, 10 );
  EEposEne  = fs->make<TH1F>( "EEposEne",  "EEposEne", 200, -5, 10 );
  EETime    = fs->make<TH1F>( "EETime", "EETime", 200, -100, 100 );
  EETimeTh  = fs->make<TH1F>( "EETimeTh", "EETimeTh", 200, -100, 100 );
  EETimeX   = fs->make<TH1F>( "EETimeX", "EETimeX", 200, -100, 100 );
  EETimeY   = fs->make<TH1F>( "EETimeY", "EETimeY", 200, -100, 100 );
  EEnegTime = fs->make<TH1F>( "EEnegTime", "EEnegTime", 200, -100, 100 );
  EEposTime = fs->make<TH1F>( "EEposTime", "EEposTime", 200, -100, 100 );

  h_ptCal     = fs->make<TH1F>( "ptCal",  "p_{T} of CalJet", 100, 0, 50 );
  h_etaCal    = fs->make<TH1F>( "etaCal", "#eta of  CalJet", 100, -4, 4 );
  h_phiCal    = fs->make<TH1F>( "phiCal", "#phi of  CalJet", 50, -M_PI, M_PI );

  h_nGenJets  =  fs->make<TH1F>( "nGenJets",  "Number of GenJets", 20, 0, 20 );

  h_ptGen     =  fs->make<TH1F>( "ptGen",  "p_{T} of GenJet", 100, 0, 50 );
  h_etaGen    =  fs->make<TH1F>( "etaGen", "#eta of GenJet", 100, -4, 4 );
  h_phiGen    =  fs->make<TH1F>( "phiGen", "#phi of GenJet", 50, -M_PI, M_PI );

  h_ptGenL    =  fs->make<TH1F>( "ptGenL",  "p_{T} of GenJetL", 100, 0, 50 );
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

  jetHOEne              = fs->make<TH1F>("jetHOEne" ,"HO Energy in Jet",100, 0,100);
  jetEMFraction         = fs->make<TH1F>( "jetEMFraction", "Jet EM Fraction", 100, -1.1, 1.1 );
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


  EBvHB           = fs->make<TH2F>( "EBvHB", "EB vs HB",1000,0,4500000.,1000,0,1000000.);
  EEvHE           = fs->make<TH2F>( "EEvHE", "EE vs HE",1000,0,4500000.,1000,0,200000.);

  ECALvHCAL       = fs->make<TH2F>( "ECALvHCAL", "ECAL vs HCAL",100,0,20000000.,100,-500000,500000.);
  ECALvHCALEta1   = fs->make<TH2F>( "ECALvHCALEta1", "ECAL vs HCALEta1",100,0,20000000.,100,-500000,500000.);
  ECALvHCALEta2   = fs->make<TH2F>( "ECALvHCALEta2", "ECAL vs HCALEta2",100,0,20000000.,100,-500000,500000.);
  ECALvHCALEta3   = fs->make<TH2F>( "ECALvHCALEta3", "ECAL vs HCALEta3",100,0,20000000.,100,-500000,500000.);

  EMF_Eta   = fs->make<TProfile>("EMF_Eta","EMF Eta", 100, -50, 50, 0, 10);
  EMF_Phi   = fs->make<TProfile>("EMF_Phi","EMF Phi", 100, 0, 100, 0, 10);
  EMF_EtaX  = fs->make<TProfile>("EMF_EtaX","EMF EtaX", 100, -50, 50, 0, 10);
  EMF_PhiX  = fs->make<TProfile>("EMF_PhiX","EMF PhiX", 100, 0, 100, 0, 10);


  totBNC = 0;
  for (int i=0; i<4000; i++)  nBNC[i] = 0;

}

// ************************
// ************************
void myJetAna::analyze( const edm::Event& evt, const edm::EventSetup& es ) {
 
  using namespace edm;

  bool Pass, Pass_HFTime, Pass_DiJet, Pass_BunchCrossing, Pass_Trigger, Pass_Vertex;

  int EtaOk10, EtaOk13, EtaOk40;

  double LeadMass;

  double HFRecHit[100][100][2];

  double towerEtCut, towerECut, towerE;

  towerEtCut = 1.0;
  towerECut  = 1.0;

  double HBHEThreshold = 2.0;
  double HFThreshold   = 2.0;
  double HOThreshold   = 2.0;
  double EBEEThreshold = 2.0;

  float pt1;

  float minJetPt = 5.;
  float minJetPt10 = 10.;
  int jetInd, allJetInd;
  LeadMass = -1;

  //  Handle<DcsStatusCollection> dcsStatus;
  //  evt.getByLabel("scalersRawToDigi", dcsStatus);
  //  std::cout << dcsStatus << std::endl;
  //  if (dcsStatus.isValid()) {
  //  }

  //  DcsStatus dcsStatus;
  //  Handle<DcsStatus> dcsStatus;
  //  evt.getByLabel("dcsStatus", dcsStatus);


  math::XYZTLorentzVector p4tmp[2], p4cortmp[2];

  // --------------------------------------------------------------
  // --------------------------------------------------------------

  std::cout << ">>>> ANA: Run = "    << evt.id().run() 
	    << " Event = " << evt.id().event()
	    << " Bunch Crossing = " << evt.bunchCrossing() 
	    << " Orbit Number = "   << evt.orbitNumber()
	    << " Luminosity Block = "  << evt.luminosityBlock()
	    <<  std::endl;
  
  // *********************
  // *** Filter Event
  // *********************
  Pass = false;

  /***
  if (evt.bunchCrossing()== 100) {
    Pass = true;
  } else {
    Pass = false;
  }
  ***/

  // ***********************
  // ***  Pass Trigger
  // ***********************


  // **** Get the TriggerResults container
  Handle<TriggerResults> triggerResults;
  evt.getByLabel(theTriggerResultsLabel, triggerResults);
  //  evt.getByLabel("TriggerResults::HLT", triggerResults);

  if (triggerResults.isValid()) {
    if (DEBUG) std::cout << "trigger valid " << std::endl;
    const edm::TriggerNames & triggerNames = evt.triggerNames(*triggerResults);
    unsigned int n = triggerResults->size();
    for (unsigned int i=0; i!=n; i++) {

      /***
      std::cout << ">>> Trigger Name (" <<  i << ") = " << triggerNames.triggerName(i)
		<< " Accept = " << triggerResults->accept(i)
		<< std::endl;
      ***/
      /****
      if (triggerResults->accept(i) == 1) {
	std::cout << "+++ Trigger Name (" <<  i << ") = " << triggerNames.triggerName(i)
		  << " Accept = " << triggerResults->accept(i)
		  << std::endl;
      }
      ****/

      if (DEBUG) std::cout <<  triggerNames.triggerName(i) << std::endl;

      //      if ( (triggerNames.triggerName(i) == "HLT_ZeroBias")  || 
      //	   (triggerNames.triggerName(i) == "HLT_MinBias")   || 
      //	   (triggerNames.triggerName(i) == "HLT_MinBiasHcal") )  {

      if (triggerNames.triggerName(i) == "HLT_MinBiasBSC") {
	Pass_Trigger = true;
      } else {
	Pass_Trigger = false;
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

    Pass_Trigger = true;

    //return;
  }

  
  /***
  Handle<L1GlobalTriggerReadoutRecord> gtRecord;
  evt.getByLabel("gtDigis",gtRecord);
  const TechnicalTriggerWord tWord = gtRecord->technicalTriggerWord();

  if (gtRecord.isValid()) {
    if (tWord.at(40)) {
      Pass_Trigger = true;
    } else {
      Pass_Trigger = false;
    }
  } else {
    Pass_Trigger = false;
  }
  ****/


  // *************************
  // ***  Pass Bunch Crossing
  // *************************

  // *** Check Luminosity Section
  if (evt.id().run() == 122294)
    if ( (evt.luminosityBlock() >= 37) && (evt.luminosityBlock() <= 43) ) 
      Pass = true;
  if (evt.id().run() == 122314)
    if ( (evt.luminosityBlock() >= 24) && (evt.luminosityBlock() <= 37) ) 
      Pass = true;
  if (evt.id().run() == 123575)
      Pass = true;
  if (evt.id().run() == 123596)
      Pass = true;

  // ***********
  if (evt.id().run() == 124009) {
    if ( (evt.bunchCrossing() == 51) ||
	 (evt.bunchCrossing() == 151) ||
	 (evt.bunchCrossing() == 2824) ) {
      Pass = true;
    }
  }

  if (evt.id().run() == 124020) {
    if ( (evt.bunchCrossing() == 51) ||
	 (evt.bunchCrossing() == 151) ||
	 (evt.bunchCrossing() == 2824) ) {
      Pass = true;
    }
  }

  if (evt.id().run() == 124024) {
    if ( (evt.bunchCrossing() == 51) ||
	 (evt.bunchCrossing() == 151) ||
	 (evt.bunchCrossing() == 2824) ) {
      Pass = true;
    }
  }

  if ( (evt.bunchCrossing() == 51)   ||
       (evt.bunchCrossing() == 151)  ||
       (evt.bunchCrossing() == 2596) || 
       (evt.bunchCrossing() == 2724) || 
       (evt.bunchCrossing() == 2824) ||
       (evt.bunchCrossing() == 3487) ) {
    Pass_BunchCrossing = true;
  } else {
    Pass_BunchCrossing = false;
  }
  

  // ***********************
  // ***  Pass HF Timing 
  // ***********************

  double HFM_ETime, HFP_ETime;
  double HFM_E, HFP_E;
  double HF_PMM;

  HFM_ETime = 0.;
  HFM_E = 0.;
  HFP_ETime = 0.;
  HFP_E = 0.;

  for (int i=0; i<100; i++) {
    for (int j=0; j<100; j++) {
      HFRecHit[i][j][0] = -10.;
      HFRecHit[i][j][1] = -10.;
    }
  }


  try {
    std::vector<edm::Handle<HFRecHitCollection> > colls;
    evt.getManyByType(colls);
    std::vector<edm::Handle<HFRecHitCollection> >::iterator i;
    for (i=colls.begin(); i!=colls.end(); i++) {
      for (HFRecHitCollection::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++) {
        if (j->id().subdet() == HcalForward) {

	  float en = j->energy();
	  HcalDetId id(j->detid().rawId());
	  int ieta = id.ieta();
	  int iphi = id.iphi();
	  int depth = id.depth();
	  
	  HFRecHit[ieta+41][iphi][depth-1] = en;

	  if (j->id().ieta()<0) {
	    if (j->energy() > HFThreshold) {
	      HFM_ETime += j->energy()*j->time(); 
	      HFM_E     += j->energy();
	    }
	  } else {
	    if (j->energy() > HFThreshold) {
	      HFP_ETime += j->energy()*j->time(); 
	      HFP_E     += j->energy();
	    }
	  }

        }
      }
    }
  } catch (...) {
    cout << "No HF RecHits." << endl;
  }

  if ((HFP_E > 0.) && (HFM_E > 0.)) {
    HF_PMM = (HFP_ETime / HFP_E) - (HFM_ETime / HFM_E);
    HFTimePMa->Fill(HF_PMM); 
  } else {
    HF_PMM = INVALID;
  }

  
  if (fabs(HF_PMM) < 10.)  {
    Pass_HFTime = true;
  } else {
    Pass_HFTime = false;
  }


  // **************************
  // ***  Pass DiJet Criteria
  // **************************
  double highestPt;
  double nextPt;
  double dphi;
  int    nDiJet, nJet;

  nJet      = 0;
  nDiJet    = 0;
  highestPt = 0.0;
  nextPt    = 0.0;

  allJetInd = 0;
  Handle<CaloJetCollection> caloJets;
  evt.getByLabel( CaloJetAlgorithm, caloJets );
  for( CaloJetCollection::const_iterator cal = caloJets->begin(); cal != caloJets->end(); ++ cal ) {

    // TODO: verify first two jets are the leading jets
    if (nJet == 0) p4tmp[0] = cal->p4();
    if (nJet == 1) p4tmp[1] = cal->p4();

    if ( (cal->pt() > 3.) && 
	 (fabs(cal->eta()) < 3.0) ) { 
      nDiJet++;
    }
    nJet++;

  }  
  

  if (nDiJet > 1) {
    dphi = deltaPhi(p4tmp[0].phi(), p4tmp[1].phi());
    Pass_DiJet = true;
  } else {
    dphi = INVALID;
    Pass_DiJet = false;
  }
      

  // **************************
  // ***  Pass Vertex
  // **************************
  double VTX = 0.;
  int nVTX = 0;

  edm::Handle<reco::VertexCollection> vertexCollection;
  evt.getByLabel("offlinePrimaryVertices", vertexCollection);
  const reco::VertexCollection vC = *(vertexCollection.product());

  std::cout << "Reconstructed "<< vC.size() << " vertices" << std::endl ;

  nVTX = vC.size();
  for (reco::VertexCollection::const_iterator vertex=vC.begin(); vertex!=vC.end(); vertex++){
    VTX  = vertex->z();
  }

  if ( (fabs(VTX) < 20.) && (nVTX > 0) ){
    Pass_Vertex = true;
  } else {
    Pass_Vertex = false;
  }

  // ***********************
  // ***********************


  nBNC[evt.bunchCrossing()]++;
  totBNC++;
    
  //  Pass = true;

  // *** Check for tracks
  //  edm::Handle<reco::TrackCollection> trackCollection;
  //  evt.getByLabel("generalTracks", trackCollection);
  //  const reco::TrackCollection tC = *(trackCollection.product());
  //  if ((Pass) && (tC.size()>1)) {
  //  } else {
  //    Pass = false;
  //  } 


  // **************************
  // *** Event Passed Selection
  // **************************


  if (evt.id().run() == 1) {
    if ( (Pass_DiJet)         &&
	 (Pass_Vertex) ) {
      Pass = true;
    } else {
      Pass = false;
    }
  } else {
    if ( (Pass_BunchCrossing) && 
	 (Pass_HFTime)        &&
	 (Pass_Vertex) ) {
      Pass = true;
    } else {
      Pass = false;
    }
  }

  std::cout << "+++ Result " 
	    << " Event = " 
	    << evt.id().run()
	    << " LS = "
	    << evt.luminosityBlock()
	    << " dphi = "
	    << dphi
	    << " Pass = " 
	    << Pass
	    << std::endl;
        
  if (Pass) {
 

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

  double ETotal, emFrac;
  double HCALTotalCaloTowerE, ECALTotalCaloTowerE;
  double HCALTotalCaloTowerE_Eta1, ECALTotalCaloTowerE_Eta1;
  double HCALTotalCaloTowerE_Eta2, ECALTotalCaloTowerE_Eta2;
  double HCALTotalCaloTowerE_Eta3, ECALTotalCaloTowerE_Eta3;

  ETotal = 0.;
  emFrac = 0.;
    
  HCALTotalCaloTowerE = 0;
  ECALTotalCaloTowerE = 0;
  HCALTotalCaloTowerE_Eta1 = 0.;
  ECALTotalCaloTowerE_Eta1 = 0.;
  HCALTotalCaloTowerE_Eta2 = 0.; 
  ECALTotalCaloTowerE_Eta2 = 0.;
  HCALTotalCaloTowerE_Eta3 = 0.;
  ECALTotalCaloTowerE_Eta3 = 0.;

  for (CaloTowerCollection::const_iterator tower = caloTowers->begin();
       tower != caloTowers->end(); tower++) {
    ETotal += tower->hadEnergy();
    ETotal += tower->emEnergy();
  }

  for (CaloTowerCollection::const_iterator tower = caloTowers->begin();
       tower != caloTowers->end(); tower++) {

    // Raw tower energy without grouping or thresholds

    towerHadEn->Fill(tower->hadEnergy());
    towerEmEn->Fill(tower->emEnergy());
    towerOuterEn->Fill(tower->outerEnergy());

    //    towerHadEt->Fill(tower->hadEt());
    //    towerEmEt->Fill(tower->emEt());
    //    towerOuterEt->Fill(tower->outerEt());

    if ((tower->emEnergy()+tower->hadEnergy()) != 0) {
      emFrac = tower->emEnergy()/(tower->emEnergy()+tower->hadEnergy());
      towerEmFrac->Fill(emFrac);
    } else {
      emFrac = 0.;
    }

    /***
      std::cout << "ETotal = " << ETotal 
		<< " EMF = " << emFrac
		<< " EM = " << tower->emEnergy()
		<< " Tot = " << tower->emEnergy()+tower->hadEnergy()
      		<< " ieta/iphi = " <<  tower->ieta() << " / "  << tower->iphi() 
      		<< std::endl;
    ***/

    if (abs(tower->iphi()) < 100) EMF_Phi->Fill(tower->iphi(), emFrac);
    if (abs(tower->ieta()) < 100) EMF_Eta->Fill(tower->ieta(), emFrac);
    if ( (evt.id().run() == 120020) && (evt.id().event() == 453) ) {
      std::cout << "Bunch Crossing = " << evt.bunchCrossing() 
		<< " Orbit Number = "  << evt.orbitNumber()
		<<  std::endl;

      if (abs(tower->iphi()) < 100) EMF_PhiX->Fill(tower->iphi(), emFrac);
      if (abs(tower->ieta()) < 100) EMF_EtaX->Fill(tower->ieta(), emFrac);
    }
    
    HCALTotalCaloTowerE += tower->hadEnergy();
    ECALTotalCaloTowerE += tower->emEnergy();

    towerE = tower->hadEnergy() + tower->emEnergy();
    if (tower->et() > towerEtCut) caloEtaEt->Fill(tower->eta());
    if (towerE      > towerECut)  caloEta->Fill(tower->eta());
    caloPhi->Fill(tower->phi());

    if (fabs(tower->eta()) < 1.3) {
      HCALTotalCaloTowerE_Eta1 += tower->hadEnergy();
      ECALTotalCaloTowerE_Eta1 += tower->emEnergy();
    }
    if ((fabs(tower->eta()) >= 1.3) && (fabs(tower->eta()) < 2.5)) {
      HCALTotalCaloTowerE_Eta2 += tower->hadEnergy();
      ECALTotalCaloTowerE_Eta2 += tower->emEnergy();
    }
    if (fabs(tower->eta()) > 2.5) {
      HCALTotalCaloTowerE_Eta3 += tower->hadEnergy();
      ECALTotalCaloTowerE_Eta3 += tower->emEnergy();
    }

    /***
    std::cout << "had = "  << tower->hadEnergy()
	      << " em = "  << tower->emEnergy()
	      << " fabs(eta) = " << fabs(tower->eta())
	      << " ieta/iphi = " <<  tower->ieta() << " / "  << tower->iphi() 
	      << std::endl;
    ***/

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

  ECALvHCAL->Fill(HCALTotalCaloTowerE, ECALTotalCaloTowerE);
  ECALvHCALEta1->Fill(HCALTotalCaloTowerE_Eta1, ECALTotalCaloTowerE_Eta1);
  ECALvHCALEta2->Fill(HCALTotalCaloTowerE_Eta2, ECALTotalCaloTowerE_Eta2);
  ECALvHCALEta3->Fill(HCALTotalCaloTowerE_Eta3, ECALTotalCaloTowerE_Eta3);

  std::cout << " Total CaloTower Energy :  "
	    << " ETotal= " << ETotal 
	    << " HCAL= " << HCALTotalCaloTowerE 
	    << " ECAL= " << ECALTotalCaloTowerE
	    << std::endl;

  /***
	    << " HCAL Eta1 = "  << HCALTotalCaloTowerE_Eta1
	    << " ECAL= " << ECALTotalCaloTowerE_Eta1
	    << " HCAL Eta2 = " << HCALTotalCaloTowerE_Eta2
	    << " ECAL= " << ECALTotalCaloTowerE_Eta2
	    << " HCAL Eta3 = " << HCALTotalCaloTowerE_Eta3
	    << " ECAL= " << ECALTotalCaloTowerE_Eta3
	    << std::endl;
  ***/


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
    const edm::TriggerNames & triggerNames = evt.triggerNames(*triggerResults);
    unsigned int n = triggerResults->size();
    for (unsigned int i=0; i!=n; i++) {

      /***
      std::cout << "   Trigger Name = " << triggerNames.triggerName(i)
		<< " Accept = " << triggerResults->accept(i)
		<< std::endl;
      ***/

      if (DEBUG) std::cout <<  triggerNames.triggerName(i) << std::endl;

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

  double HCALTotalE, HBTotalE, HETotalE, HOTotalE, HFTotalE;
  double ECALTotalE, EBTotalE, EETotalE;

  std::vector<CaloTowerPtr>   UsedTowerList;
  std::vector<CaloTower>      TowerUsedInJets;
  std::vector<CaloTower>      TowerNotUsedInJets;

  // *********************
  // *** Hcal recHits
  // *********************

  edm::Handle<HcalSourcePositionData> spd;

  HCALTotalE = HBTotalE = HETotalE = HOTotalE = HFTotalE = 0.;
  try {
    std::vector<edm::Handle<HBHERecHitCollection> > colls;
    evt.getManyByType(colls);
    std::vector<edm::Handle<HBHERecHitCollection> >::iterator i;
    for (i=colls.begin(); i!=colls.end(); i++) {
      for (HBHERecHitCollection::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++) {
        //      std::cout << *j << std::endl;
        if (j->id().subdet() == HcalBarrel) {
	  HBEne->Fill(j->energy()); 
	  HBTime->Fill(j->time()); 
	  HBTvsE->Fill(j->energy(), j->time());

	  if ((j->time()<25.) || (j->time()>75.)) {
	    HBEneOOT->Fill(j->energy()); 
	  }

	  if (j->energy() > HBHEThreshold) {
	    HBEneTh->Fill(j->energy()); 
	    HBTimeTh->Fill(j->time()); 
	    HBTotalE += j->energy();
	    HBocc->Fill(j->id().ieta(),j->id().iphi());
	    hitEta->Fill(j->id().ieta());
	    hitPhi->Fill(j->id().iphi());
	  }
	  if ( (evt.id().run() == 120020) && (evt.id().event() == 453) ) {
	    HBEneX->Fill(j->energy()); 
	    if (j->energy() > HBHEThreshold) HBTimeX->Fill(j->time()); 
	  }
	  if ( (evt.id().run() == 120020) && (evt.id().event() == 457) ) {
	    HBEneY->Fill(j->energy()); 
	    if (j->energy() > HBHEThreshold) HBTimeY->Fill(j->time()); 
	  }
        }
        if (j->id().subdet() == HcalEndcap) {
	  HEEne->Fill(j->energy()); 
	  HETime->Fill(j->time()); 
	  HETvsE->Fill(j->energy(), j->time());

	  if ((j->time()<25.) || (j->time()>75.)) {
	    HEEneOOT->Fill(j->energy()); 
	  }

	  if (j->energy() > HBHEThreshold) {
	    HEEneTh->Fill(j->energy()); 
	    HETimeTh->Fill(j->time()); 
	    HETotalE += j->energy();
	    HEocc->Fill(j->id().ieta(),j->id().iphi());
	    hitEta->Fill(j->id().ieta());
	    hitPhi->Fill(j->id().iphi());
	  }

	  if ( (evt.id().run() == 120020) && (evt.id().event() == 453) ) {
	    HEEneX->Fill(j->energy()); 
	    if (j->energy() > HBHEThreshold) HETimeX->Fill(j->time()); 
	  }
	  if ( (evt.id().run() == 120020) && (evt.id().event() == 457) ) {
	    HEEneY->Fill(j->energy()); 
	    if (j->energy() > HBHEThreshold) HETimeY->Fill(j->time()); 
	  }

	  // Fill +-HE separately
	  if (j->id().ieta()<0) {
	    HEnegEne->Fill(j->energy()); 
	    if (j->energy() > HBHEThreshold) {
	      HEnegTime->Fill(j->time()); 
	    }
	  } else {
	    HEposEne->Fill(j->energy()); 
	    if (j->energy() > HBHEThreshold) {
	      HEposTime->Fill(j->time()); 
	    }
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


  HFM_ETime = 0.;
  HFM_E = 0.;
  HFP_ETime = 0.;
  HFP_E = 0.;
  
  try {
    std::vector<edm::Handle<HFRecHitCollection> > colls;
    evt.getManyByType(colls);
    std::vector<edm::Handle<HFRecHitCollection> >::iterator i;
    for (i=colls.begin(); i!=colls.end(); i++) {
      for (HFRecHitCollection::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++) {

	/****
	float en = j->energy();
	HcalDetId id(j->detid().rawId());
	int ieta = id.ieta();
	int iphi = id.iphi();
	int depth = id.depth();
	*****/

	//  std::cout << *j << std::endl;

        if (j->id().subdet() == HcalForward) {
	  HFEne->Fill(j->energy()); 
	  HFTime->Fill(j->time()); 
	  HFTvsE->Fill(j->energy(), j->time());
	  if (j->energy() > HFThreshold) {
	    HFEneTh->Fill(j->energy()); 
	    HFTimeTh->Fill(j->time()); 
	    HFTotalE += j->energy();
	    HFocc->Fill(j->id().ieta(),j->id().iphi());
	    hitEta->Fill(j->id().ieta());
	    hitPhi->Fill(j->id().iphi());
	  }

	  if (j->id().ieta()<0) {
	    if (j->energy() > HFThreshold) {
	      //	      HFTimeM->Fill(j->time()); 
	      HFEneM->Fill(j->energy()); 
	      HFM_ETime += j->energy()*j->time(); 
	      HFM_E     += j->energy();
	    }
	  } else {
	    if (j->energy() > HFThreshold) {
	      //	      HFTimeP->Fill(j->time()); 
	      HFEneP->Fill(j->energy()); 
	      HFP_ETime += j->energy()*j->time(); 
	      HFP_E     += j->energy();
	    }
	  }

	  // Long and short fibers
	  if (j->id().depth() == 1){
	    HFLEne->Fill(j->energy()); 
	    if (j->energy() > HFThreshold) HFLTime->Fill(j->time());
	  } else {
	    HFSEne->Fill(j->energy()); 
	    if (j->energy() > HFThreshold) HFSTime->Fill(j->time());
	  }
        }
      }
    }
  } catch (...) {
    cout << "No HF RecHits." << endl;
  }

  for (int i=0; i<100; i++) {
     for (int j=0; j<100; j++) {
       HFLvsS->Fill(HFRecHit[i][j][1], HFRecHit[i][j][0]);  
     }
  }

  if (HFP_E > 0.) HFTimeP->Fill(HFP_ETime / HFP_E);
  if (HFM_E > 0.) HFTimeM->Fill(HFM_ETime / HFM_E);

  if ((HFP_E > 0.) && (HFM_E > 0.)) {
    HF_PMM = (HFP_ETime / HFP_E) - (HFM_ETime / HFM_E);
    HFTimePM->Fill(HF_PMM); 
  } else {
    HF_PMM = INVALID;
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
	  HOTvsE->Fill(j->energy(), j->time());
	  if (j->energy() > HOThreshold) {
	    HOEneTh->Fill(j->energy()); 
	    HOTimeTh->Fill(j->time());
	    HOTotalE += j->energy();
	    HOocc->Fill(j->id().ieta(),j->id().iphi());
	  }

	  // Separate SiPMs and HPDs:
	  if (((j->id().iphi()>=59 && j->id().iphi()<=70 && 
		j->id().ieta()>=11 && j->id().ieta()<=15) || 
	       (j->id().iphi()>=47 && j->id().iphi()<=58 && 
		j->id().ieta()>=5 && j->id().ieta()<=10)))
	  {  
	    HOSEne->Fill(j->energy());
	    if (j->energy() > HOThreshold) HOSTime->Fill(j->time());
	  } else if ((j->id().iphi()<59 || j->id().iphi()>70 || 
		      j->id().ieta()<11 || j->id().ieta()>15) && 
		     (j->id().iphi()<47 || j->id().iphi()>58 ||
		      j->id().ieta()<5  || j->id().ieta()>10))
	  {
	    HOHEne->Fill(j->energy());
	    if (j->energy() > HOThreshold) HOHTime->Fill(j->time());
	    // Separate rings -1,-2,0,1,2 in HPDs:
	    if (j->id().ieta()<= -11){
	      HOHrm2Ene->Fill(j->energy());
	      if (j->energy() > HOThreshold) HOHrm2Time->Fill(j->time());
	    } else if (j->id().ieta()>= -10 && j->id().ieta() <= -5) {
	      HOHrm1Ene->Fill(j->energy());
	      if (j->energy() > HOThreshold) HOHrm1Time->Fill(j->time());
	    } else if (j->id().ieta()>= -4 && j->id().ieta() <= 4) {
	      HOHr0Ene->Fill(j->energy());
	      if (j->energy() > HOThreshold) HOHr0Time->Fill(j->time());
	    } else if (j->id().ieta()>= 5 && j->id().ieta() <= 10) {
	      HOHrp1Ene->Fill(j->energy());
	      if (j->energy() > HOThreshold) HOHrp1Time->Fill(j->time());
	    } else if (j->id().ieta()>= 11) {
	      HOHrp2Ene->Fill(j->energy());
	      if (j->energy() > HOThreshold) HOHrp2Time->Fill(j->time());
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

  HCALTotalE = HBTotalE + HETotalE + HFTotalE + HOTotalE;
  ECALTotalE = EBTotalE = EETotalE = 0.;


  try {
    std::vector<edm::Handle<EcalRecHitCollection> > colls;
    evt.getManyByType(colls);
    std::vector<edm::Handle<EcalRecHitCollection> >::iterator i;
    for (i=colls.begin(); i!=colls.end(); i++) {
      for (EcalRecHitCollection::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++) {
	if (j->id().subdetId() == EcalBarrel) {
	  EBEne->Fill(j->energy()); 
	  EBTime->Fill(j->time()); 
	  if (j->energy() > EBEEThreshold) {
	    EBEneTh->Fill(j->energy()); 
	    EBTimeTh->Fill(j->time()); 
	  }
	  if ( (evt.id().run() == 120020) && (evt.id().event() == 453) ) {
	    EBEneX->Fill(j->energy()); 
	    EBTimeX->Fill(j->time()); 
	  }
	  if ( (evt.id().run() == 120020) && (evt.id().event() == 457) ) {
	    EBEneY->Fill(j->energy()); 
	    EBTimeY->Fill(j->time()); 
	  }
	  EBTotalE += j->energy();
	}
	if (j->id().subdetId() == EcalEndcap) {
	  EEEne->Fill(j->energy()); 
	  EETime->Fill(j->time());
	  if (j->energy() > EBEEThreshold) {
	    EEEneTh->Fill(j->energy()); 
	    EETimeTh->Fill(j->time()); 
	  }
	  if ( (evt.id().run() == 120020) && (evt.id().event() == 453) ) {
	    EEEneX->Fill(j->energy()); 
	    EETimeX->Fill(j->time()); 
	  }
	  if ( (evt.id().run() == 120020) && (evt.id().event() == 457 ) ) {
	    EEEneY->Fill(j->energy()); 
	    EETimeY->Fill(j->time()); 
	  }
	  EETotalE += j->energy();
	}
	//	std::cout << *j << std::endl;
	//	std::cout << "EB ID = " << j->id().subdetId() << "/" << EcalBarrel << std::endl;
      }
    }
  } catch (...) {
    cout << "No ECAL RecHits." << endl;
  }

  EBvHB->Fill(HBTotalE, EBTotalE);
  EEvHE->Fill(HETotalE, EETotalE);

  /*****
  try {
    std::vector<edm::Handle<EBRecHitCollection> > colls;
    evt.getManyByType(colls);
    std::vector<edm::Handle<EBRecHitCollection> >::iterator i;

    for (i=colls.begin(); i!=colls.end(); i++) {
      for (EBRecHitCollection::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++) {
	//	if (j->id().subdetId() == EcalBarrel) {
	  EBEne->Fill(j->energy()); 
	  EBTime->Fill(j->time()); 
	  //	  EBTotalE = j->energy();
	  //	}
	  //	std::cout << *j << std::endl;
	  //	std::cout << "EB ID = " << j->id().subdetId() << "/" << EcalBarrel << std::endl;
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
	//	if (j->id().subdetId() == EcalEndcap) {
	  EEEne->Fill(j->energy()); 
	  EETime->Fill(j->time());
	  //	  EETotalE = j->energy();
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
	  //	}
	//	std::cout << *j << std::endl;
      }
    }
  } catch (...) {
    cout << "No EE RecHits." << endl;
  }
  ******/

  ECALTotalE = EBTotalE + EETotalE;

  if ( (EBTotalE > 320000)  && (EBTotalE < 330000) && 
       (HBTotalE > 2700000) && (HBTotalE < 2800000) ) {

    std::cout << ">>> Off Axis! " 
	      << std::endl;
    
  }

  std::cout << " Rechits: Total Energy :  " 
	    << " HCAL= " << HCALTotalE 
	    << " ECAL= " << ECALTotalE
	    << " HB = " << HBTotalE
	    << " EB = " << EBTotalE
	    << std::endl;


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

	    /****
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
	    ****/
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
  VTX  = INVALID;
  nVTX = 0;

  edm::Handle<reco::VertexCollection> vertexCollection;
  evt.getByLabel("offlinePrimaryVertices", vertexCollection);
  const reco::VertexCollection vC = *(vertexCollection.product());

  std::cout << "Reconstructed "<< vC.size() << " vertices" << std::endl ;
  nVTX = vC.size();
  for (reco::VertexCollection::const_iterator vertex=vC.begin(); vertex!=vC.end(); vertex++){

    h_Vx->Fill(vertex->x());
    h_Vy->Fill(vertex->y());
    h_Vz->Fill(vertex->z());
    VTX  = vertex->z();
    //    h_VNTrks->Fill(vertex->tracksSize());

  }

  if ((HF_PMM != INVALID) || (nVTX > 0)) {
    HFvsZ->Fill(HF_PMM,VTX);
  }

  // ********************************
  // *** Tracks
  // ********************************
  edm::Handle<reco::TrackCollection> trackCollection;
  //  evt.getByLabel("ctfWithMaterialTracks", trackCollection);
  evt.getByLabel("generalTracks", trackCollection);

  const reco::TrackCollection tC = *(trackCollection.product());

  std::cout << "ANA: Reconstructed "<< tC.size() << " tracks" << std::endl ;

  h_Trk_NTrk->Fill(tC.size());
  for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++){

    h_Trk_pt->Fill(track->pt());

  }


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
      cout << " CaloJet: Event Type = "   << evtType 
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

    jetHOEne->Fill(ijet->hadEnergyInHO());    
    jetEMFraction->Fill(ijet->emEnergyFraction());    
      
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

}

// ***********************************
// ***********************************
void myJetAna::endJob() {

  for (int i=0; i<4000; i++) {
    if ((nBNC[i]/totBNC) > 0.05) {
      std::cout << "+++ " << i << " " 
		<< (nBNC[i]/totBNC) << " "
		<< nBNC[i]          << " " 
		<< totBNC           << " " 
		<< std::endl;      
    }
  }


}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(myJetAna);

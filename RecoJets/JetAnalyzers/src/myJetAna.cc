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
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"

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

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

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
// #include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

// #include "DataFormats/Scalers/interface/DcsStatus.h"

// include files
#include "CommonTools/RecoAlgos/interface/HBHENoiseFilter.h"
#include "DataFormats/METReco/interface/HcalNoiseSummary.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"  


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
  GenJetAlgorithm( cfg.getParameter<string>( "GenJetAlgorithm" ) ),
  hcalNoiseSummaryTag_(cfg.getParameter<edm::InputTag>("hcalNoiseSummaryTag"))
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
  h_eta    = fs->make<TH1F>( "eta", "Jet #eta", 100, -5, 5 );
  h_phi    = fs->make<TH1F>( "phi", "Jet #phi", 50, -M_PI, M_PI );
  // ---

  hitEtaEt  = fs->make<TH1F>( "hitEtaEt", "RecHit #eta", 90, -45, 45 );
  hitEta    = fs->make<TH1F>( "hitEta", "RecHit #eta", 90, -45, 45 );
  hitPhi    = fs->make<TH1F>( "hitPhi", "RecHit #phi", 73, 0, 73 );

  caloEtaEt  = fs->make<TH1F>( "caloEtaEt", "CaloTower #eta", 100, -4, 4 );
  caloEta    = fs->make<TH1F>( "caloEta", "CaloTower #eta", 100, -4, 4 );
  caloPhi    = fs->make<TH1F>( "caloPhi", "CaloTower #phi", 50, -M_PI, M_PI );

  dijetMass  =  fs->make<TH1F>("dijetMass","DiJet Mass",100,0,100);

  totEneLeadJetEta1 = fs->make<TH1F>("totEneLeadJetEta1","Total Energy Lead Jet Eta1 1",100,0,100);
  totEneLeadJetEta2 = fs->make<TH1F>("totEneLeadJetEta2","Total Energy Lead Jet Eta2 1",150,0,150);
  totEneLeadJetEta3 = fs->make<TH1F>("totEneLeadJetEta3","Total Energy Lead Jet Eta3 1",150,0,150);

  hadEneLeadJetEta1 = fs->make<TH1F>("hadEneLeadJetEta1","Hadronic Energy Lead Jet Eta1 1",50,0,50);
  hadEneLeadJetEta2 = fs->make<TH1F>("hadEneLeadJetEta2","Hadronic Energy Lead Jet Eta2 1",100,0,100);
  hadEneLeadJetEta3 = fs->make<TH1F>("hadEneLeadJetEta3","Hadronic Energy Lead Jet Eta3 1",100,0,100);
  emEneLeadJetEta1  = fs->make<TH1F>("emEneLeadJetEta1","EM Energy Lead Jet Eta1 1",50,0,50);
  emEneLeadJetEta2  = fs->make<TH1F>("emEneLeadJetEta2","EM Energy Lead Jet Eta2 1",100,0,100);
  emEneLeadJetEta3  = fs->make<TH1F>("emEneLeadJetEta3","EM Energy Lead Jet Eta3 1",100,0,100);


  hadFracEta1 = fs->make<TH1F>("hadFracEta11","Hadronic Fraction Eta1 Jet 1",100,0,1);
  hadFracEta2 = fs->make<TH1F>("hadFracEta21","Hadronic Fraction Eta2 Jet 1",100,0,1);
  hadFracEta3 = fs->make<TH1F>("hadFracEta31","Hadronic Fraction Eta3 Jet 1",100,0,1);

  HFSumEt  = fs->make<TH1F>("HFSumEt","HFSumEt",100,0,100);
  HFMET    = fs->make<TH1F>("HFMET",  "HFMET",120,0,120);

  SumEt  = fs->make<TH1F>("SumEt","SumEt",100,0,100);
  MET    = fs->make<TH1F>("MET",  "MET",120,0,120);
  OERMET    = fs->make<TH1F>("OERMET",  "OERMET",120,0,120);
  METSig = fs->make<TH1F>("METSig",  "METSig",100,0,50);
  MEx    = fs->make<TH1F>("MEx",  "MEx",100,-20,20);
  MEy    = fs->make<TH1F>("MEy",  "MEy",100,-20,20);
  METPhi = fs->make<TH1F>("METPhi",  "METPhi",315,0,3.15);
  MET_RBX    = fs->make<TH1F>("MET_RBX",  "MET",100,0,1000);
  MET_HPD    = fs->make<TH1F>("MET_HPD",  "MET",100,0,1000);
  MET_Tower  = fs->make<TH1F>("MET_Tower",  "MET",100,0,1000);

  SiClusters = fs->make<TH1F>("SiClusters",  "SiClusters",150,0,1500);

  h_Vx     = fs->make<TH1F>("Vx",  "Vx",100,-0.5,0.5);
  h_Vy     = fs->make<TH1F>("Vy",  "Vy",100,-0.5,0.5);
  h_Vz     = fs->make<TH1F>("Vz",  "Vz",100,-20,20);
  h_VNTrks = fs->make<TH1F>("VNTrks",  "VNTrks",10,1,100);

  h_Trk_pt   = fs->make<TH1F>("Trk_pt",  "Trk_pt",100,0,20);
  h_Trk_NTrk = fs->make<TH1F>("Trk_NTrk",  "Trk_NTrk",150,0,150);

  hf_sumTowerAllEx = fs->make<TH1F>("sumTowerAllEx","Tower Ex",100,-1000,1000);
  hf_sumTowerAllEy = fs->make<TH1F>("sumTowerAllEy","Tower Ey",100,-1000,1000);

  hf_TowerJetEt   = fs->make<TH1F>("TowerJetEt","Tower/Jet Et 1",50,0,1);

  ETime = fs->make<TH1F>("ETime","Ecal Time",200,-200,200);
  HTime = fs->make<TH1F>("HTime","Hcal Time",200,-200,200);

  towerHadEnHB    = fs->make<TH1F>("towerHadEnHB" ,"HB: Calo Tower HAD Energy",210,-1,20);
  towerHadEnHE    = fs->make<TH1F>("towerHadEnHE" ,"HE: Calo Tower HAD Energy",510,-1,50);
  towerHadEnHF    = fs->make<TH1F>("towerHadEnHF" ,"HF: Calo Tower HAD Energy",510,-1,50);

  towerEmEnHB    = fs->make<TH1F>("towerEmEnHB" ,"HB: Calo Tower EM Energy",210,-1,20);
  towerEmEnHE    = fs->make<TH1F>("towerEmEnHE" ,"HE: Calo Tower EM Energy",510,-1,50);
  towerEmEnHF    = fs->make<TH1F>("towerEmEnHF" ,"HF: Calo Tower EM Energy",510,-1,50);

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

  HBEneOOTTh  = fs->make<TH1F>( "HBEneOOTTh",  "HBEneOOTTh", 200, -5, 10 );
  HEEneOOTTh  = fs->make<TH1F>( "HEEneOOTTh",  "HEEneOOTTh", 200, -5, 10 );
  HFEneOOTTh  = fs->make<TH1F>( "HFEneOOTTh",  "HFEneOOTTh", 200, -5, 10 );
  HOEneOOTTh  = fs->make<TH1F>( "HOEneOOTTh",  "HOEneOOTTh", 200, -5, 10 );

  HBEneOOTTh1  = fs->make<TH1F>( "HBEneOOTTh1",  "HBEneOOT", 200, -5, 10 );
  HEEneOOTTh1  = fs->make<TH1F>( "HEEneOOTTh1",  "HEEneOOT", 200, -5, 10 );
  HFEneOOTTh1  = fs->make<TH1F>( "HFEneOOTTh1",  "HFEneOOT", 200, -5, 10 );
  HOEneOOTTh1  = fs->make<TH1F>( "HOEneOOTTh1",  "HOEneOOT", 200, -5, 10 );

  HBEneTThr     = fs->make<TH1F>( "HBEneTThr",  "HBEneTThr", 105, -5, 100 );
  HEEneTThr     = fs->make<TH1F>( "HEEneTThr",  "HEEneTThr", 105, -5, 100 );
  HFEneTThr     = fs->make<TH1F>( "HFEneTThr",  "HFEneTThr", 105, -5, 100 );


  HBEne     = fs->make<TH1F>( "HBEne",  "HBEne", 205, -5, 200 );
  HBEneTh   = fs->make<TH1F>( "HBEneTh",  "HBEneTh", 205, -5, 200 );
  HBEneTh1  = fs->make<TH1F>( "HBEneTh1",  "HBEneTh1", 205, -5, 200 );
  HBEneX    = fs->make<TH1F>( "HBEneX",  "HBEneX", 200, -5, 10 );
  HBEneY    = fs->make<TH1F>( "HBEneY",  "HBEnedY", 200, -5, 10 );
  HBTime    = fs->make<TH1F>( "HBTime", "HBTime", 200, -100, 100 );
  HBTimeTh  = fs->make<TH1F>( "HBTimeTh", "HBTimeTh", 200, -100, 100 );
  HBTimeTh1  = fs->make<TH1F>( "HBTimeTh1", "HBTimeTh1", 200, -100, 100 );
  HBTimeTh2  = fs->make<TH1F>( "HBTimeTh2", "HBTimeTh2", 200, -100, 100 );
  HBTimeTh3  = fs->make<TH1F>( "HBTimeTh3", "HBTimeTh3", 200, -100, 100 );
  HBTimeThR  = fs->make<TH1F>( "HBTimeThR", "HBTimeThR", 200, -100, 100 );
  HBTimeTh1R  = fs->make<TH1F>( "HBTimeTh1R", "HBTimeTh1R", 200, -100, 100 );
  HBTimeTh2R  = fs->make<TH1F>( "HBTimeTh2R", "HBTimeTh2R", 200, -100, 100 );
  HBTimeTh3R  = fs->make<TH1F>( "HBTimeTh3R", "HBTimeTh3R", 200, -100, 100 );

  HBTimeFlagged    = fs->make<TH1F>( "HBTimeFlagged", "HBTimeFlagged", 200, -100, 100 );
  HBTimeThFlagged    = fs->make<TH1F>( "HBTimeThFlagged", "HBTimeThFlagged", 200, -100, 100 );
  HBTimeTh1Flagged    = fs->make<TH1F>( "HBTimeTh1Flagged", "HBTimeTh1Flagged", 200, -100, 100 );
  HBTimeTh2Flagged    = fs->make<TH1F>( "HBTimeTh2Flagged", "HBTimeTh2Flagged", 200, -100, 100 );

  HBTimeFlagged2    = fs->make<TH1F>( "HBTimeFlagged2", "HBTimeFlagged2", 200, -100, 100 );
  HBTimeThFlagged2    = fs->make<TH1F>( "HBTimeThFlagged2", "HBTimeThFlagged2", 200, -100, 100 );
  HBTimeTh1Flagged2    = fs->make<TH1F>( "HBTimeTh1Flagged2", "HBTimeTh1Flagged2", 200, -100, 100 );
  HBTimeTh2Flagged2    = fs->make<TH1F>( "HBTimeTh2Flagged2", "HBTimeTh2Flagged2", 200, -100, 100 );

  HBTimeX   = fs->make<TH1F>( "HBTimeX", "HBTimeX", 200, -100, 100 );
  HBTimeY   = fs->make<TH1F>( "HBTimeY", "HBTimeY", 200, -100, 100 );
  HEEne     = fs->make<TH1F>( "HEEne",  "HEEne", 205, -5, 200 );
  HEEneTh   = fs->make<TH1F>( "HEEneTh",  "HEEneTh", 205, -5, 200 );
  HEEneTh1  = fs->make<TH1F>( "HEEneTh1",  "HEEneTh1", 205, -5, 200 );
  HEEneX    = fs->make<TH1F>( "HEEneX",  "HEEneX", 200, -5, 10 );
  HEEneY    = fs->make<TH1F>( "HEEneY",  "HEEneY", 200, -5, 10 );
  HEposEne  = fs->make<TH1F>( "HEposEne",  "HEposEne", 200, -5, 10 );
  HEnegEne  = fs->make<TH1F>( "HEnegEne",  "HEnegEne", 200, -5, 10 );
  HETime    = fs->make<TH1F>( "HETime", "HETime", 200, -100, 100 );
  HETimeTh  = fs->make<TH1F>( "HETimeTh", "HETimeTh", 200, -100, 100 );
  HETimeTh1  = fs->make<TH1F>( "HETimeTh1", "HETimeTh1", 200, -100, 100 );
  HETimeTh2  = fs->make<TH1F>( "HETimeTh2", "HETimeTh2", 200, -100, 100 );
  HETimeTh3  = fs->make<TH1F>( "HETimeTh3", "HETimeTh3", 200, -100, 100 );
  HETimeThR  = fs->make<TH1F>( "HETimeThR", "HETimeThR", 200, -100, 100 );
  HETimeTh1R  = fs->make<TH1F>( "HETimeTh1R", "HETimeTh1R", 200, -100, 100 );
  HETimeTh2R  = fs->make<TH1F>( "HETimeTh2R", "HETimeTh2R", 200, -100, 100 );
  HETimeTh3R  = fs->make<TH1F>( "HETimeTh3R", "HETimeTh3R", 200, -100, 100 );

  HETimeFlagged    = fs->make<TH1F>( "HETimeFlagged", "HETimeFlagged", 200, -100, 100 );
  HETimeThFlagged    = fs->make<TH1F>( "HETimeThFlagged", "HETimeThFlagged", 200, -100, 100 );
  HETimeTh1Flagged    = fs->make<TH1F>( "HETimeTh1Flagged", "HETimeTh1Flagged", 200, -100, 100 );
  HETimeTh2Flagged    = fs->make<TH1F>( "HETimeTh2Flagged", "HETimeTh2Flagged", 200, -100, 100 );

  HETimeFlagged2    = fs->make<TH1F>( "HETimeFlagged2", "HETimeFlagged2", 200, -100, 100 );
  HETimeThFlagged2    = fs->make<TH1F>( "HETimeThFlagged2", "HETimeThFlagged2", 200, -100, 100 );
  HETimeTh1Flagged2    = fs->make<TH1F>( "HETimeTh1Flagged2", "HETimeTh1Flagged2", 200, -100, 100 );
  HETimeTh2Flagged2    = fs->make<TH1F>( "HETimeTh2Flagged2", "HETimeTh2Flagged2", 200, -100, 100 );

  HETimeX   = fs->make<TH1F>( "HETimeX", "HETimeX", 200, -100, 100 );
  HETimeY   = fs->make<TH1F>( "HETimeY", "HETimeY", 200, -100, 100 );
  HEposTime = fs->make<TH1F>( "HEposTime",  "HEposTime", 200, -100, 100 );
  HEnegTime = fs->make<TH1F>( "HEnegTime",  "HEnegTime", 200, -100, 100 );
  HOEne     = fs->make<TH1F>( "HOEne",  "HOEne", 200, -5, 10 );
  HOEneTh   = fs->make<TH1F>( "HOEneTh",  "HOEneTh", 200, -5, 10 );
  HOEneTh1  = fs->make<TH1F>( "HOEneTh1",  "HOEneTh1", 200, -5, 10 );
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

  HBTvsE    = fs->make<TH2F>( "HBTvsE", "HBTvsE",305, -5, 300, 100, -100, 100);
  HETvsE    = fs->make<TH2F>( "HETvsE", "HETvsE",305, -5, 300, 100, -100, 100);

  HFTvsE            = fs->make<TH2F>( "HFTvsE", "HFTvsE",305, -5, 300, 100, -100, 100);
  HFTvsEFlagged     = fs->make<TH2F>( "HFTvsEFlagged", "HFTvsEFlagged",305, -5, 300, 100, -100, 100);  
  HFTvsEFlagged2    = fs->make<TH2F>( "HFTvsEFlagged2", "HFTvsEFlagged2",305, -5, 300, 100, -100, 100);

  HFTvsEThr    = fs->make<TH2F>( "HFTvsEThr", "HFTvsEThr",305, -5, 300, 100, -100, 100);
  HFTvsEFlaggedThr    = fs->make<TH2F>( "HFTvsEFlaggedThr", "HFTvsEFlaggedThr",305, -5, 300, 100, -100, 100);  
  HFTvsEFlagged2Thr    = fs->make<TH2F>( "HFTvsEFlagged2Thr", "HFTvsEFlagged2Thr",305, -5, 300, 100, -100, 100);

  HOTvsE    = fs->make<TH2F>( "HOTvsE", "HOTvsE",305, -5, 300, 100, -100, 100);

  HFvsZ    = fs->make<TH2F>( "HFvsZ", "HFvsZ",100,-50,50,100,-50,50);



  HOocc    = fs->make<TH2F>( "HOocc", "HOocc",85,-42.5,42.5,70,0.5,70.5);
  HBocc    = fs->make<TH2F>( "HBocc", "HBocc",85,-42.5,42.5,70,0.5,70.5);
  HEocc    = fs->make<TH2F>( "HEocc", "HEocc",85,-42.5,42.5,70,0.5,70.5);
  HFocc    = fs->make<TH2F>( "HFocc", "HFocc",85,-42.5,42.5,70,0.5,70.5);
  HFoccTime    = fs->make<TH2F>( "HFoccTime", "HFoccTime",85,-42.5,42.5,70,0.5,70.5);
  HFoccFlagged    = fs->make<TH2F>( "HFoccFlagged", "HFoccFlagged",85,-42.5,42.5,70,0.5,70.5);
  HFoccFlagged2    = fs->make<TH2F>( "HFoccFlagged2", "HFoccFlagged2",85,-42.5,42.5,70,0.5,70.5);

  HFEtaPhiNFlagged    = fs->make<TH2F>( "HFEtaPhiNFlagged", "HFEtaPhiNFlagged",85,-42.5,42.5,70,0.5,70.5);

  //  HFEtaFlagged    = fs->make<TProfile>( "HFEtaFlagged", "HFEtaFlagged",85,-42.5,42.5,0, 10000);
  HFEtaFlagged     = fs->make<TH1F>( "HFEtaFlagged", "HFEtaFlagged",85,-42.5,42.5);
  HFEtaFlaggedL    = fs->make<TH1F>( "HFEtaFlaggedL", "HFEtaFlaggedL",85,-42.5,42.5);
  HFEtaFlaggedLN    = fs->make<TH1F>( "HFEtaFlaggedLN", "HFEtaFlaggedLN",85,-42.5,42.5);
  HFEtaFlaggedS    = fs->make<TH1F>( "HFEtaFlaggedS", "HFEtaFlaggedS",85,-42.5,42.5);
  HFEtaFlaggedSN    = fs->make<TH1F>( "HFEtaFlaggedSN", "HFEtaFlaggedSN",85,-42.5,42.5);

  HFEtaNFlagged    = fs->make<TProfile>( "HFEtaNFlagged", "HFEtaNFlagged",85,-42.5,42.5,0, 10000);

  HOoccOOT    = fs->make<TH2F>( "HOoccOOT", "HOoccOOT",85,-42.5,42.5,70,0.5,70.5);
  HBoccOOT    = fs->make<TH2F>( "HBoccOOT", "HBoccOOT",85,-42.5,42.5,70,0.5,70.5);
  HEoccOOT    = fs->make<TH2F>( "HEoccOOT", "HEoccOOT",85,-42.5,42.5,70,0.5,70.5);
  HFoccOOT    = fs->make<TH2F>( "HFoccOOT", "HFoccOOT",85,-42.5,42.5,70,0.5,70.5);

  HFEnePMT0     = fs->make<TH1F>( "HFEnePMT0",  "HFEnePMT0", 210, -10, 200 );
  HFEnePMT1     = fs->make<TH1F>( "HFEnePMT1",  "HFEnePMT1", 210, -10, 200 );
  HFEnePMT2     = fs->make<TH1F>( "HFEnePMT2",  "HFEnePMT2", 210, -10, 200 );
  HFTimePMT0    = fs->make<TH1F>( "HFTimePMT0", "HFTimePMT0", 200, -100, 100 );
  HFTimePMT1    = fs->make<TH1F>( "HFTimePMT1", "HFTimePMT1", 200, -100, 100 );
  HFTimePMT2    = fs->make<TH1F>( "HFTimePMT2", "HFTimePMT2", 200, -100, 100 );

  HFEne     = fs->make<TH1F>( "HFEne",  "HFEne", 210, -10, 200 );
  HFEneFlagged     = fs->make<TH1F>( "HFEneFlagged",  "HFEneFlagged", 210, -10, 200 );
  HFEneFlagged2     = fs->make<TH1F>( "HFEneFlagged2",  "HFEneFlagged2", 210, -10, 200 );
  HFEneTh   = fs->make<TH1F>( "HFEneTh",  "HFEneTh", 210, -10, 200 );
  HFEneTh1  = fs->make<TH1F>( "HFEneTh1",  "HFEneTh1", 210, -10, 200 );
  HFEneP    = fs->make<TH1F>( "HFEneP",  "HFEneP", 200, -5, 10 );
  HFEneM    = fs->make<TH1F>( "HFEneM",  "HFEneM", 200, -5, 10 );
  HFTime    = fs->make<TH1F>( "HFTime", "HFTime", 200, -100, 100 );
  PMTHits   = fs->make<TH1F>( "PMTHits", "PMTHits", 10, 0, 10 );
  HFTimeFlagged  = fs->make<TH1F>( "HFTimeFlagged", "HFTimeFlagged", 200, -100, 100 );

  HFTimeFlagged2  = fs->make<TH1F>( "HFTimeFlagged2", "HFTimeFlagged2", 200, -100, 100 );
  HFTimeThFlagged2  = fs->make<TH1F>( "HFTimeThFlagged2", "HFTimeThFlagged2", 200, -100, 100 );
  HFTimeTh1Flagged2  = fs->make<TH1F>( "HFTimeTh1Flagged2", "HFTimeTh1Flagged2", 200, -100, 100 );
  HFTimeTh2Flagged2  = fs->make<TH1F>( "HFTimeTh2Flagged2", "HFTimeTh2Flagged2", 200, -100, 100 );
  HFTimeTh3Flagged2  = fs->make<TH1F>( "HFTimeTh3Flagged2", "HFTimeTh3Flagged2", 200, -100, 100 );

  HFTimeFlagged3  = fs->make<TH1F>( "HFTimeFlagged3", "HFTimeFlagged3", 200, -100, 100 );
  HFTimeThFlagged3  = fs->make<TH1F>( "HFTimeThFlagged3", "HFTimeThFlagged3", 200, -100, 100 );
  HFTimeTh1Flagged3  = fs->make<TH1F>( "HFTimeTh1Flagged3", "HFTimeTh1Flagged3", 200, -100, 100 );
  HFTimeTh2Flagged3  = fs->make<TH1F>( "HFTimeTh2Flagged3", "HFTimeTh2Flagged3", 200, -100, 100 );
  HFTimeTh3Flagged3  = fs->make<TH1F>( "HFTimeTh3Flagged3", "HFTimeTh3Flagged3", 200, -100, 100 );

  HFTimeThFlagged  = fs->make<TH1F>( "HFTimeThFlagged", "HFTimeThFlagged", 200, -100, 100 );
  HFTimeTh2Flagged  = fs->make<TH1F>( "HFTimeTh2Flagged", "HFTimeTh2Flagged", 200, -100, 100 );
  HFTimeTh3Flagged  = fs->make<TH1F>( "HFTimeTh3Flagged", "HFTimeTh3Flagged", 200, -100, 100 );

  HFTimeThFlaggedR  = fs->make<TH1F>( "HFTimeThFlaggedR", "HFTimeThFlaggedR", 200, -100, 100 );
  HFTimeThFlaggedR1  = fs->make<TH1F>( "HFTimeThFlaggedR1", "HFTimeThFlaggedR1", 200, -100, 100 );
  HFTimeThFlaggedR2  = fs->make<TH1F>( "HFTimeThFlaggedR2", "HFTimeThFlaggedR2", 200, -100, 100 );
  HFTimeThFlaggedR3  = fs->make<TH1F>( "HFTimeThFlaggedR3", "HFTimeThFlaggedR3", 200, -100, 100 );
  HFTimeThFlaggedR4  = fs->make<TH1F>( "HFTimeThFlaggedR4", "HFTimeThFlaggedR4", 200, -100, 100 );
  HFTimeThFlaggedRM  = fs->make<TH1F>( "HFTimeThFlaggedRM", "HFTimeThFlaggedRM", 200, -100, 100 );
  TrkMultFlagged0  = fs->make<TH1F>( "TrkMultFlagged0", "TrkMultFlagged0", 100, 0, 100 );
  TrkMultFlagged1  = fs->make<TH1F>( "TrkMultFlagged1", "TrkMultFlagged1", 100, 0, 100 );
  TrkMultFlagged2  = fs->make<TH1F>( "TrkMultFlagged2", "TrkMultFlagged2", 100, 0, 100 );
  TrkMultFlagged3  = fs->make<TH1F>( "TrkMultFlagged3", "TrkMultFlagged3", 100, 0, 100 );
  TrkMultFlagged4  = fs->make<TH1F>( "TrkMultFlagged4", "TrkMultFlagged4", 100, 0, 100 );
  TrkMultFlaggedM  = fs->make<TH1F>( "TrkMultFlaggedM", "TrkMultFlaggedM", 100, 0, 100 );
  HFTimeTh  = fs->make<TH1F>( "HFTimeTh", "HFTimeTh", 200, -100, 100 );
  HFTimeTh1  = fs->make<TH1F>( "HFTimeTh1", "HFTimeTh1", 200, -100, 100 );
  HFTimeTh2  = fs->make<TH1F>( "HFTimeTh2", "HFTimeTh2", 200, -100, 100 );
  HFTimeTh3  = fs->make<TH1F>( "HFTimeTh3", "HFTimeTh3", 200, -100, 100 );
  HFTimeThR  = fs->make<TH1F>( "HFTimeThR", "HFTimeThR", 200, -100, 100 );
  HFTimeTh1R  = fs->make<TH1F>( "HFTimeTh1R", "HFTimeTh1R", 200, -100, 100 );
  HFTimeTh2R  = fs->make<TH1F>( "HFTimeTh2R", "HFTimeTh2R", 200, -100, 100 );
  HFTimeTh3R  = fs->make<TH1F>( "HFTimeTh3R", "HFTimeTh3R", 200, -100, 100 );
  HFTimeP   = fs->make<TH1F>( "HFTimeP", "HFTimeP", 100, -100, 50 );
  HFTimeM   = fs->make<TH1F>( "HFTimeM", "HFTimeM", 100, -100, 50 );
  HFTimePMa = fs->make<TH1F>( "HFTimePMa", "HFTimePMa", 100, -100, 100 );
  HFTimePM  = fs->make<TH1F>( "HFTimePM", "HFTimePM", 100, -100, 100 );

  // Histos for separating HF long/short fibers:
  HFLEneAll  = fs->make<TH1F>( "HFLEneAll",  "HFLEneAll", 210, -10, 200 );
  HFLEneAllF = fs->make<TH1F>( "HFLEneAllF",  "HFLEneAllF", 210, -10, 200 );
  HFSEneAll  = fs->make<TH1F>( "HFSEneAll",  "HFSEneAll", 210, -10, 200 );
  HFSEneAllF = fs->make<TH1F>( "HFSEneAllF",  "HFSEneAllF", 210, -10, 200 );
  HFLEne     = fs->make<TH1F>( "HFLEne",  "HFLEne", 200, -5, 10 );
  HFLTime    = fs->make<TH1F>( "HFLTime", "HFLTime", 200, -100, 100 );
  HFSEne     = fs->make<TH1F>( "HFSEne",  "HFSEne", 200, -5, 10 );
  HFSTime    = fs->make<TH1F>( "HFSTime", "HFSTime", 200, -100, 100 );
  HFLSRatio  = fs->make<TH1F>( "HFLSRatio",  "HFLSRatio", 220, -1.1, 1.1 );

  HFOERatio  = fs->make<TH1F>( "HFOERatio",  "HFOERatio", 2200, -1.1, 1.1 );

  HFLvsS     = fs->make<TH2F>( "HFLvsS", "HFLvsS",220,-20,200,220,-20,200);
  HFLEneNoS  = fs->make<TH1F>( "HFLEneNoS",  "HFLEneNoS", 205, -5, 200 );
  HFSEneNoL  = fs->make<TH1F>( "HFSEneNoL",  "HFSEneNoL", 205, -5, 200 );
  HFLEneNoSFlagged  = fs->make<TH1F>( "HFLEneNoSFlagged",  "HFLEneNoSFlagged", 205, -5, 200 );
  HFSEneNoLFlagged  = fs->make<TH1F>( "HFSEneNoLFlagged",  "HFSEneNoLFlagged", 205, -5, 200 );
  HFLEneNoSFlaggedN  = fs->make<TH1F>( "HFLEneNoSFlaggedN",  "HFLEneNoSFlaggedN", 205, -5, 200 );
  HFSEneNoLFlaggedN  = fs->make<TH1F>( "HFSEneNoLFlaggedN",  "HFSEneNoLFlaggedN", 205, -5, 200 );


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

  h_nTowersCal = fs->make<TH1F>( "nTowersCal",  "N Towers in Jet", 100, 0, 50 );
  h_EMFracCal  = fs->make<TH1F>( "EMFracCal",  "EM Fraction in Jet", 100, -1.1, 1.1 );
  h_ptCal      = fs->make<TH1F>( "ptCal",  "p_{T} of CalJet", 100, 0, 50 );
  h_etaCal     = fs->make<TH1F>( "etaCal", "#eta of  CalJet", 100, -4, 4 );
  h_phiCal     = fs->make<TH1F>( "phiCal", "#phi of  CalJet", 50, -M_PI, M_PI );

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
  h_jet1Eta       = fs->make<TH1F>( "jet1Eta", "Jet1 Eta", 50, -5, 5 );
  h_jet2Eta       = fs->make<TH1F>( "jet2Eta", "Jet2 Eta", 50, -5, 5 );
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

  HFTimeVsiEtaP  = fs->make<TProfile>("HFTimeVsiEtaP","HFTimeVsiEtaP", 13, 28.5, 41.5, -100, 100);
  HFTimeVsiEtaM  = fs->make<TProfile>("HFTimeVsiEtaM","HFTimeVsiEtaM", 13, -41.5, -28.5, -100, 100);

  HFTimeVsiEtaP5  = fs->make<TProfile>("HFTimeVsiEtaP5","HFTimeVsiEtaP5", 13, 28.5, 41.5, -100, 100);
  HFTimeVsiEtaM5  = fs->make<TProfile>("HFTimeVsiEtaM5","HFTimeVsiEtaM5", 13, -41.5, -28.5, -100, 100);

  HFTimeVsiEtaP20  = fs->make<TProfile>("HFTimeVsiEtaP20","HFTimeVsiEtaP20", 13, 28.5, 41.5, -100, 100);
  HFTimeVsiEtaM20  = fs->make<TProfile>("HFTimeVsiEtaM20","HFTimeVsiEtaM20", 13, -41.5, -28.5, -100, 100);

  NPass          = fs->make<TH1F>( "NPass", "NPass", 3, -1, 1 );
  NTotal         = fs->make<TH1F>( "NTotal", "NTotal", 3, -1, 1 );
  NTime         = fs->make<TH1F>( "NTime", "NTime", 10, 0, 10 );


  HFRecHitEne   = fs->make<TH1F>( "HFRecHitEne", "HFRecHitEne", 300, 0, 3000 );
  HFRecHitEneClean   = fs->make<TH1F>( "HFRecHitEneClean", "HFRecHitEneClean", 300, 0, 3000 );
  HFRecHitTime  = fs->make<TH1F>( "HFRecHitTime", "HFRecHitTime", 120, -60, 60 );


  HFLongShortPhi  = fs->make<TH1F>( "HFLongShortPhi", "HFLongShortPhi", 73, 0, 73 );
  HFLongShortEta  = fs->make<TH1F>( "HFLongShortEta", "HFLongShortEta", 90, -45, 45 );
  HFLongShortEne  = fs->make<TH1F>( "HFLongShortEne", "HFLongShortEne", 300, 0, 3000 );
  HFLongShortTime = fs->make<TH1F>( "HFLongShortTime", "HFLongShortTime", 120, -60, 60 );
  HFLongShortNHits = fs->make<TH1F>( "HFLongShortNHits", "HFLongShortNHits", 30, 0, 30 );

  HFDigiTimePhi  = fs->make<TH1F>( "HFDigiTimePhi", "HFDigiTimePhi", 73, 0, 73 );
  HFDigiTimeEta  = fs->make<TH1F>( "HFDigiTimeEta", "HFDigiTimeEta", 90, -45, 45 );
  HFDigiTimeEne  = fs->make<TH1F>( "HFDigiTimeEne", "HFDigiTimeEne", 300, 0, 3000 );
  HFDigiTimeTime = fs->make<TH1F>( "HFDigiTimeTime", "HFDigiTimeTime", 120, -60, 60 );
  HFDigiTimeNHits = fs->make<TH1F>( "HFDigiTimeNHits", "HFDigiTimeNHits", 30, 0, 30 );


  totBNC = 0;
  for (int i=0; i<4000; i++)  nBNC[i] = 0;

}

// ************************
// ************************
void myJetAna::analyze( const edm::Event& evt, const edm::EventSetup& es ) {

  using namespace edm;

  bool Pass, Pass_HFTime, Pass_DiJet, Pass_BunchCrossing, Pass_Vertex;

  int EtaOk10, EtaOk13, EtaOk40;

  double LeadMass;

  double HFRecHit[100][100][2];
  int HFRecHitFlag[100][100][2];

  double towerEtCut, towerECut, towerE;

  towerEtCut = 1.0;
  towerECut  = 1.0;

  unsigned int StableRun = 123732;

  double HBHEThreshold = 2.0;
  double HFThreshold   = 2.0;
  double HOThreshold   = 2.0;
  double EBEEThreshold = 2.0;

  double HBHEThreshold1 = 4.0;
  double HFThreshold1   = 4.0;
  double HOThreshold1   = 4.0;
  //double EBEEThreshold1 = 4.0;

  double HBHEThreshold2 = 10.0;
  double HFThreshold2   = 10.0;
  //double HOThreshold2   = 10.0;
  //double EBEEThreshold2 = 10.0;

  double HBHEThreshold3 = 40.0;
  double HFThreshold3   = 40.0;
  //double HOThreshold3   = 40.0;
  //double EBEEThreshold3 = 40.0;

  float minJetPt = 20.;
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

  /***
  std::cout << ">>>> ANA: Run = "    << evt.id().run() 
	    << " Event = " << evt.id().event()
	    << " Bunch Crossing = " << evt.bunchCrossing() 
	    << " Orbit Number = "   << evt.orbitNumber()
	    << " Luminosity Block = "  << evt.luminosityBlock()
	    <<  std::endl;
  ***/

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
    //    edm::TriggerNames triggerNames;    // TriggerNames class
    //    triggerNames.init(*triggerResults);
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

      //      if (DEBUG) std::cout <<  triggerNames.triggerName(i) << std::endl;

      //      if ( (triggerNames.triggerName(i) == "HLT_ZeroBias")  || 
      //	   (triggerNames.triggerName(i) == "HLT_MinBias")   || 
      //	   (triggerNames.triggerName(i) == "HLT_MinBiasHcal") )  {

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


  
  /***
  Handle<L1GlobalTriggerReadoutRecord> gtRecord;
  evt.getByLabel("gtDigis",gtRecord);
  const TechnicalTriggerWord tWord = gtRecord->technicalTriggerWord();

  ***/


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
      HFRecHitFlag[i][j][0]  = 0;
      HFRecHitFlag[i][j][1]  = 0;
    }
  }


  int nTime = 0;
  int NHFLongShortHits;
  int NHFDigiTimeHits;
  NHFLongShortHits = 0;
  NHFDigiTimeHits = 0;

  //  edm::Handle<reco::VertexCollection> vertexCollection;

  try {
    std::vector<edm::Handle<HFRecHitCollection> > colls;
    evt.getManyByType(colls);

    std::vector<edm::Handle<HFRecHitCollection> >::iterator i;
    for (i=colls.begin(); i!=colls.end(); i++) {
      
      for (HFRecHitCollection::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++) {
        if (j->id().subdet() == HcalForward) {

	  HFRecHitEne->Fill(j->energy());
	  if ( (j->flagField(HcalCaloFlagLabels::HFLongShort) == 0) && 
	       (j->flagField(HcalCaloFlagLabels::HFDigiTime)  == 0) ) {
	    HFRecHitEneClean->Fill(j->energy());
	  }
 
	  HFRecHitTime->Fill(j->time());

          int myFlag;
          myFlag= j->flagField(HcalCaloFlagLabels::HFLongShort);
          if (myFlag==1) {
	    NHFLongShortHits++;
	    HFLongShortPhi->Fill(j->id().iphi());
	    HFLongShortEta->Fill(j->id().ieta());
	    HFLongShortEne->Fill(j->energy());
	    HFLongShortTime->Fill(j->time());
	  }

          myFlag= j->flagField(HcalCaloFlagLabels::HFDigiTime);
          if (myFlag==1) {
	    NHFDigiTimeHits++;
	    HFDigiTimePhi->Fill(j->id().iphi());
	    HFDigiTimeEta->Fill(j->id().ieta());
	    HFDigiTimeEne->Fill(j->energy());
	    HFDigiTimeTime->Fill(j->time());
	  }

	  
	  float en = j->energy();
	  float time = j->time();
	  if ((en > 20.) && (time>20.)) {
	    HFoccTime->Fill(j->id().ieta(),j->id().iphi());
	    nTime++;
	  }
	  HcalDetId id(j->detid().rawId());
	  int ieta = id.ieta();
	  int iphi = id.iphi();
	  int depth = id.depth();


	  // Long:  depth = 1
	  // Short: depth = 2
	  HFRecHit[ieta+41][iphi][depth-1] = en;
	  HFRecHitFlag[ieta+41][iphi][depth-1] = j->flagField(0);

	  /****
	  std::cout << "RecHit Flag = " 
		    << j->flagField(0)a
		    << std::endl;
	  ***/

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
      break;
    }
  } catch (...) {
    cout << "No HF RecHits." << endl;
  }

  cout << "N HF Hits" << NHFLongShortHits << " " << NHFDigiTimeHits << endl;
  HFLongShortNHits->Fill(NHFLongShortHits);
  HFDigiTimeNHits->Fill(NHFDigiTimeHits);

  NTime->Fill(nTime);

  double OER = 0, OddEne, EvenEne;
  int nOdd, nEven;

  for (int iphi=0; iphi<100; iphi++) {
    OddEne = EvenEne = 0.;
    nOdd  = 0;
    nEven = 0;
    for (int ieta=0; ieta<100; ieta++) {
      if (HFRecHit[ieta][iphi][0] > 1.0) {
	if (ieta%2 == 0) {
	  EvenEne += HFRecHit[ieta][iphi][0]; 
	  nEven++;
	} else {
	  OddEne  += HFRecHit[ieta][iphi][0];
	  nOdd++;
	}
      }
      if (HFRecHit[ieta][iphi][1] > 1.0) {
	if (ieta%2 == 0) {
	  EvenEne += HFRecHit[ieta][iphi][1]; 
	  nEven++;
	} else {
	  OddEne  += HFRecHit[ieta][iphi][1]; 
	  nOdd++;
	}
      }
    }
    if (((OddEne + EvenEne) > 10.) && (nOdd > 1) && (nEven > 1)) {
      OER = (OddEne - EvenEne) / (OddEne + EvenEne);
      HFOERatio->Fill(OER);
    }
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
  // double dphi;
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
    //dphi = deltaPhi(p4tmp[0].phi(), p4tmp[1].phi());
    Pass_DiJet = true;
  } else {
    // dphi = INVALID;
    Pass_DiJet = false;
  }
      

  // **************************
  // ***  Pass Vertex
  // **************************
  double VTX;
  int nVTX;

  edm::Handle<reco::VertexCollection> vertexCollection;
  evt.getByLabel("offlinePrimaryVertices", vertexCollection);
  const reco::VertexCollection vC = *(vertexCollection.product());

  //  std::cout << "Reconstructed "<< vC.size() << " vertices" << std::endl ;

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


  // ********************************
  // *** Pixel Clusters
  // ********************************
  edm::Handle< edmNew::DetSetVector<SiPixelCluster> > hClusterColl;
  evt.getByLabel("siPixelClusters", hClusterColl);
  const edmNew::DetSetVector<SiPixelCluster> clustColl = *(hClusterColl.product());

  edm::Handle<reco::TrackCollection> trackCollection;
  evt.getByLabel("generalTracks", trackCollection);
  const reco::TrackCollection tC = *(trackCollection.product());


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
    Pass = true;

  } else {
    if ( (Pass_BunchCrossing) && 
	 (Pass_HFTime)        &&
	 (Pass_Vertex) ) {
      Pass = true;
    } else {
      Pass = false;
    }
  }

  /***
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
  ***/

  NTotal->Fill(0);
  
  Pass = false;
  if ((tC.size() > 100) && (clustColl.size() > 1000)) Pass = true;
  Pass = true;

  /****
  if (Pass_HFTime) {
    Pass = true;
  } else {
    Pass = false;
  }
  ****/

  // **************************
  // *** Noise Summary Object
  // **************************

  edm::Handle<HcalNoiseSummary> summary_h;
  evt.getByLabel(hcalNoiseSummaryTag_, summary_h);
  if(!summary_h.isValid()) {
    throw edm::Exception(edm::errors::ProductNotFound) << " could not find HcalNoiseSummary.\n";
    //    return true;
  }

  const HcalNoiseSummary summary = *summary_h;

  bool Pass_NoiseSummary;
  Pass_NoiseSummary = true;
  if(summary.minE2Over10TS()<0.7) {
    Pass_NoiseSummary = false;
  }
  if(summary.maxE2Over10TS()>0.96) {
    Pass_NoiseSummary = false;
  }
  if(summary.maxHPDHits()>=17) {
    Pass_NoiseSummary = false;
  }
  if(summary.maxRBXHits()>=999) {
    Pass_NoiseSummary = false;
  }
  if(summary.maxHPDNoOtherHits()>=10) {
    Pass_NoiseSummary = false;
  }
  if(summary.maxZeros()>=10) {
    Pass_NoiseSummary = false;
  }
  if(summary.min25GeVHitTime()<-9999.0) {
    Pass_NoiseSummary = false;
  }
  if(summary.max25GeVHitTime()>9999.0) {
    Pass_NoiseSummary = false;
  }
  if(summary.minRBXEMF()<0.01) {
  }

  if (Pass_NoiseSummary) {
    Pass = false;
  } else {
    Pass = true;
  }


  Pass = true;
  if (Pass) {

    NPass->Fill(0);

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
    if (abs(tower->ieta()) < 100) EMF_Eta->Fill(tower->ieta(), emFrac);

    if (abs(tower->ieta()) < 15) {
      towerHadEnHB->Fill(tower->hadEnergy());
      towerEmEnHB->Fill(tower->emEnergy());
    }
    if ( (abs(tower->ieta()) > 17) && ((abs(tower->ieta()) < 30)) ){
      towerHadEnHE->Fill(tower->hadEnergy());
      towerEmEnHE->Fill(tower->emEnergy());
    }
    if (abs(tower->ieta()) > 29) {
      towerHadEnHF->Fill(tower->hadEnergy());
      towerEmEnHF->Fill(tower->emEnergy());
    }

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

  /***
  std::cout << " Total CaloTower Energy :  "
	    << " ETotal= " << ETotal 
	    << " HCAL= " << HCALTotalCaloTowerE 
	    << " ECAL= " << ECALTotalCaloTowerE
	    << std::endl;
  ***/

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
    //    edm::TriggerNames triggerNames;    // TriggerNames class
    //    triggerNames.init(*triggerResults);
    unsigned int n = triggerResults->size();
    for (unsigned int i=0; i!=n; i++) {

      /***
      std::cout << "   Trigger Name = " << triggerNames.triggerName(i)
		<< " Accept = " << triggerResults->accept(i)
		<< std::endl;
      ***/

      //      if (DEBUG) std::cout <<  triggerNames.triggerName(i) << std::endl;

      /***
      if ( triggerNames.triggerName(i) == "HLT_Jet30" ) {
        JetLoPass =  triggerResults->accept(i);
        if (DEBUG) std::cout << "Found  HLT_Jet30 " 
			     << JetLoPass
			     << std::endl;
      }
      ***/

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
      h_jet1Eta->Fill( cal->eta() );
      if (JetLoPass != 0) h_jet1PtHLT->Fill( cal->pt() );
      p4tmp[0] = cal->p4();
      if ( fabs(cal->eta()) < 1.0) EtaOk10++;
      if ( fabs(cal->eta()) < 1.3) EtaOk13++;
      if ( fabs(cal->eta()) < 4.0) EtaOk40++;            
    }
    if (allJetInd == 2) {
      h_jet2Pt->Fill( cal->pt() );
      h_jet2Eta->Fill( cal->eta() );
      p4tmp[1] = cal->p4();
      if ( fabs(cal->eta()) < 1.0) EtaOk10++;
      if ( fabs(cal->eta()) < 1.3) EtaOk13++;
      if ( fabs(cal->eta()) < 4.0) EtaOk40++;
    }

    if ( cal->pt() > minJetPt) {
      const std::vector<CaloTowerPtr> jetCaloRefs = cal->getCaloConstituents();
      int nConstituents = jetCaloRefs.size();
      h_nTowersCal->Fill(nConstituents);
      h_EMFracCal->Fill(cal->emEnergyFraction());    
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
      hadEneLeadJetEta1->Fill(ijet->hadEnergyInHB()); 
      emEneLeadJetEta1->Fill(ijet->emEnergyInEB());       
      if (ijet->pt() > minJetPt10) hadFracEta1->Fill(had);
    }

    // *** EndCap
    if ((fabs(ijet->eta()) > 1.3) && (fabs(ijet->eta()) < 3.) ) {
      totEneLeadJetEta2->Fill(hadEne+emEne); 
      hadEneLeadJetEta2->Fill(ijet->hadEnergyInHE()); 
      emEneLeadJetEta2->Fill(ijet->emEnergyInEE());       
      if (ijet->pt() > minJetPt10) hadFracEta2->Fill(had);
    }

    // *** Forward
    if (fabs(ijet->eta()) > 3.) {
      totEneLeadJetEta3->Fill(hadEne+emEne); 
      hadEneLeadJetEta3->Fill(hadEne); 
      emEneLeadJetEta3->Fill(emEne); 
      if (ijet->pt() > minJetPt10) hadFracEta3->Fill(had);
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

  // double HCALTotalE;
  double HBTotalE, HETotalE, HOTotalE, HFTotalE;
  // double ECALTotalE;
  double EBTotalE, EETotalE;

  std::vector<CaloTowerPtr>   UsedTowerList;
  std::vector<CaloTower>      TowerUsedInJets;
  std::vector<CaloTower>      TowerNotUsedInJets;

  // *********************
  // *** Hcal recHits
  // *********************

  edm::Handle<HcalSourcePositionData> spd;

  // HCALTotalE = 0.;
  HBTotalE = HETotalE = HOTotalE = HFTotalE = 0.;
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
	  if (!Pass_NoiseSummary) HBTimeFlagged2->Fill(j->time()); 
	  if (j->flagField(HcalCaloFlagLabels::HBHETimingShapedCutsBits) != 0) HBTimeFlagged->Fill(j->time()); 
	  HBTvsE->Fill(j->energy(), j->time());

	  if (j->time() > 20.) HBEneTThr->Fill(j->energy()); 
	  
	  if ((j->time()<-25.) || (j->time()>75.)) {
	    HBEneOOT->Fill(j->energy()); 
	    if (j->energy() > HBHEThreshold)  HBEneOOTTh->Fill(j->energy()); 
	    if (j->energy() > HBHEThreshold1) HBEneOOTTh1->Fill(j->energy()); 
	  }
	  if (j->energy() > HBHEThreshold) {
	    HBEneTh->Fill(j->energy()); 
	    HBTimeTh->Fill(j->time()); 
	    if (!Pass_NoiseSummary) HBTimeThFlagged2->Fill(j->time()); 
	    if (j->flagField(HcalCaloFlagLabels::HBHETimingShapedCutsBits) != 0) HBTimeThFlagged->Fill(j->time()); 

	    if (evt.id().run() >= StableRun) HBTimeThR->Fill(j->time()); 
	    HBTotalE += j->energy();
	    HBocc->Fill(j->id().ieta(),j->id().iphi());
	    hitEta->Fill(j->id().ieta());
	    hitPhi->Fill(j->id().iphi());
	  }
	  if (j->energy() > HBHEThreshold1) {
	    HBEneTh1->Fill(j->energy()); 
	    HBTimeTh1->Fill(j->time()); 
	    if (!Pass_NoiseSummary) HBTimeTh1Flagged2->Fill(j->time()); 
	    if (j->flagField(HcalCaloFlagLabels::HBHETimingShapedCutsBits) != 0) HBTimeTh1Flagged->Fill(j->time()); 

	    if (evt.id().run() >= StableRun) HBTimeTh1R->Fill(j->time()); 
	    if ((j->time()<-25.) || (j->time()>75.)) {
	      HBoccOOT->Fill(j->id().ieta(),j->id().iphi());
	    }
	  }
	  if (j->energy() > HBHEThreshold2) {
	    HBTimeTh2->Fill(j->time()); 
	    if (!Pass_NoiseSummary) HBTimeTh2Flagged2->Fill(j->time()); 
	    if (j->flagField(HcalCaloFlagLabels::HBHETimingShapedCutsBits) != 0) HBTimeTh2Flagged->Fill(j->time()); 

	    if (evt.id().run() >= StableRun) HBTimeTh2R->Fill(j->time()); 
	  }
	  if (j->energy() > HBHEThreshold3) {
	    HBTimeTh3->Fill(j->time()); 
	    if (evt.id().run() >= StableRun) HBTimeTh3R->Fill(j->time()); 
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
	  if (!Pass_NoiseSummary) HETimeFlagged2->Fill(j->time()); 
	  if (j->flagField(HcalCaloFlagLabels::HBHETimingShapedCutsBits) != 0) HETimeFlagged->Fill(j->time()); 
	  HETvsE->Fill(j->energy(), j->time());

	  if (j->time() > 20.) HEEneTThr->Fill(j->energy()); 

	  if ((j->time()<-25.) || (j->time()>75.)) {
	    HEEneOOT->Fill(j->energy()); 
	    if (j->energy() > HBHEThreshold)  HEEneOOTTh->Fill(j->energy());  
	    if (j->energy() > HBHEThreshold1) HEEneOOTTh1->Fill(j->energy());  
	  }

	  if (j->energy() > HBHEThreshold) {
	    HEEneTh->Fill(j->energy()); 
	    HETimeTh->Fill(j->time()); 
	    if (!Pass_NoiseSummary) HETimeThFlagged2->Fill(j->time()); 
	    if (j->flagField(HcalCaloFlagLabels::HBHETimingShapedCutsBits) != 0)  HETimeThFlagged->Fill(j->time()); 

	    if (evt.id().run() >= StableRun) HETimeThR->Fill(j->time()); 
	    HETotalE += j->energy();
	    HEocc->Fill(j->id().ieta(),j->id().iphi());
	    hitEta->Fill(j->id().ieta());
	    hitPhi->Fill(j->id().iphi());
	  }
	  if (j->energy() > HBHEThreshold1) {
	    HEEneTh1->Fill(j->energy()); 
	    HETimeTh1->Fill(j->time()); 
	    if (!Pass_NoiseSummary) HETimeTh1Flagged2->Fill(j->time()); 
	    if (j->flagField(HcalCaloFlagLabels::HBHETimingShapedCutsBits) != 0) HETimeTh1Flagged->Fill(j->time()); 
	    if (evt.id().run() >= StableRun) HETimeTh1R->Fill(j->time()); 
	    if ((j->time()<-25.) || (j->time()>75.)) {
	      HEoccOOT->Fill(j->id().ieta(),j->id().iphi());
	    }
	  }
	  if (j->energy() > HBHEThreshold2) {
	    HETimeTh2->Fill(j->time()); 
	    if (!Pass_NoiseSummary) HETimeTh2Flagged2->Fill(j->time()); 
	    if (j->flagField(HcalCaloFlagLabels::HBHETimingShapedCutsBits) != 0) HETimeTh2Flagged->Fill(j->time()); 
	    if (evt.id().run() >= StableRun) HETimeTh2R->Fill(j->time()); 
	  }
	  if (j->energy() > HBHEThreshold3) {
	    HETimeTh3->Fill(j->time()); 
	    if (evt.id().run() >= StableRun) HETimeTh3R->Fill(j->time()); 
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

  int NPMTHits;
  NPMTHits = 0;
  try {
    std::vector<edm::Handle<HFRecHitCollection> > colls;
    evt.getManyByType(colls);
    std::vector<edm::Handle<HFRecHitCollection> >::iterator i;
    for (i=colls.begin(); i!=colls.end(); i++) {
      for (HFRecHitCollection::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++) {
	if ( (j->flagField(HcalCaloFlagLabels::HFLongShort) == 1) ||
	     (j->flagField(HcalCaloFlagLabels::HFDigiTime)  == 1) ) {
	     NPMTHits++;
	}
      }
      break;
    }
  } catch (...) {
    cout << "No HF RecHits." << endl;
  }


  PMTHits->Fill(NPMTHits); 

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

	  if (NPMTHits == 1) {
	    if ( (j->flagField(HcalCaloFlagLabels::HFLongShort) == 1) ||
		 (j->flagField(HcalCaloFlagLabels::HFDigiTime)  == 1) ) {
	      HFEtaFlagged->Fill(j->id().ieta());
	      if (j->id().depth() == 1) HFEtaFlaggedL->Fill(j->id().ieta());
	      if (j->id().depth() == 2) HFEtaFlaggedS->Fill(j->id().ieta());
	    } else {
	      HFEtaNFlagged->Fill(j->id().ieta(), j->energy());
	      HFEtaPhiNFlagged->Fill(j->id().ieta(),j->id().iphi(),j->energy());
	    }
	  }
	  if (j->energy() > 20.) {
	    if (NPMTHits == 0) {
	      HFEnePMT0->Fill(j->energy()); 
	      HFTimePMT0->Fill(j->time()); 
	    }
	    if (NPMTHits == 1) {
	      if ( (j->flagField(HcalCaloFlagLabels::HFLongShort) == 1) ||
		   (j->flagField(HcalCaloFlagLabels::HFDigiTime)  == 1) ) {
		HFEnePMT1->Fill(j->energy()); 
		HFTimePMT1->Fill(j->time()); 
	      }
	    }
	    if (NPMTHits > 1) {
	      if ( (j->flagField(HcalCaloFlagLabels::HFLongShort) == 1) ||
		   (j->flagField(HcalCaloFlagLabels::HFDigiTime)  == 1) ) {
		HFEnePMT2->Fill(j->energy()); 
		HFTimePMT2->Fill(j->time()); 
	      }
	    }
	  }

	  HFTimeVsiEtaP->Fill(j->id().ieta(), j->time());
	  HFTimeVsiEtaM->Fill(j->id().ieta(), j->time());

	  if (j->energy() > 5.) { 
	    HFTimeVsiEtaP5->Fill(j->id().ieta(), j->time());
	    HFTimeVsiEtaM5->Fill(j->id().ieta(), j->time());
	  }	  

	  if (j->energy() > 20.) { 
	    HFTimeVsiEtaP20->Fill(j->id().ieta(), j->time());
	    HFTimeVsiEtaM20->Fill(j->id().ieta(), j->time());
	  }	  

	  HFEne->Fill(j->energy()); 
	  HFTime->Fill(j->time()); 
	  HFTvsE->Fill(j->energy(), j->time());

	  if (j->time() > 20.) HFEneTThr->Fill(j->energy()); 
	 
	  if (j->energy() > 10.) HFTvsEThr->Fill(j->energy(), j->time());

	  if ( (j->flagField(HcalCaloFlagLabels::HFLongShort) == 1)|| 
	       (j->flagField(HcalCaloFlagLabels::HFDigiTime)  == 1) ) {
	    HFEneFlagged->Fill(j->energy());
	    HFoccFlagged->Fill(j->id().ieta(),j->id().iphi());
	    HFTimeFlagged->Fill(j->time()); 
	    HFTvsEFlagged->Fill(j->energy(), j->time());

	    //	    std::cout << "Flagged:  " << j->energy() << " "
	    //		      << j->time()
	    //		      << std::endl;
	  }


	  if (j->flagField(HcalCaloFlagLabels::HFLongShort) == 1) {
	    HFEneFlagged2->Fill(j->energy());
	    HFoccFlagged2->Fill(j->id().ieta(),j->id().iphi());
	    HFTimeFlagged2->Fill(j->time()); 
	    HFTvsEFlagged2->Fill(j->energy(), j->time());
	    if (j->energy() > 10.) HFTvsEFlagged2Thr->Fill(j->energy(), j->time());
	  }

	  if (j->flagField(HcalCaloFlagLabels::HFDigiTime) == 1) {
	    HFTimeFlagged3->Fill(j->time()); 
	  }

	  if (j->energy() > HFThreshold) {
	    HFEneTh->Fill(j->energy()); 
	    HFTimeTh->Fill(j->time()); 
	    if (j->flagField(HcalCaloFlagLabels::HFLongShort) == 1) HFTimeThFlagged2->Fill(j->time()); 
	    if (j->flagField(HcalCaloFlagLabels::HFDigiTime)  == 1) HFTimeThFlagged3->Fill(j->time()); 

	    if (evt.id().run() >= StableRun) HFTimeThR->Fill(j->time()); 
	    if ( (j->flagField(HcalCaloFlagLabels::HFLongShort) == 1)|| 
		 (j->flagField(HcalCaloFlagLabels::HFDigiTime)  == 1) ) {

	      HFTimeThFlagged->Fill(j->time()); 

	      if (j->energy() > HFThreshold2) HFTimeTh2Flagged->Fill(j->time()); 
	      if (j->energy() > HFThreshold3) HFTimeTh3Flagged->Fill(j->time()); 

	      if (evt.id().run() >= StableRun) {
		HFTimeThFlaggedR->Fill(j->time()); 
		if (NPMTHits == 1) HFTimeThFlaggedR1->Fill(j->time()); 
		if (NPMTHits == 2) HFTimeThFlaggedR2->Fill(j->time()); 
		if (NPMTHits == 3) HFTimeThFlaggedR3->Fill(j->time()); 
		if (NPMTHits == 4) HFTimeThFlaggedR4->Fill(j->time()); 
		if (NPMTHits > 1) HFTimeThFlaggedRM->Fill(j->time()); 
	      }
	    }
	    HFTotalE += j->energy();
	    HFocc->Fill(j->id().ieta(),j->id().iphi());
	    hitEta->Fill(j->id().ieta());
	    hitPhi->Fill(j->id().iphi());
	  }

	  if (j->energy() > HFThreshold1) {
	    HFEneTh1->Fill(j->energy());
	    HFTimeTh1->Fill(j->time()); 
	    if (j->flagField(HcalCaloFlagLabels::HFLongShort) == 1) HFTimeTh1Flagged2->Fill(j->time()); 
	    if (j->flagField(HcalCaloFlagLabels::HFDigiTime)  == 1) HFTimeTh1Flagged3->Fill(j->time()); 
	    if (evt.id().run() >= StableRun) HFTimeTh1R->Fill(j->time()); 
	    if ((j->time()<-20.) || (j->time()>20.)) {
	      HFoccOOT->Fill(j->id().ieta(),j->id().iphi());
	    }
	  } 
	  if (j->energy() > HFThreshold2) {
	    HFTimeTh2->Fill(j->time()); 
	    if (j->flagField(HcalCaloFlagLabels::HFLongShort) == 1) HFTimeTh2Flagged2->Fill(j->time()); 
	    if (j->flagField(HcalCaloFlagLabels::HFDigiTime)  == 1) HFTimeTh2Flagged3->Fill(j->time()); 
	    if (evt.id().run() >= StableRun) HFTimeTh2R->Fill(j->time()); 
	  }
	  if (j->energy() > HFThreshold3) {
	    HFTimeTh3->Fill(j->time()); 
	    if (j->flagField(HcalCaloFlagLabels::HFLongShort) == 1) HFTimeTh3Flagged2->Fill(j->time()); 
	    if (j->flagField(HcalCaloFlagLabels::HFDigiTime)  == 1) HFTimeTh3Flagged3->Fill(j->time()); 
	    if (evt.id().run() >= StableRun) HFTimeTh3R->Fill(j->time()); 
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
      break;

    }

  } catch (...) {
    cout << "No HF RecHits." << endl;
  }



  for (int ieta=0; ieta<100; ieta++) {
     for (int iphi=0; iphi<100; iphi++) {
       double longF, shortF;
       if (HFRecHit[ieta][iphi][0] == -10.) {
	 longF = 0.;
       } else {
	 longF = HFRecHit[ieta][iphi][0];
       }
       if (HFRecHit[ieta][iphi][1] == -10.) {
	 shortF = 0.;
       } else {
	 shortF = HFRecHit[ieta][iphi][1];
       }
       //       if ((longF > HFThreshold) || (shortF > HFThreshold)) HFLSRatio->Fill((longF-shortF)/(longF+shortF));

       if (longF > 0.) HFLEneAll->Fill(longF);
       if (shortF > 0.) HFSEneAll->Fill(shortF);


       if ((longF > 20.) || (shortF > 20.)) {
	 double R = (longF-shortF)/(longF+shortF);
	 HFLSRatio->Fill(R);
	 if (fabs(R) > 0.995) {	   

	   //	   if (longF > 110.)  {
	   //	   if (longF > 50.)  {
	   if (longF > (162.4-10.19*abs(ieta-41)+.21*abs(ieta-41)*abs(ieta-41)) )  {
	     HFEtaFlaggedLN->Fill(ieta-41);

	     HFLEneAllF->Fill(longF);

	     if (shortF == 0.) HFLEneNoSFlaggedN->Fill(longF);
	   }
	   //	   if (shortF > 70.)  {
	   //	   if (shortF > 50.)  {
	   if (shortF > (129.9-6.61*abs(ieta-41)+0.1153*abs(ieta-41)*abs(ieta-41)) ) {
	     HFEtaFlaggedSN->Fill(ieta-41);

	     HFSEneAllF->Fill(shortF);

	     if (longF == 0.) HFSEneNoLFlaggedN->Fill(shortF);
	   }
	 }
       }
       /***
       cout << "HF LS Ratio long= " 
	    << longF
	    << " short= "
	    << shortF
	    << endl;
       ***/

       HFLvsS->Fill(HFRecHit[ieta][iphi][1], HFRecHit[ieta][iphi][0]);         
       if ( (HFRecHit[ieta][iphi][1] == -10.) && (HFRecHit[ieta][iphi][0] != -10.) ) {
         HFLEneNoS->Fill(HFRecHit[ieta][iphi][0]);
	 if (HFRecHitFlag[ieta][iphi][0] !=0 ) HFLEneNoSFlagged->Fill(HFRecHit[ieta][iphi][0]);
       }
       if ( (HFRecHit[ieta][iphi][0] == -10.) && (HFRecHit[ieta][iphi][1] != -10.) ) {
         HFSEneNoL->Fill(HFRecHit[ieta][iphi][1]);
	 if (HFRecHitFlag[ieta][iphi][1] !=0 ) HFSEneNoLFlagged->Fill(HFRecHit[ieta][iphi][1]);
       }

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
	  if (j->energy() > HOThreshold1) {
	    HOEneTh1->Fill(j->energy()); 
	  }
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

  // HCALTotalE = HBTotalE + HETotalE + HFTotalE + HOTotalE;
  // ECALTotalE = 0.;
  EBTotalE = EETotalE = 0.;


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

  // ECALTotalE = EBTotalE + EETotalE;

  if ( (EBTotalE > 320000)  && (EBTotalE < 330000) && 
       (HBTotalE > 2700000) && (HBTotalE < 2800000) ) {

    std::cout << ">>> Off Axis! " 
	      << std::endl;
    
  }

  /***
  std::cout << " Rechits: Total Energy :  " 
	    << " HCAL= " << HCALTotalE 
	    << " ECAL= " << ECALTotalE
	    << " HB = " << HBTotalE
	    << " EB = " << EBTotalE
	    << std::endl;
  ***/


  // *********************
  // *** CaloTowers
  // *********************
  //  Handle<CaloTowerCollection> caloTowers;
  //  evt.getByLabel( "towerMaker", caloTowers );

  nTow1 = nTow2 = nTow3 = nTow4 = 0;

  double sum_et = 0.0;
  double sum_ex = 0.0;
  double sum_ey = 0.0;

  double HFsum_et = 0.0;
  double HFsum_ex = 0.0;
  double HFsum_ey = 0.0;
  //  double sum_ez = 0.0;


  //  std::cout<<">>>> Run " << evt.id().run() << " Event " << evt.id().event() << std::endl;
  // --- Loop over towers and make a lists of used and unused towers
  for (CaloTowerCollection::const_iterator tower = caloTowers->begin();
       tower != caloTowers->end(); tower++) {

    Double_t  et   = tower->et();
    Double_t  phix = tower->phi();
    
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

    if (fabs(tower->ieta()) > 29) {
      HFsum_et += et;
      HFsum_ex += et*cos(phix);
      HFsum_ey += et*sin(phix);
    }


    if (et>0.5) {

      ETime->Fill(tower->ecalTime());
      HTime->Fill(tower->hcalTime());

      // ********
      //      double theta = tower->theta();
      //      double e     = tower->energy();
      //      double et    = e*sin(theta);
      //      double et    = tower->emEt() + tower->hadEt();
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

  HFSumEt->Fill(HFsum_et);
  HFMET->Fill(sqrt( HFsum_ex*HFsum_ex + HFsum_ey*HFsum_ey));

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
    if (std::abs(OER) > 0.8) OERMET->Fill(caloMET);

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

  //  std::cout << "Reconstructed "<< vC.size() << " vertices" << std::endl ;
  nVTX = vC.size();
  //double vertex_numTrks;
  for (reco::VertexCollection::const_iterator vertex=vC.begin(); vertex!=vC.end(); vertex++){

    h_Vx->Fill(vertex->x());
    h_Vy->Fill(vertex->y());
    h_Vz->Fill(vertex->z());
    VTX  = vertex->z();
    //    vertex_numTrks = vertex->tracksSize();
    //    h_VNTrks->Fill(vertex_numTrks);

  }

  if ((HF_PMM != INVALID) || (nVTX > 0)) {
    HFvsZ->Fill(HF_PMM,VTX);
  }

  // ********************************
  // *** Pixel Clusters
  // ********************************
  //  edm::Handle< edmNew::DetSetVector<SiPixelCluster> > hClusterColl;
  //  evt.getByLabel("siPixelClusters", hClusterColl);
  //  const edmNew::DetSetVector<SiPixelCluster> clustColl = *(hClusterColl.product());

  SiClusters->Fill(clustColl.size());

  // ********************************
  // *** Tracks
  // ********************************
  //  edm::Handle<reco::TrackCollection> trackCollection;
  //  evt.getByLabel("ctfWithMaterialTracks", trackCollection);
  //  evt.getByLabel("generalTracks", trackCollection);
  //  const reco::TrackCollection tC = *(trackCollection.product());

  //  std::cout << "ANA: Reconstructed "<< tC.size() << " tracks" << std::endl ;

  // *************************************
  /*****
  //Get the Vertex Collection
  edm::Handle<std::vector<reco::Vertex> > verticies;  evt.getByLabel("offlinePrimaryVertices", verticies);

  //Fill the variables
  int _ntracksw5 = 0;
  for (std::vector<reco::Vertex>::const_iterator it = verticies->begin(); it != verticies->end(); ++it) {

    //    ntracks->push_back(int(it->tracksSize())); //all tracks considered for vertexing
    //    isvalid->push_back(int(it->isValid()));
    //    isfake->push_back(int(it->isFake()));

    if(it->tracksSize() > 0) {
      std::vector<TrackBaseRef>::const_iterator trackIt;
      for( trackIt = it->tracks_begin(); trackIt != it->tracks_end(); trackIt++) {
	if(fabs((**trackIt).charge()) <= 1.)  { 
	  //tracks that contribute with more than 0.5 weight in vertex reconstruction
	  if (it->trackWeight(*trackIt) >= 0.5 ) 
	    _ntracksw5++;
	}
      }
    }
  }
  *****/
  // *************************************
  

  h_Trk_NTrk->Fill(tC.size());
  if (NPMTHits == 0) TrkMultFlagged0->Fill(tC.size());
  if (NPMTHits == 1) TrkMultFlagged1->Fill(tC.size());
  if (NPMTHits == 2) TrkMultFlagged2->Fill(tC.size());
  if (NPMTHits == 3) TrkMultFlagged3->Fill(tC.size());
  if (NPMTHits == 4) TrkMultFlagged4->Fill(tC.size());
  if (NPMTHits > 1) TrkMultFlaggedM->Fill(tC.size());
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
    //    if (iJet == 1) {
    //      cout << " CaloJet: Event Type = "   << evtType 
    //	   << " pt = " << ijet->pt()
    //	   << endl; 
    //    }
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

#include "DQMOffline/JetMET/interface/BeamHaloAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//author : Ronny Remington, University of Florida
//date : 11/11/09

using namespace edm;
using namespace std;
using namespace reco;

int Phi_To_iPhi(float phi) 
{
  phi = phi < 0 ? phi + 2.*TMath::Pi() : phi ;
  float phi_degrees = phi * (360.) / ( 2. * TMath::Pi() ) ;
  int iPhi = (int) ( ( phi_degrees/5. ) + 1.);
   
  return iPhi < 73 ? iPhi : 73 ;
}


BeamHaloAnalyzer::BeamHaloAnalyzer( const edm::ParameterSet& iConfig)
{
  OutputFileName = iConfig.getParameter<std::string>("OutputFile"); 
  TextFileName   = iConfig.getParameter<std::string>("TextFile");

  if(TextFileName.size())
    out = new ofstream(TextFileName.c_str() );


  if( iConfig.exists("StandardDQM") )  // If StandardDQM == true , coarse binning is used on selected (important) histograms   
    StandardDQM = iConfig.getParameter<bool>("StandardDQM");
  else
    StandardDQM = false;
  
  //Get Input Tags
  //Digi Level 
  IT_L1MuGMTReadout = iConfig.getParameter<edm::InputTag>("L1MuGMTReadoutLabel");
  
  //RecHit Level
  IT_CSCRecHit   = iConfig.getParameter<edm::InputTag>("CSCRecHitLabel");
  IT_EBRecHit    = iConfig.getParameter<edm::InputTag>("EBRecHitLabel");
  IT_EERecHit    = iConfig.getParameter<edm::InputTag>("EERecHitLabel");
  IT_ESRecHit    = iConfig.getParameter<edm::InputTag>("ESRecHitLabel");
  IT_HBHERecHit  = iConfig.getParameter<edm::InputTag>("HBHERecHitLabel");
  IT_HFRecHit    = iConfig.getParameter<edm::InputTag>("HFRecHitLabel");
  IT_HORecHit    = iConfig.getParameter<edm::InputTag>("HORecHitLabel");

  //Higher Level Reco 
  IT_CSCSegment = iConfig.getParameter<edm::InputTag>("CSCSegmentLabel");  
  IT_CosmicStandAloneMuon = iConfig.getParameter<edm::InputTag>("CosmicStandAloneMuonLabel"); 
  IT_BeamHaloMuon = iConfig.getParameter<edm::InputTag>("BeamHaloMuonLabel");
  IT_CollisionMuon = iConfig.getParameter<edm::InputTag>("CollisionMuonLabel");
  IT_CollisionStandAloneMuon  = iConfig.getParameter<edm::InputTag>("CollisionStandAloneMuonLabel"); 
  IT_met = iConfig.getParameter<edm::InputTag>("metLabel");
  IT_CaloTower = iConfig.getParameter<edm::InputTag>("CaloTowerLabel");
  IT_SuperCluster = iConfig.getParameter<edm::InputTag>("SuperClusterLabel");
  IT_Photon = iConfig.getParameter<edm::InputTag>("PhotonLabel") ;
  
  //Halo Data
  IT_CSCHaloData = iConfig.getParameter<edm::InputTag> ("CSCHaloDataLabel");
  IT_EcalHaloData = iConfig.getParameter<edm::InputTag>("EcalHaloDataLabel");
  IT_HcalHaloData = iConfig.getParameter<edm::InputTag>("HcalHaloDataLabel");
  IT_GlobalHaloData = iConfig.getParameter<edm::InputTag>("GlobalHaloDataLabel");
  IT_BeamHaloSummary = iConfig.getParameter<edm::InputTag>("BeamHaloSummaryLabel");

  FolderName = iConfig.getParameter<std::string>("folderName");
  DumpMET = iConfig.getParameter<double>("DumpMET");
}


void BeamHaloAnalyzer::beginJob(void){}

void BeamHaloAnalyzer::beginRun(const edm::Run&, const edm::EventSetup& iSetup){
  
  dqm = edm::Service<DQMStore>().operator->();
  if( dqm ) {
  
    // EcalHaloData
    dqm->setCurrentFolder(FolderName+"/EcalHaloData");
    if(StandardDQM)
      {
	ME["EcalHaloData_PhiWedgeMultiplicity"] = dqm->book1D("EcalHaloData_PhiWedgeMultiplicity","",20, -0.5, 19.5);
	ME["EcalHaloData_PhiWedgeConstituents"] = dqm->book1D("EcalHaloData_PhiWedgeConstituents","",20,-0.5, 19.5);
	ME["EcalHaloData_PhiWedgeiPhi"]         = dqm->book1D("EcalHaloData_PhiWedgeiPhi","", 360, 0.5, 360.5) ;
	ME["EcalHaloData_PhiWedgeZDirectionConfidence"] = dqm->book1D("EcalHaloData_ZDirectionConfidence","",  120, -1.2, 1.2);
	ME["EcalHaloData_SuperClusterShowerShapes"]  = dqm->book2D("EcalHaloData_SuperClusterShowerShapes","", 25,0.0, TMath::Pi(), 25,0.0, 2.0);
	ME["EcalHaloData_SuperClusterEnergy"] = dqm->book1D("EcalHaloData_SuperClusterEnergy","",100,-0.5,99.5); 
	ME["EcalHaloData_SuperClusterNHits"] = dqm->book1D("EcalHaloData_SuperClusterNHits", "", 20, -0.5, 19.5);
      }
    else
      {
	ME["EcalHaloData_PhiWedgeMultiplicity"] = dqm->book1D("EcalHaloData_PhiWedgeMultiplicity","",20, -0.5, 19.5);
	ME["EcalHaloData_PhiWedgeEnergy"]       = dqm->book1D("EcalHaloData_PhiWedgeEnergy","", 50,-0.5,199.5);
	ME["EcalHaloData_PhiWedgeConstituents"] = dqm->book1D("EcalHaloData_PhiWedgeConstituents","",20,-0.5, 19.5);
	ME["EcalHaloData_PhiWedgeMinTime"]      = dqm->book1D("EcalHaloData_PhiWedgeMinTime","", 100, -225.0, 225.0);
	ME["EcalHaloData_PhiWedgeMaxTime"]      = dqm->book1D("EcalHaloData_PhiWedgeMaxTime","", 100, -225.0, 225.0);
	ME["EcalHaloData_PhiWedgeiPhi"]         = dqm->book1D("EcalHaloData_PhiWedgeiPhi","", 360, 0.5, 360.5) ;
	ME["EcalHaloData_PhiWedgePlusZDirectionConfidence"] = dqm->book1D("EcalHaloData_PlusZDirectionConfidence","",  50, 0., 1.0);
	ME["EcalHaloData_PhiWedgeZDirectionConfidence"] = dqm->book1D("EcalHaloData_ZDirectionConfidence","",  120, -1.2, 1.2);
	ME["EcalHaloData_PhiWedgeMinVsMaxTime"] = dqm->book2D("EcalHaloData_PhiWedgeMinVsMaxTime","", 50,-100.0, 100.0, 50, -100.0, 100.0);
	ME["EcalHaloData_SuperClusterShowerShapes"]  = dqm->book2D("EcalHaloData_SuperClusterShowerShapes","", 25,0.0, TMath::Pi(), 25,0.0, 2.0);
	ME["EcalHaloData_SuperClusterEnergy"] = dqm->book1D("EcalHaloData_SuperClusterEnergy","",100,-0.5,99.5); 
	ME["EcalHaloData_SuperClusterNHits"] = dqm->book1D("EcalHaloData_SuperClusterNHits", "", 20, -0.5, 19.5);
	ME["EcalHaloData_SuperClusterPhiVsEta"] = dqm->book2D("EcalHaloData_SuperClusterPhiVsEta","",60, -3.0, 3.0,72, -TMath::Pi(), TMath::Pi());  
      }

    // HcalHaloData
    dqm->setCurrentFolder(FolderName+"/HcalHaloData");    
    if( StandardDQM )
      { 
	ME["HcalHaloData_PhiWedgeMultiplicity"] = dqm->book1D("HcalHaloData_PhiWedgeMultiplicity","", 20, -0.5, 19.5);
	ME["HcalHaloData_PhiWedgeConstituents"] = dqm->book1D("HcalHaloData_PhiWedgeConstituents","", 20,-0.5, 19.5);
	ME["HcalHaloData_PhiWedgeiPhi"]         = dqm->book1D("HcalHaloData_PhiWedgeiPhi","", 72, 0.5,72.5);
	ME["HcalHaloData_PhiWedgeZDirectionConfidence"] = dqm->book1D("HcalHaloData_ZDirectionConfidence","",  120, -1.2, 1.2);
      }
    else
      {
	ME["HcalHaloData_PhiWedgeMultiplicity"] = dqm->book1D("HcalHaloData_PhiWedgeMultiplicity","", 20, -0.5, 19.5);
	ME["HcalHaloData_PhiWedgeEnergy"]       = dqm->book1D("HcalHaloData_PhiWedgeEnergy", "", 50,-0.5,199.5);
	ME["HcalHaloData_PhiWedgeConstituents"] = dqm->book1D("HcalHaloData_PhiWedgeConstituents","", 20,-0.5, 19.5);
	ME["HcalHaloData_PhiWedgeiPhi"]         = dqm->book1D("HcalHaloData_PhiWedgeiPhi","", 72, 0.5,72.5);
	ME["HcalHaloData_PhiWedgeMinTime"]      = dqm->book1D("HcalHaloData_PhiWedgeMinTime", "", 50, -100.0, 100.0);
	ME["HcalHaloData_PhiWedgeMaxTime"]      = dqm->book1D("HcalHaloData_PhiWedgeMaxTime", "", 50, -100.0, 100.0);
	ME["HcalHaloData_PhiWedgePlusZDirectionConfidence"] = dqm->book1D("HcalHaloData_PlusZDirectionConfidence","",  50, 0., 1.0);
	ME["HcalHaloData_PhiWedgeZDirectionConfidence"] = dqm->book1D("HcalHaloData_ZDirectionConfidence","",  120, -1.2, 1.2);
	ME["HcalHaloData_PhiWedgeMinVsMaxTime"] = dqm->book2D("HcalHaloData_PhiWedgeMinVsMaxTime","" , 50,-100.0, 100.0, 50, -100.0, 100.0);
      }

    // CSCHaloData
    dqm->setCurrentFolder(FolderName+"/CSCHaloData");
    if( StandardDQM ) 
      {
	ME["CSCHaloData_TrackMultiplicity"]  = dqm->book1D("CSCHaloData_TrackMultiplicity", "", 15, -0.5, 14.5);
	ME["CSCHaloData_TrackMultiplicityMEPlus"]  = dqm->book1D("CSCHaloData_TrackMultiplicityMEPlus", "", 15, -0.5, 14.5);
	ME["CSCHaloData_TrackMultiplicityMEMinus"]  = dqm->book1D("CSCHaloData_TrackMultiplicityMEMinus", "", 15, -0.5, 14.5);
	ME["CSCHaloData_InnerMostTrackHitR"]  = dqm->book1D("CSCHaloData_InnerMostTrackHitR", "", 400, -0.5, 799.5);
	ME["CSCHaloData_InnerMostTrackHitPhi"]  = dqm->book1D("CSCHaloData_InnerMostTrackHitPhi","", 72, -TMath::Pi(), TMath::Pi());
	ME["CSCHaloData_L1HaloTriggersMEPlus"]  = dqm->book1D("CSCHaloData_L1HaloTriggersMEPlus", "", 10, -0.5, 9.5);
	ME["CSCHaloData_L1HaloTriggersMEMinus"]  = dqm->book1D("CSCHaloData_L1HaloTriggersMEMinus", "" , 10, -0.5, 9.5);
	ME["CSCHaloData_L1HaloTriggers"]  = dqm->book1D("CSCHaloData_L1HaloTriggers", "", 10, -0.5, 9.5);
	ME["CSCHaloData_HLHaloTriggers"]  = dqm->book1D("CSCHaloData_HLHaloTriggers", "", 2, -0.5, 1.5);
	ME["CSCHaloData_NOutOfTimeTriggersvsL1HaloExists"]  = dqm->book2D("CSCHaloData_NOutOfTimeTriggersvsL1HaloExists", "", 20, -0.5, 19.5, 2, -0.5, 1.5);
	ME["CSCHaloData_NOutOfTimeTriggers"]  = dqm->book1D("CSCHaloData_NOutOfTimeTriggers", "", 20, -0.5, 19.5);
	ME["CSCHaloData_NOutOfTimeHits"]  = dqm->book1D("CSCHaloData_NOutOfTimeHits", "", 60, -0.5, 59.5);
      }
    else 
      {
	ME["CSCHaloData_TrackMultiplicity"]  = dqm->book1D("CSCHaloData_TrackMultiplicity", "", 15, -0.5, 14.5);
	ME["CSCHaloData_TrackMultiplicityMEPlus"]  = dqm->book1D("CSCHaloData_TrackMultiplicityMEPlus", "", 15, -0.5, 14.5);
	ME["CSCHaloData_TrackMultiplicityMEMinus"]  = dqm->book1D("CSCHaloData_TrackMultiplicityMEMinus", "", 15, -0.5, 14.5);
	ME["CSCHaloData_InnerMostTrackHitXY"]  = dqm->book2D("CSCHaloData_InnerMostTrackHitXY","", 100,-700,700,100, -700,700);
	ME["CSCHaloData_InnerMostTrackHitR"]  = dqm->book1D("CSCHaloData_InnerMostTrackHitR", "", 400, -0.5, 799.5);
	ME["CSCHaloData_InnerMostTrackHitRPlusZ"] = dqm->book2D("CSCHaloData_InnerMostTrackHitRPlusZ","", 400 , 400, 1200, 400, -0.5, 799.5 );
	ME["CSCHaloData_InnerMostTrackHitRMinusZ"] = dqm->book2D("CSCHaloData_InnerMostTrackHitRMinusZ","", 400 , -1200, -400, 400, -0.5, 799.5 );
	ME["CSCHaloData_InnerMostTrackHitiPhi"]  = dqm->book1D("CSCHaloData_InnerMostTrackHitiPhi","", 72, 0.5, 72.5);
	ME["CSCHaloData_InnerMostTrackHitPhi"]  = dqm->book1D("CSCHaloData_InnerMostTrackHitPhi","", 72, -TMath::Pi(), TMath::Pi());
	ME["CSCHaloData_L1HaloTriggersMEPlus"]  = dqm->book1D("CSCHaloData_L1HaloTriggersMEPlus", "", 10, -0.5, 9.5);
	ME["CSCHaloData_L1HaloTriggersMEMinus"]  = dqm->book1D("CSCHaloData_L1HaloTriggersMEMinus", "" , 10, -0.5, 9.5);
	ME["CSCHaloData_L1HaloTriggers"]  = dqm->book1D("CSCHaloData_L1HaloTriggers", "", 10, -0.5, 9.5);
	ME["CSCHaloData_HLHaloTriggers"]  = dqm->book1D("CSCHaloData_HLHaloTriggers", "", 2, -0.5, 1.5);
	ME["CSCHaloData_NOutOfTimeTriggersvsL1HaloExists"]  = dqm->book2D("CSCHaloData_NOutOfTimeTriggersvsL1HaloExists", "", 20, -0.5, 19.5, 2, -0.5, 1.5);
	ME["CSCHaloData_NOutOfTimeTriggers"]  = dqm->book1D("CSCHaloData_NOutOfTimeTriggers", "", 20, -0.5, 19.5);
	ME["CSCHaloData_NOutOfTimeHits"]  = dqm->book1D("CSCHaloData_NOutOfTimeHits", "", 60, -0.5, 59.5);
      }

    // GlobalHaloData
    dqm->setCurrentFolder(FolderName+"/GlobalHaloData");
    if(!StandardDQM)
      {
	ME["GlobalHaloData_MExCorrection"]  = dqm->book1D("GlobalHaloData_MExCorrection", "" , 200, -200., 200.);
	ME["GlobalHaloData_MEyCorrection"]  = dqm->book1D("GlobalHaloData_MEyCorrection", "" , 200, -200., 200.);
	ME["GlobalHaloData_SumEtCorrection"] = dqm->book1D("GlobalHaloData_SumEtCorrection", "" , 200, -0.5, 399.5);
	ME["GlobalHaloData_HaloCorrectedMET"] = dqm->book1D("GlobalHaloData_HaloCorrectedMET", "" , 500, -0.5, 1999.5);
	ME["GlobalHaloData_RawMETMinusHaloCorrectedMET"] = dqm->book1D("GlobalHaloData_RawMETMinusHaloCorrectedMET","" , 250, -500., 500.);
	ME["GlobalHaloData_RawMETOverSumEt"]  = dqm->book1D("GlobalHaloData_RawMETOverSumEt","" , 100, 0.0, 1.0);
	ME["GlobalHaloData_MatchedHcalPhiWedgeMultiplicity"] = dqm->book1D("GlobalHaloData_MatchedHcalPhiWedgeMultiplicity","", 15, -0.5, 14.5);    
	ME["GlobalHaloData_MatchedHcalPhiWedgeEnergy"]       = dqm->book1D("GlobalHaloData_MatchedHcalPhiWedgeEnergy", "", 50,-0.5,199.5);
	ME["GlobalHaloData_MatchedHcalPhiWedgeConstituents"] = dqm->book1D("GlobalHaloData_MatchedHcalPhiWedgeConstituents","", 20,-0.5, 19.5);
	ME["GlobalHaloData_MatchedHcalPhiWedgeiPhi"]         = dqm->book1D("GlobalHaloData_MatchedHcalPhiWedgeiPhi","", 1, 0.5,72.5);
	ME["GlobalHaloData_MatchedHcalPhiWedgeMinTime"]      = dqm->book1D("GlobalHaloData_MatchedHcalPhiWedgeMinTime", "", 50, -100.0, 100.0);
	ME["GlobalHaloData_MatchedHcalPhiWedgeMaxTime"]      = dqm->book1D("GlobalHaloData_MatchedHcalPhiWedgeMaxTime", "", 50, -100.0, 100.0);
	ME["GlobalHaloData_MatchedHcalPhiWedgeZDirectionConfidence"] = dqm->book1D("GlobalHaloData_MatchedHcalPhiWedgeZDirectionConfidence","",  120, -1.2, 1.2);
	ME["GlobalHaloData_MatchedEcalPhiWedgeMultiplicity"] = dqm->book1D("GlobalHaloData_MatchedEcalPhiWedgeMultiplicity","", 15, -0.5, 14.5);
	ME["GlobalHaloData_MatchedEcalPhiWedgeEnergy"]       = dqm->book1D("GlobalHaloData_MatchedEcalPhiWedgeEnergy", "", 50,-0.5,199.5);
	ME["GlobalHaloData_MatchedEcalPhiWedgeConstituents"] = dqm->book1D("GlobalHaloData_MatchedEcalPhiWedgeConstituents","", 20,-0.5, 19.5);
	ME["GlobalHaloData_MatchedEcalPhiWedgeiPhi"]         = dqm->book1D("GlobalHaloData_MatchedEcalPhiWedgeiPhi","", 360, 0.5,360.5);
	ME["GlobalHaloData_MatchedEcalPhiWedgeMinTime"]      = dqm->book1D("GlobalHaloData_MatchedEcalPhiWedgeMinTime", "", 50, -100.0, 100.0);
	ME["GlobalHaloData_MatchedEcalPhiWedgeMaxTime"]      = dqm->book1D("GlobalHaloData_MatchedEcalPhiWedgeMaxTime", "", 50, -100.0, 100.0);
	ME["GlobalHaloData_MatchedEcalPhiWedgeZDirectionConfidence"] = dqm->book1D("GlobalHaloData_MatchedEcalPhiWedgeZDirectionConfidence","",  120, 1.2, 1.2);
      }
    // BeamHaloSummary 
    dqm->setCurrentFolder(FolderName+"/BeamHaloSummary");

    ME["BeamHaloSummary_Id"] = dqm->book1D("BeamHaloSumamry_Id", "", 11, 0.5,11.5);
    ME["BeamHaloSummary_Id"] ->setBinLabel(1,"CSC Loose");
    ME["BeamHaloSummary_Id"] ->setBinLabel(2,"CSC Tight");
    ME["BeamHaloSummary_Id"] ->setBinLabel(3,"Ecal Loose");
    ME["BeamHaloSummary_Id"] ->setBinLabel(4,"Ecal Tight");
    ME["BeamHaloSummary_Id"] ->setBinLabel(5,"Hcal Loose");
    ME["BeamHaloSummary_Id"] ->setBinLabel(6,"Hcal Tight");
    ME["BeamHaloSummary_Id"] ->setBinLabel(7,"Global Loose");
    ME["BeamHaloSummary_Id"] ->setBinLabel(8,"Global Tight");
    ME["BeamHaloSummary_Id"] ->setBinLabel(9,"Event Loose");
    ME["BeamHaloSummary_Id"] ->setBinLabel(10,"Event Tight");
    ME["BeamHaloSummary_Id"] ->setBinLabel(11,"Nothing");
    if(!StandardDQM)
      {
	ME["BeamHaloSummary_BXN"] = dqm->book2D("BeamHaloSummary_BXN", "",11, 0.5, 11.5, 4000, -0.5,3999.5);
	ME["BeamHaloSummary_BXN"] ->setBinLabel(1,"CSC Loose");
	ME["BeamHaloSummary_BXN"] ->setBinLabel(2,"CSC Tight");
	ME["BeamHaloSummary_BXN"] ->setBinLabel(3,"Ecal Loose");
	ME["BeamHaloSummary_BXN"] ->setBinLabel(4,"Ecal Tight");
	ME["BeamHaloSummary_BXN"] ->setBinLabel(5,"Hcal Loose");
	ME["BeamHaloSummary_BXN"] ->setBinLabel(6,"Hcal Tight");
	ME["BeamHaloSummary_BXN"] ->setBinLabel(7,"Global Loose");
	ME["BeamHaloSummary_BXN"] ->setBinLabel(8,"Global Tight");
	ME["BeamHaloSummary_BXN"] ->setBinLabel(9,"Event Loose");
	ME["BeamHaloSummary_BXN"] ->setBinLabel(10,"Event Tight");
	ME["BeamHaloSummary_BXN"] ->setBinLabel(11,"Nothing");
      }
    // Extra
    dqm->setCurrentFolder(FolderName+"/ExtraHaloData");
    if(StandardDQM)
      {
	ME["Extra_CSCTrackInnerOuterDPhi"] = dqm->book1D("Extra_CSCTrackInnerOuterDPhi","",100, 0, TMath::Pi() );
	ME["Extra_CSCTrackInnerOuterDEta"] = dqm->book1D("Extra_CSCTrackInnerOuterDEta","", 100, 0, TMath::Pi() );
	ME["Extra_CSCTrackChi2Ndof"]  = dqm->book1D("Extra_CSCTrackChi2Ndof","", 100, 0, 10);
	ME["Extra_CSCTrackNHits"]     = dqm->book1D("Extra_CSCTrackNHits","", 75,0, 75);
	ME["Extra_CSCActivityWithMET"]= dqm->book2D("Extra_CSCActivityWithMET", "", 4, 0.5, 4.5, 4, 0.5, 4.5);
	ME["Extra_CSCActivityWithMET"]->setBinLabel(1,"Track",1);
	ME["Extra_CSCActivityWithMET"]->setBinLabel(1,"Track",2);
	ME["Extra_CSCActivityWithMET"]->setBinLabel(2, "Segments",1);
	ME["Extra_CSCActivityWithMET"]->setBinLabel(2, "Segments",2);
	ME["Extra_CSCActivityWithMET"]->setBinLabel(3, "RecHits", 1);
	ME["Extra_CSCActivityWithMET"]->setBinLabel(3, "RecHits", 2);
	ME["Extra_CSCActivityWithMET"]->setBinLabel(4, "Nothing", 1);
	ME["Extra_CSCActivityWithMET"]->setBinLabel(4, "Nothing", 2);
	ME["Extra_InnerMostTrackHitR"]  = dqm->book1D("Extra_InnerMostTrackHitR", "", 400, -0.5, 799.5);
	ME["Extra_InnerMostTrackHitPhi"]  = dqm->book1D("Extra_InnerMostTrackHitPhi","", 72, -TMath::Pi(), TMath::Pi());
      }
    else 
      {
	ME["Extra_CSCActivityWithMET"]= dqm->book2D("Extra_CSCActivityWithMET", "", 4, 0.5, 4.5, 4, 0.5, 4.5);
	ME["Extra_CSCActivityWithMET"]->setBinLabel(1,"Track",1);
	ME["Extra_CSCActivityWithMET"]->setBinLabel(1,"Track",2);
	ME["Extra_CSCActivityWithMET"]->setBinLabel(2, "Segments",1);
	ME["Extra_CSCActivityWithMET"]->setBinLabel(2, "Segments",2);
	ME["Extra_CSCActivityWithMET"]->setBinLabel(3, "RecHits", 1);
	ME["Extra_CSCActivityWithMET"]->setBinLabel(3, "RecHits", 2);
	ME["Extra_CSCActivityWithMET"]->setBinLabel(4, "Nothing", 1);
	ME["Extra_CSCActivityWithMET"]->setBinLabel(4, "Nothing", 2);
	ME["Extra_HcalToF"]  = dqm->book2D("Extra_HcalToF","" , 83,-41.5,41.5 , 1000, -125., 125.); 
	ME["Extra_HcalToF_HaloId"]  = dqm->book2D("Extra_HcalToF_HaloId","", 83,-41.5,41.5 , 1000, -125., 125.); 
	ME["Extra_EcalToF"]  = dqm->book2D("Extra_EcalToF","",  171,-85.5,85.5 , 2000, -225., 225.); 
	ME["Extra_EcalToF_HaloId"]  = dqm->book2D("Extra_EcalToF_HaloId","",  171,-85.5,85.5 , 2000, -225., 225.); 
	ME["Extra_CSCTrackInnerOuterDPhi"] = dqm->book1D("Extra_CSCTrackInnerOuterDPhi","",100, 0, TMath::Pi() );
	ME["Extra_CSCTrackInnerOuterDEta"] = dqm->book1D("Extra_CSCTrackInnerOuterDEta","", 100, 0, TMath::Pi() );
	ME["Extra_CSCTrackChi2Ndof"]  = dqm->book1D("Extra_CSCTrackChi2Ndof","", 100, 0, 10);
	ME["Extra_CSCTrackNHits"]     = dqm->book1D("Extra_CSCTrackNHits","", 75,0, 75);
	ME["Extra_InnerMostTrackHitXY"]  = dqm->book2D("Extra_InnerMostTrackHitXY","", 100,-700,700,100, -700,700);
	ME["Extra_InnerMostTrackHitR"]  = dqm->book1D("Extra_InnerMostTrackHitR", "", 400, -0.5, 799.5);
	ME["Extra_InnerMostTrackHitRPlusZ"] = dqm->book2D("Extra_InnerMostTrackHitRPlusZ","", 400 , 400, 1200, 400, -0.5, 799.5 );
	ME["Extra_InnerMostTrackHitRMinusZ"] = dqm->book2D("Extra_InnerMostTrackHitRMinusZ","", 400 , -1200, -400, 400, -0.5, 799.5 );
	ME["Extra_InnerMostTrackHitiPhi"]  = dqm->book1D("Extra_InnerMostTrackHitiPhi","", 72, 0.5, 72.5);
	ME["Extra_InnerMostTrackHitPhi"]  = dqm->book1D("Extra_InnerMostTrackHitPhi","", 72, -TMath::Pi(), TMath::Pi());
	ME["Extra_BXN"] = dqm->book1D("Extra_BXN", "BXN Occupancy", 4000, 0.5, 4000.5);
      }
  }
}

void BeamHaloAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  EventID TheEvent = iEvent.id();
  int BXN = iEvent.bunchCrossing() ;
  bool Dump = TextFileName.size();
  int TheEventNumber = TheEvent.event();
  int Lumi = iEvent.luminosityBlock();
  int Run  = iEvent.run();

  //Get CSC Geometry
  edm::ESHandle<CSCGeometry> TheCSCGeometry;
  iSetup.get<MuonGeometryRecord>().get(TheCSCGeometry);

  //Get CaloGeometry
  edm::ESHandle<CaloGeometry> TheCaloGeometry;
  iSetup.get<CaloGeometryRecord>().get(TheCaloGeometry);

  //Get Stand-alone Muons from Cosmic Muon Reconstruction
  edm::Handle< reco::TrackCollection > TheCosmics;
  iEvent.getByLabel(IT_CosmicStandAloneMuon, TheCosmics);
  bool CSCTrackPlus = false; bool CSCTrackMinus = false;
  if( TheCosmics.isValid() )
    {
      for( reco::TrackCollection::const_iterator cosmic = TheCosmics ->begin() ; cosmic != TheCosmics->end() ; cosmic++ )
	{
	  if( !CSCTrackPlus || !CSCTrackMinus)
            {
              if( cosmic->eta() > 0 || cosmic->outerPosition().z() > 0  || cosmic->innerPosition().z() > 0 ) CSCTrackPlus = true ;
	      else if( cosmic->eta() < 0 || cosmic->outerPosition().z() < 0 || cosmic->innerPosition().z() < 0) CSCTrackMinus = true;
	    }
	  
	  float innermost_phi = 0.;
	  float outermost_phi = 0.;
	  float innermost_z = 99999.;
	  float outermost_z = 0.;
	  float innermost_eta = 0.;
	  float outermost_eta = 0.;
	  float innermost_x =0.;
	  float innermost_y =0.;
	  float innermost_r =0.;
	  for(unsigned int j = 0 ; j < cosmic->extra()->recHits().size(); j++ )
	    {
	      edm::Ref<TrackingRecHitCollection> hit( cosmic->extra()->recHits(), j );
	      DetId TheDetUnitId(hit->geographicalId());
	      if( TheDetUnitId.det() != DetId::Muon ) continue;
	      if( TheDetUnitId.subdetId() != MuonSubdetId::CSC ) continue;

	      const GeomDetUnit *TheUnit = TheCSCGeometry->idToDetUnit(TheDetUnitId);
	      LocalPoint TheLocalPosition = hit->localPosition();  
	      const BoundPlane& TheSurface = TheUnit->surface();
	      const GlobalPoint TheGlobalPosition = TheSurface.toGlobal(TheLocalPosition);
	      
	      float z = TheGlobalPosition.z();
	      if( TMath::Abs(z) < innermost_z )
		{
		  innermost_phi = TheGlobalPosition.phi();
		  innermost_eta = TheGlobalPosition.eta();
		  innermost_z   = TheGlobalPosition.z();
		  innermost_x   = TheGlobalPosition.x();
		  innermost_y   = TheGlobalPosition.y();
		  innermost_r = TMath::Sqrt( innermost_x*innermost_x + innermost_y*innermost_y );
		}
	      if( TMath::Abs(z) > outermost_z)
		{
		  outermost_phi = TheGlobalPosition.phi() ;
		  outermost_eta = TheGlobalPosition.eta() ;
		  outermost_z   = TheGlobalPosition.z();
		}
            }
	  float dphi = TMath::Abs( outermost_phi - innermost_phi );
	  float deta = TMath::Abs( outermost_eta - innermost_eta );
	  ME["Extra_CSCTrackInnerOuterDPhi"] -> Fill( dphi );
	  ME["Extra_CSCTrackInnerOuterDEta"] -> Fill( deta ); 
	  ME["Extra_CSCTrackChi2Ndof"]  -> Fill(cosmic->normalizedChi2() );
	  ME["Extra_CSCTrackNHits"]     -> Fill(cosmic->numberOfValidHits() );
	  ME["Extra_InnerMostTrackHitR"]  ->Fill(innermost_r);
	  ME["Extra_InnerMostTrackHitPhi"] ->Fill(innermost_phi);	  
	  if( !StandardDQM )
	    {
	      ME["Extra_InnerMostTrackHitXY"]  ->Fill(innermost_x, innermost_y);
	      ME["Extra_InnerMostTrackHitiPhi"] ->Fill(Phi_To_iPhi(innermost_phi));	      
	      if(innermost_z > 0 ) 
		ME["Extra_InnerMostTrackHitRPlusZ"] ->Fill(innermost_z, innermost_r);
	      else 
		ME["Extra_InnerMostTrackHitRMinusZ"] ->Fill(innermost_z, innermost_r);
	    }
	}
    }
  
  //Get CSC Segments
  edm::Handle<CSCSegmentCollection> TheCSCSegments;
  iEvent.getByLabel(IT_CSCSegment, TheCSCSegments);

  // Group segments according to endcaps
  std::vector< CSCSegment> vCSCSegments_Plus;
  std::vector< CSCSegment> vCSCSegments_Minus;

  bool CSCSegmentPlus = false; 
  bool CSCSegmentMinus=false;
  if( TheCSCSegments.isValid() ) 
    {
      for(CSCSegmentCollection::const_iterator iSegment = TheCSCSegments->begin(); iSegment != TheCSCSegments->end(); iSegment++) 
	{
	  const std::vector<CSCRecHit2D> vCSCRecHits = iSegment->specificRecHits();
	  CSCDetId iDetId  = (CSCDetId)(*iSegment).cscDetId();
	  
	  if ( iDetId.endcap() == 1 ) vCSCSegments_Plus.push_back( *iSegment );
	  else vCSCSegments_Minus.push_back( *iSegment );
	}      
    }
  
  // Are there segments on the plus/minus side?  
  if( vCSCSegments_Plus.size() ) CSCSegmentPlus = true;
  if( vCSCSegments_Minus.size() ) CSCSegmentMinus = true;

  //Get CSC RecHits
  Handle<CSCRecHit2DCollection> TheCSCRecHits;
  iEvent.getByLabel(IT_CSCRecHit, TheCSCRecHits);
  bool CSCRecHitPlus = false; 
  bool CSCRecHitMinus = false;
  if( TheCSCRecHits.isValid() )
    {
      for(CSCRecHit2DCollection::const_iterator iCSCRecHit = TheCSCRecHits->begin();   iCSCRecHit != TheCSCRecHits->end(); iCSCRecHit++ )
	{
	  DetId TheDetUnitId(iCSCRecHit->geographicalId());
	  const GeomDetUnit *TheUnit = (*TheCSCGeometry).idToDetUnit(TheDetUnitId);
	  LocalPoint TheLocalPosition = iCSCRecHit->localPosition();
	  const BoundPlane& TheSurface = TheUnit->surface();
	  GlobalPoint TheGlobalPosition = TheSurface.toGlobal(TheLocalPosition);

	  //Are there hits on the plus/minus side?
	  if ( TheGlobalPosition.z() > 0 ) CSCRecHitPlus = true;
	  else CSCRecHitMinus = true;
	}
    }
  
  //Get  EB RecHits
  edm::Handle<EBRecHitCollection> TheEBRecHits;
  iEvent.getByLabel(IT_EBRecHit, TheEBRecHits);
  int EBHits=0;
  if( TheEBRecHits.isValid() )
    {
      for( EBRecHitCollection::const_iterator iEBRecHit = TheEBRecHits->begin() ; iEBRecHit != TheEBRecHits->end(); iEBRecHit++)
	{
	  if( iEBRecHit->energy() < 0.5 ) continue;
	  DetId id = DetId( iEBRecHit->id() ) ;
	  EBDetId EcalId ( id.rawId() );
	  int ieta = EcalId.ieta() ;
	  if(!StandardDQM)
	    ME["Extra_EcalToF"] ->Fill(ieta, iEBRecHit->time() );
	  EBHits++;
	}
    }
  

  //Get HB/HE RecHits
  edm::Handle<HBHERecHitCollection> TheHBHERecHits;
  iEvent.getByLabel(IT_HBHERecHit, TheHBHERecHits);
  if( TheHBHERecHits.isValid() )
    {
      for( HBHERecHitCollection::const_iterator iHBHERecHit = TheHBHERecHits->begin(); iHBHERecHit != TheHBHERecHits->end(); iHBHERecHit++)  
	{
	  if( iHBHERecHit->energy() < 1.) continue;
	  HcalDetId id = HcalDetId( iHBHERecHit->id() );
	  if(!StandardDQM)
	    ME["Extra_HcalToF"]->Fill( id.ieta(), iHBHERecHit->time() ) ;
	}
    }

  //Get MET
  edm::Handle< reco::CaloMETCollection > TheCaloMET;
  iEvent.getByLabel(IT_met, TheCaloMET);

  //Get CSCHaloData
  edm::Handle<reco::CSCHaloData> TheCSCDataHandle;
  iEvent.getByLabel(IT_CSCHaloData,TheCSCDataHandle);
  int TheHaloOrigin = 0;
  if (TheCSCDataHandle.isValid())
    {
      const CSCHaloData CSCData = (*TheCSCDataHandle.product());
      if( CSCData.NumberOfHaloTriggers(1) && !CSCData.NumberOfHaloTriggers(-1) ) TheHaloOrigin = 1;
      else if ( CSCData.NumberOfHaloTriggers(-1) && !CSCData.NumberOfHaloTriggers(1) ) TheHaloOrigin = -1 ;
      for( std::vector<GlobalPoint>::const_iterator i=CSCData.GetCSCTrackImpactPositions().begin();  i != CSCData.GetCSCTrackImpactPositions().end() ; i++ )   
	{                          
	  float r = TMath::Sqrt( i->x()*i->x() + i->y()*i->y() );
	  if( !StandardDQM )
	    {
	      ME["CSCHaloData_InnerMostTrackHitXY"]->Fill( i->x(), i->y() );
	      ME["CSCHaloData_InnerMostTrackHitiPhi"]  ->Fill( Phi_To_iPhi( i->phi())); 
	      if( i->z() > 0 ) 
		ME["CSCHaloData_InnerMostTrackHitRPlusZ"] ->Fill(i->z(), r) ;
	      else
		ME["CSCHaloData_InnerMostTrackHitRMinusZ"] ->Fill(i->z(), r) ;
	    }
	  ME["CSCHaloData_InnerMostTrackHitR"]  ->Fill(r);
	  ME["CSCHaloData_InnerMostTrackHitPhi"]  ->Fill( i->phi()); 
	}
      ME["CSCHaloData_L1HaloTriggersMEPlus"]   -> Fill ( CSCData.NumberOfHaloTriggers(1) );
      ME["CSCHaloData_L1HaloTriggersMEMinus"]  -> Fill ( CSCData.NumberOfHaloTriggers(-1));
      ME["CSCHaloData_L1HaloTriggers"]  -> Fill ( CSCData.NumberOfHaloTriggers());
      ME["CSCHaloData_HLHaloTriggers"]  -> Fill ( CSCData.CSCHaloHLTAccept());
      ME["CSCHaloData_TrackMultiplicityMEPlus"] ->Fill ( CSCData.NumberOfHaloTracks(1) );
      ME["CSCHaloData_TrackMultiplicityMEMinus"] ->Fill ( CSCData.NumberOfHaloTracks(-1) );
      ME["CSCHaloData_TrackMultiplicity"]->Fill( CSCData.GetTracks().size() );
      ME["CSCHaloData_NOutOfTimeTriggers"]->Fill( CSCData.NOutOfTimeTriggers() );
      ME["CSCHaloData_NOutOfTimeHits"]->Fill( CSCData.NOutOfTimeHits() );
      ME["CSCHaloData_NOutOfTimeTriggersvsL1HaloExists"]->Fill( CSCData.NOutOfTimeTriggers(), CSCData.NumberOfHaloTriggers() >0 );
    }

  //Get EcalHaloData 
  edm::Handle<reco::EcalHaloData> TheEcalHaloData;
  iEvent.getByLabel(IT_EcalHaloData, TheEcalHaloData );
  if( TheEcalHaloData.isValid() ) 
    {
      const EcalHaloData EcalData = (*TheEcalHaloData.product()); 
      std::vector<PhiWedge> EcalWedges = EcalData.GetPhiWedges();                                                                                              
      for(std::vector<PhiWedge>::const_iterator iWedge = EcalWedges.begin() ; iWedge != EcalWedges.end(); iWedge ++ )                                  
	{                                                                                                                                                     
	  if(!StandardDQM ) 
	    {
	      ME["EcalHaloData_PhiWedgeEnergy"]->Fill( iWedge->Energy() );
	      ME["EcalHaloData_PhiWedgeMinTime"]     ->Fill( iWedge->MinTime() );
	      ME["EcalHaloData_PhiWedgeMaxTime"]     ->Fill( iWedge->MaxTime() );
	      ME["EcalHaloData_PhiWedgeMinVsMaxTime"]->Fill(iWedge->MinTime() , iWedge->MaxTime() ) ;
	      ME["EcalHaloData_PhiWedgePlusZDirectionConfidence"]->Fill( iWedge->PlusZDirectionConfidence() );
	    }
	  ME["EcalHaloData_PhiWedgeZDirectionConfidence"] ->Fill( iWedge->ZDirectionConfidence() );
	  ME["EcalHaloData_PhiWedgeConstituents"]->Fill( iWedge->NumberOfConstituents() ) ;
	  ME["EcalHaloData_PhiWedgeiPhi"]->Fill(iWedge->iPhi() ) ;
	}      

      ME["EcalHaloData_PhiWedgeMultiplicity"]->Fill( EcalWedges.size() );

      edm::ValueMap<float> vm_Angle = EcalData.GetShowerShapesAngle();
      edm::ValueMap<float> vm_Roundness = EcalData.GetShowerShapesRoundness();
      //Access selected SuperClusters
      for(unsigned int n = 0 ; n < EcalData.GetSuperClusters().size() ; n++ )
	{
	  edm::Ref<SuperClusterCollection> cluster(EcalData.GetSuperClusters(), n );
	  float angle = vm_Angle[cluster];
	  float roundness = vm_Roundness[cluster];
	  ME["EcalHaloData_SuperClusterShowerShapes"]->Fill(angle, roundness);
	  ME["EcalHaloData_SuperClusterNHits"]->Fill( cluster->size() );
	  ME["EcalHaloData_SuperClusterEnergy"]->Fill(cluster->energy() );

	  if(!StandardDQM)
	    {
	      ME["EcalHaloData_SuperClusterPhiVsEta"]->Fill(cluster->eta() ,cluster->phi() );
	    }
	}
    }

  //Get HcalHaloData
  edm::Handle<reco::HcalHaloData> TheHcalHaloData;
  iEvent.getByLabel(IT_HcalHaloData ,TheHcalHaloData );
  if( TheHcalHaloData.isValid( ) )
    {
      const HcalHaloData HcalData = (*TheHcalHaloData.product());                                                                
      std::vector<PhiWedge> HcalWedges = HcalData.GetPhiWedges();                                                                                   
      ME["HcalHaloData_PhiWedgeMultiplicity"] ->Fill( HcalWedges.size() );
      for(std::vector<PhiWedge>::const_iterator iWedge = HcalWedges.begin() ; iWedge != HcalWedges.end(); iWedge ++ )                               
	{
	  if( !StandardDQM ) 
	    {
	      ME["HcalHaloData_PhiWedgeEnergy"]       ->Fill( iWedge->Energy() );
	      ME["HcalHaloData_PhiWedgeMinTime"]      ->Fill( iWedge->MinTime() );
	      ME["HcalHaloData_PhiWedgeMaxTime"]      ->Fill( iWedge->MaxTime() );
	      ME["HcalHaloData_PhiWedgePlusZDirectionConfidence"] ->Fill( iWedge->PlusZDirectionConfidence() );
	      ME["HcalHaloData_PhiWedgeMinVsMaxTime"]  ->Fill( iWedge->MinTime() , iWedge->MaxTime() );
	    }	  
	  
	  ME["HcalHaloData_PhiWedgeConstituents"] ->Fill( iWedge->NumberOfConstituents() );
	  ME["HcalHaloData_PhiWedgeiPhi"]         ->Fill( iWedge->iPhi() );
	  ME["HcalHaloData_PhiWedgeZDirectionConfidence"] ->Fill( iWedge->ZDirectionConfidence() );
	}
    }
  

  if(!StandardDQM)
    {
      //Get GlobalHaloData
      edm::Handle<reco::GlobalHaloData> TheGlobalHaloData;
      iEvent.getByLabel(IT_GlobalHaloData, TheGlobalHaloData );
      if( TheGlobalHaloData.isValid() ) 
	{
	  const GlobalHaloData GlobalData =(*TheGlobalHaloData.product());                                                           
	  if( TheCaloMET.isValid() ) 
	    {
	      // Get Raw Uncorrected CaloMET
	      const CaloMETCollection *calometcol = TheCaloMET.product();
	      const CaloMET *RawMET = &(calometcol->front());
	      
	      // Get BeamHalo Corrected CaloMET 
	      const CaloMET CorrectedMET = GlobalData.GetCorrectedCaloMET(*RawMET);
	      ME["GlobalHaloData_MExCorrection"]  ->Fill( GlobalData.DeltaMEx() );
	      ME["GlobalHaloData_MEyCorrection"]  ->Fill( GlobalData.DeltaMEy() );
	      ME["GlobalHaloData_HaloCorrectedMET"]->Fill(CorrectedMET.pt() );
	      ME["GlobalHaloData_RawMETMinusHaloCorrectedMET"] ->Fill( RawMET->pt() - CorrectedMET.pt() );
	      if( RawMET->sumEt() )
		ME["GlobalHaloData_RawMETOverSumEt"] ->Fill( RawMET->pt() / RawMET->sumEt() ); 
	      
	    }                
	  
	  // Get Matched Hcal Phi Wedges
	  std::vector<PhiWedge> HcalWedges = GlobalData.GetMatchedHcalPhiWedges();
	  ME["GlobalHaloData_MatchedHcalPhiWedgeMultiplicity"] ->Fill(HcalWedges.size());
	  // Loop over Matched Hcal Phi Wedges
	  for( std::vector<PhiWedge>::const_iterator iWedge = HcalWedges.begin() ; iWedge != HcalWedges.end() ; iWedge ++ )
	    {
	      ME["GlobalHaloData_MatchedHcalPhiWedgeEnergy"]       ->Fill( iWedge->Energy() );
	      ME["GlobalHaloData_MatchedHcalPhiWedgeConstituents"] ->Fill( iWedge->NumberOfConstituents());
	      ME["GlobalHaloData_MatchedHcalPhiWedgeiPhi"]         ->Fill( iWedge->iPhi() );
	      ME["GlobalHaloData_MatchedHcalPhiWedgeMinTime"]      ->Fill( iWedge->MinTime() );
	      ME["GlobalHaloData_MatchedHcalPhiWedgeMaxTime"]      ->Fill( iWedge->MaxTime() );
	      ME["GlobalHaloData_MatchedHcalPhiWedgeZDirectionConfidence"] ->Fill( iWedge->ZDirectionConfidence() ) ;
	      if( TheHBHERecHits.isValid() )
		{
		  for( HBHERecHitCollection::const_iterator iHBHERecHit = TheHBHERecHits->begin(); iHBHERecHit != TheHBHERecHits->end(); iHBHERecHit++)  
		    {
		      HcalDetId id = HcalDetId( iHBHERecHit->id() ) ;
		      int iphi = id.iphi() ;
		      if( iphi != iWedge->iPhi() ) continue;
		      if( iHBHERecHit->energy() < 1.0) continue;  // Otherwise there are thousands of hits per event (even with negative energies)
		      
		      float time = iHBHERecHit->time();
		      int ieta = id.ieta();
		      ME["Extra_HcalToF_HaloId"] ->Fill( ieta, time );
		    }
		}
	    }

	  // Get Matched Hcal Phi Wedges
	  std::vector<PhiWedge> EcalWedges = GlobalData.GetMatchedEcalPhiWedges();
	  ME["GlobalHaloData_MatchedEcalPhiWedgeMultiplicity"] ->Fill(EcalWedges.size());
	  for( std::vector<PhiWedge>::const_iterator iWedge = EcalWedges.begin() ; iWedge != EcalWedges.end() ; iWedge ++ )
	    {
	      ME["GlobalHaloData_MatchedEcalPhiWedgeEnergy"]       ->Fill(iWedge->Energy());
	      ME["GlobalHaloData_MatchedEcalPhiWedgeConstituents"] ->Fill(iWedge->NumberOfConstituents());
	      ME["GlobalHaloData_MatchedEcalPhiWedgeiPhi"]         ->Fill(iWedge->iPhi());
	      ME["GlobalHaloData_MatchedEcalPhiWedgeMinTime"]      ->Fill(iWedge->MinTime());
	      ME["GlobalHaloData_MatchedEcalPhiWedgeMaxTime"]      ->Fill(iWedge->MaxTime());
	      ME["GlobalHaloData_MatchedEcalPhiWedgeZDirectionConfidence"] ->Fill( iWedge->ZDirectionConfidence() ) ;
	      if( TheEBRecHits.isValid() ) 
		{
		  for( EBRecHitCollection::const_iterator iEBRecHit = TheEBRecHits->begin() ; iEBRecHit != TheEBRecHits->end(); iEBRecHit++ )
		    {
		      if( iEBRecHit->energy() < 0.5 ) continue;
		      DetId id = DetId( iEBRecHit->id() ) ;
		      EBDetId EcalId ( id.rawId() );
		      int iPhi = EcalId.iphi() ;
		      iPhi = (iPhi-1)/5 + 1;
		      if( iPhi != iWedge->iPhi() ) continue;
		      ME["Extra_EcalToF_HaloId"] ->Fill(EcalId.ieta(), iEBRecHit->time() );
		    }
		}
	    }
	}
    }


  // Get BeamHaloSummary 
  edm::Handle<BeamHaloSummary> TheBeamHaloSummary ;
  iEvent.getByLabel(IT_BeamHaloSummary, TheBeamHaloSummary) ;
  if( TheBeamHaloSummary.isValid() ) 
    {
      const BeamHaloSummary TheSummary = (*TheBeamHaloSummary.product() );
      if( TheSummary.CSCLooseHaloId() ) 
	{
	  ME["BeamHaloSummary_Id"] ->Fill(1);
	  if(!StandardDQM) ME["BeamHaloSummary_BXN"] -> Fill( 1, BXN );
	  if(Dump)*out << setw(15) << "CSCLoose" << setw(15) << Run << setw(15) << Lumi << setw(15) << TheEventNumber << endl;
	}
      if( TheSummary.CSCTightHaloId() ) 
	{
	  ME["BeamHaloSummary_Id"] ->Fill(2);
	  if(!StandardDQM)ME["BeamHaloSummary_BXN"] -> Fill( 2, BXN );
	}
      if( TheSummary.EcalLooseHaloId() )
	{
	  ME["BeamHaloSummary_Id"] ->Fill(3);
	  if(!StandardDQM) ME["BeamHaloSummary_BXN"] -> Fill( 3, BXN );
	  if(Dump) *out << setw(15) << "EcalLoose" << setw(15) << Run << setw(15) << Lumi << setw(15) << TheEventNumber << endl;
	}
      if( TheSummary.EcalTightHaloId() ) 
	{
	  ME["BeamHaloSummary_Id"] ->Fill(4);
	  if(!StandardDQM)ME["BeamHaloSummary_BXN"] -> Fill( 4, BXN );
	}
      if( TheSummary.HcalLooseHaloId() ) 
	{
	  ME["BeamHaloSummary_Id"] ->Fill(5);
	  if(!StandardDQM) ME["BeamHaloSummary_BXN"] -> Fill( 5, BXN );
	  if(Dump) *out << setw(15) << "HcalLoose" << setw(15) << Run << setw(15) << Lumi << setw(15) << TheEventNumber << endl;
	}
      if( TheSummary.HcalTightHaloId() ) 
	{
	  ME["BeamHaloSummary_Id"] ->Fill(6);
	  if(!StandardDQM)ME["BeamHaloSummary_BXN"] -> Fill( 6, BXN );
	}
      if( TheSummary.GlobalLooseHaloId()) 
	{
	  ME["BeamHaloSummary_Id"] ->Fill(7);
	  if(!StandardDQM)ME["BeamHaloSummary_BXN"] -> Fill( 7, BXN );
	  if(Dump) *out << setw(15) << "GlobalLoose" << setw(15) << Run << setw(15) << Lumi << setw(15) << TheEventNumber << endl;
	}
      if( TheSummary.GlobalTightHaloId() )
	{
	  ME["BeamHaloSummary_Id"] ->Fill(8);	
	  if(!StandardDQM)ME["BeamHaloSummary_BXN"] -> Fill( 8, BXN );
	}
      if( TheSummary.LooseId() ) 
	{
	  ME["BeamHaloSummary_Id"] ->Fill(9);
	  if(!StandardDQM)ME["BeamHaloSummary_BXN"] -> Fill( 9, BXN );
	}
      if( TheSummary.TightId() )
	{
	  ME["BeamHaloSummary_Id"] ->Fill(10);
	  if(!StandardDQM)ME["BeamHaloSummary_BXN"] -> Fill( 10, BXN );
	}
      if( !TheSummary.EcalLooseHaloId()  && !TheSummary.HcalLooseHaloId() && !TheSummary.CSCLooseHaloId() && !TheSummary.GlobalLooseHaloId() )
	{
	  ME["BeamHaloSummary_Id"] ->Fill(11);
	  if(!StandardDQM)ME["BeamHaloSummary_BXN"] -> Fill( 11, BXN );
	}
    }

  if( TheCaloMET.isValid() )
    {
      const CaloMETCollection *calometcol = TheCaloMET.product();
      const CaloMET *calomet = &(calometcol->front());
      
      if( calomet->pt() > DumpMET )
	if(Dump) *out << setw(15) << "HighMET" << setw(15) << Run << setw(15) << Lumi << setw(15) << TheEventNumber << endl;

      //Fill CSC Activity Plot 
      if( calomet->pt() > 15.0 ) 
	{
	  if( TheHaloOrigin > 0 )
	    {
	      if( CSCTrackPlus && CSCTrackMinus ) 
		ME["Extra_CSCActivityWithMET"]->Fill(1,1);
	      else if( CSCTrackPlus && CSCSegmentMinus) 
		ME["Extra_CSCActivityWithMET"]->Fill(1,2);
	      else if( CSCTrackPlus && CSCRecHitMinus ) 
		ME["Extra_CSCActivityWithMET"]->Fill(1,3);
	      else if( CSCTrackPlus ) 
		ME["Extra_CSCActivityWithMET"]->Fill(1,4);
	      else if( CSCSegmentPlus && CSCTrackMinus ) 
		ME["Extra_CSCActivityWithMET"]->Fill(2,1);
	      else if( CSCSegmentPlus && CSCSegmentMinus )
		ME["Extra_CSCActivityWithMET"]-> Fill(2,2);
	      else if( CSCSegmentPlus && CSCRecHitMinus   )
		ME["Extra_CSCActivityWithMET"]-> Fill(2,3);
	      else if( CSCSegmentPlus ) 
		ME["Extra_CSCActivityWithMET"]->Fill(2,4 );
	      else if( CSCRecHitPlus && CSCTrackMinus  ) 
		ME["Extra_CSCActivityWithMET"]->Fill(3,1);
	      else if( CSCRecHitPlus && CSCSegmentMinus ) 
		ME["Extra_CSCActivityWithMET"]->Fill(3,2);
	      else if( CSCRecHitPlus && CSCRecHitMinus ) 
		ME["Extra_CSCActivityWithMET"]->Fill(3,3);
	      else if( CSCRecHitPlus ) 
		ME["Extra_CSCActivityWithMET"]->Fill(3,4);
	      else 
		ME["Extra_CSCActivityWithMET"]->Fill(4,4);
	    }
	  else if( TheHaloOrigin < 0 )
	    {
	      if( CSCTrackMinus && CSCTrackPlus ) 
		ME["Extra_CSCActivityWithMET"]->Fill(1,1);
	      else if( CSCTrackMinus && CSCSegmentPlus)
		ME["Extra_CSCActivityWithMET"]->Fill(1,2);
	      else if( CSCTrackMinus && CSCRecHitPlus ) 
		ME["Extra_CSCActivityWithMET"]->Fill(1,3);
	      else if( CSCTrackMinus ) 
		ME["Extra_CSCActivityWithMET"]->Fill(1,4);
	      else if( CSCSegmentMinus && CSCTrackPlus) 
		ME["Extra_CSCActivityWithMET"]->Fill(2,1);
	      else if( CSCSegmentMinus && CSCSegmentPlus ) 
		ME["Extra_CSCActivityWithMET"]->Fill(2,2 );
	      else if( CSCSegmentMinus && CSCRecHitPlus ) 
		ME["Extra_CSCActivityWithMET"]->Fill(2,3);
	      else if( CSCSegmentMinus ) 
		ME["Extra_CSCActivityWithMET"]->Fill(2,4);
	      else if( CSCRecHitMinus && CSCTrackPlus )
		ME["Extra_CSCActivityWithMET"]->Fill(3,1 );
	      else if( CSCRecHitMinus && CSCSegmentPlus )
		ME["Extra_CSCActivityWithMET"]->Fill(3,2 );
	      else if( CSCRecHitMinus && CSCRecHitPlus ) 
		ME["Extra_CSCActivityWithMET"]->Fill(3,3);
	      else if( CSCRecHitMinus )
		ME["Extra_CSCActivityWithMET"]->Fill(3,4);
	      else ME["Extra_CSCActivityWithMET"]->Fill(4,4);
	    }
	}
    }
  
}

void BeamHaloAnalyzer::endJob()
{

}

BeamHaloAnalyzer::~BeamHaloAnalyzer(){
}

//DEFINE_FWK_MODULE(CMSEventAnalyzer);





/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/05/20 13:12:05 $
 *  $Revision: 1.72 $
 *  \author F. Chlebana - Fermilab
 *          K. Hatakeyama - Rockefeller University
 */

#include "DQMOffline/JetMET/interface/CaloMETAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "TLorentzVector.h"

#include <string>
using namespace edm;

// ***********************************************************
CaloMETAnalyzer::CaloMETAnalyzer(const edm::ParameterSet& pSet) {

  parameters = pSet;

  edm::ParameterSet highptjetparms = parameters.getParameter<edm::ParameterSet>("highPtJetTrigger");
  edm::ParameterSet lowptjetparms  = parameters.getParameter<edm::ParameterSet>("lowPtJetTrigger" );
  edm::ParameterSet minbiasparms   = parameters.getParameter<edm::ParameterSet>("minBiasTrigger"  );
  edm::ParameterSet highmetparms   = parameters.getParameter<edm::ParameterSet>("highMETTrigger"  );
  //  edm::ParameterSet lowmetparms    = parameters.getParameter<edm::ParameterSet>("lowMETTrigger"   );
  edm::ParameterSet eleparms       = parameters.getParameter<edm::ParameterSet>("eleTrigger"      );
  edm::ParameterSet muonparms      = parameters.getParameter<edm::ParameterSet>("muonTrigger"     );

  //genericTriggerEventFlag_( new GenericTriggerEventFlag( conf_ ) );
  _HighPtJetEventFlag = new GenericTriggerEventFlag( highptjetparms );
  _LowPtJetEventFlag  = new GenericTriggerEventFlag( lowptjetparms  );
  _MinBiasEventFlag   = new GenericTriggerEventFlag( minbiasparms   );
  _HighMETEventFlag   = new GenericTriggerEventFlag( highmetparms   );
  //  _LowMETEventFlag    = new GenericTriggerEventFlag( lowmetparms    );
  _EleEventFlag       = new GenericTriggerEventFlag( eleparms       );
  _MuonEventFlag      = new GenericTriggerEventFlag( muonparms      );

  highPtJetExpr_ = highptjetparms.getParameter<std::vector<std::string> >("hltPaths");
  lowPtJetExpr_  = lowptjetparms .getParameter<std::vector<std::string> >("hltPaths");
  highMETExpr_   = highmetparms  .getParameter<std::vector<std::string> >("hltPaths");
  //  lowMETExpr_    = lowmetparms   .getParameter<std::vector<std::string> >("hltPaths");
  muonExpr_      = muonparms     .getParameter<std::vector<std::string> >("hltPaths");
  elecExpr_      = eleparms      .getParameter<std::vector<std::string> >("hltPaths");
  minbiasExpr_   = minbiasparms  .getParameter<std::vector<std::string> >("hltPaths");

}

// ***********************************************************
CaloMETAnalyzer::~CaloMETAnalyzer() { 

  delete _HighPtJetEventFlag;
  delete _LowPtJetEventFlag;
  delete _MinBiasEventFlag;
  delete _HighMETEventFlag;
  //  delete _LowMETEventFlag;
  delete _EleEventFlag;
  delete _MuonEventFlag;

}

// ***********************************************************
void CaloMETAnalyzer::beginJob(DQMStore * dbe) {

  evtCounter = 0;
  metname = "caloMETAnalyzer";

  // trigger information
  HLTPathsJetMBByName_ = parameters.getParameter<std::vector<std::string > >("HLTPathsJetMB");

  theCleaningParameters = parameters.getParameter<ParameterSet>("CleaningParameters"),

  //Trigger parameters
  gtTag          = theCleaningParameters.getParameter<edm::InputTag>("gtLabel");
  _techTrigsAND  = theCleaningParameters.getParameter<std::vector<unsigned > >("techTrigsAND");
  _techTrigsOR   = theCleaningParameters.getParameter<std::vector<unsigned > >("techTrigsOR");
  _techTrigsNOT  = theCleaningParameters.getParameter<std::vector<unsigned > >("techTrigsNOT");

  _doHLTPhysicsOn = theCleaningParameters.getParameter<bool>("doHLTPhysicsOn");
  _hlt_PhysDec    = theCleaningParameters.getParameter<std::string>("HLT_PhysDec");

  _tightBHFiltering     = theCleaningParameters.getParameter<bool>("tightBHFiltering");
  _tightJetIDFiltering  = theCleaningParameters.getParameter<int>("tightJetIDFiltering");

  // ==========================================================
  //DCS information
  // ==========================================================
  DCSFilter = new JetMETDQMDCSFilter(parameters.getParameter<ParameterSet>("DCSFilter"));

  //Vertex requirements
  _doPVCheck          = theCleaningParameters.getParameter<bool>("doPrimaryVertexCheck");
  vertexTag  = theCleaningParameters.getParameter<edm::InputTag>("vertexLabel");

  if (_doPVCheck) {
    _nvtx_min        = theCleaningParameters.getParameter<int>("nvtx_min");
    _nvtxtrks_min    = theCleaningParameters.getParameter<int>("nvtxtrks_min");
    _vtxndof_min     = theCleaningParameters.getParameter<int>("vtxndof_min");
    _vtxchi2_max     = theCleaningParameters.getParameter<double>("vtxchi2_max");
    _vtxz_max        = theCleaningParameters.getParameter<double>("vtxz_max");
  }


  // CaloMET information
  theCaloMETCollectionLabel       = parameters.getParameter<edm::InputTag>("METCollectionLabel");
  _source                         = parameters.getParameter<std::string>("Source");

  if (theCaloMETCollectionLabel.label() == "corMetGlobalMuons" ) {
    inputBeamSpotLabel      = parameters.getParameter<edm::InputTag>("InputBeamSpotLabel");
  }
  
  // Other data collections
  theCaloTowersLabel          = parameters.getParameter<edm::InputTag>("CaloTowersLabel");
  theJetCollectionLabel       = parameters.getParameter<edm::InputTag>("JetCollectionLabel");
  HcalNoiseRBXCollectionTag   = parameters.getParameter<edm::InputTag>("HcalNoiseRBXCollection");
  BeamHaloSummaryTag          = parameters.getParameter<edm::InputTag>("BeamHaloSummaryLabel");
  HBHENoiseFilterResultTag    = parameters.getParameter<edm::InputTag>("HBHENoiseFilterResultLabel");

  // misc
  _verbose     = parameters.getParameter<int>("verbose");
  _print       = parameters.getParameter<int>("printOut");
  _etThreshold = parameters.getParameter<double>("etThreshold"); // MET threshold
  _allhist     = parameters.getParameter<bool>("allHist");       // Full set of monitoring histograms
  _allSelection= parameters.getParameter<bool>("allSelection");  // Plot with all sets of event selection
  _cleanupSelection= parameters.getParameter<bool>("cleanupSelection");  // Plot with all sets of event selection

  _highPtJetThreshold = parameters.getParameter<double>("HighPtJetThreshold"); // High Pt Jet threshold
  _lowPtJetThreshold = parameters.getParameter<double>("LowPtJetThreshold"); // Low Pt Jet threshold
  _highMETThreshold = parameters.getParameter<double>("HighMETThreshold"); // High MET threshold
  //  _lowMETThreshold = parameters.getParameter<double>("LowMETThreshold"); // Low MET threshold

  //
  jetID = new reco::helper::JetIDHelper(parameters.getParameter<ParameterSet>("JetIDParams"));

  // DQStore stuff
  LogTrace(metname)<<"[CaloMETAnalyzer] Parameters initialization";
  std::string DirName = "JetMET/MET/"+_source;
  dbe->setCurrentFolder(DirName);

  hmetME = dbe->book1D("metReco", "metReco", 4, 1, 5);
  hmetME->setBinLabel(1,"CaloMET",1);

  _dbe = dbe;

  _FolderNames.push_back("All");
  _FolderNames.push_back("BasicCleanup");
  _FolderNames.push_back("ExtraCleanup");
  _FolderNames.push_back("HcalNoiseFilter");
  _FolderNames.push_back("JetIDMinimal");
  _FolderNames.push_back("JetIDLoose");
  _FolderNames.push_back("JetIDTight");
  _FolderNames.push_back("BeamHaloIDTightPass");
  _FolderNames.push_back("BeamHaloIDLoosePass");
  _FolderNames.push_back("Triggers");
  _FolderNames.push_back("PV");

  for (std::vector<std::string>::const_iterator ic = _FolderNames.begin(); 
       ic != _FolderNames.end(); ic++){
    if (*ic=="All")             bookMESet(DirName+"/"+*ic);
    if (_cleanupSelection){
    if (*ic=="BasicCleanup")    bookMESet(DirName+"/"+*ic);
    if (*ic=="ExtraCleanup")    bookMESet(DirName+"/"+*ic);
    }
    if (_allSelection){
    if (*ic=="HcalNoiseFilter")      bookMESet(DirName+"/"+*ic);
    if (*ic=="JetIDMinimal")         bookMESet(DirName+"/"+*ic);
    if (*ic=="JetIDLoose")           bookMESet(DirName+"/"+*ic);
    if (*ic=="JetIDTight")           bookMESet(DirName+"/"+*ic);
    if (*ic=="BeamHaloIDTightPass")  bookMESet(DirName+"/"+*ic);
    if (*ic=="BeamHaloIDLoosePass")  bookMESet(DirName+"/"+*ic);
    if (*ic=="Triggers")             bookMESet(DirName+"/"+*ic);
    if (*ic=="PV")                   bookMESet(DirName+"/"+*ic);
    }
  }
}


// ***********************************************************
void CaloMETAnalyzer::endJob() {

  delete jetID;
  delete DCSFilter;

}


// ***********************************************************
void CaloMETAnalyzer::bookMESet(std::string DirName)
{
  bool bLumiSecPlot=false;
  if (DirName.find("All")!=std::string::npos) bLumiSecPlot=true;

  bookMonitorElement(DirName,bLumiSecPlot);

  if ( _HighPtJetEventFlag->on() ) {
    bookMonitorElementTriggered(DirName+"/"+"HighPtJet",false);
    hTriggerName_HighPtJet = _dbe->bookString("triggerName_HighPtJet", highPtJetExpr_[0]);
  }  

  if ( _LowPtJetEventFlag->on() ) {
    bookMonitorElementTriggered(DirName+"/"+"LowPtJet",false);
    hTriggerName_LowPtJet = _dbe->bookString("triggerName_LowPtJet", lowPtJetExpr_[0]);
  }

  if ( _MinBiasEventFlag->on() ) {
    bookMonitorElementTriggered(DirName+"/"+"MinBias",false);
    hTriggerName_MinBias = _dbe->bookString("triggerName_MinBias", minbiasExpr_[0]);
    if (_verbose) std::cout << "_MinBiasEventFlag is on, folder created\n";
  }

  if ( _HighMETEventFlag->on() ) {
    bookMonitorElementTriggered(DirName+"/"+"HighMET",false);
    hTriggerName_HighMET = _dbe->bookString("triggerName_HighMET", highMETExpr_[0]);
  }

  //  if ( _LowMETEventFlag->on() ) {
  //    bookMonitorElementTriggered(DirName+"/"+"LowMET",false);
  //    hTriggerName_LowMET = _dbe->bookString("triggerName_LowMET", lowMETExpr_[0]);
  //  }

  if ( _EleEventFlag->on() ) {
    bookMonitorElementTriggered(DirName+"/"+"Ele",false);
    hTriggerName_Ele = _dbe->bookString("triggerName_Ele", elecExpr_[0]);
    if (_verbose) std::cout << "_EleEventFlag is on, folder created\n";
  }

  if ( _MuonEventFlag->on() ) {
    bookMonitorElementTriggered(DirName+"/"+"Muon",false);
    hTriggerName_Muon = _dbe->bookString("triggerName_Muon", muonExpr_[0]);
    if (_verbose) std::cout << "_MuonEventFlag is on, folder created\n";
  }
}  

// ***********************************************************
void CaloMETAnalyzer::bookMonitorElement(std::string DirName, bool bLumiSecPlot=false)
{
  if (_verbose) std::cout << "bookMonitorElement " << DirName << std::endl;

  _dbe->setCurrentFolder(DirName);

  bookMonitorElementTriggered(DirName, bLumiSecPlot);


  if (_allhist) {

    hCaloMaxEtInEmTowers    = _dbe->book1D("METTask_CaloMaxEtInEmTowers",   "METTask_CaloMaxEtInEmTowers"   ,100,0,2000);
    hCaloMaxEtInEmTowers->setAxisTitle("Et(Max) in EM Tower [GeV]",1);
    hCaloMaxEtInHadTowers   = _dbe->book1D("METTask_CaloMaxEtInHadTowers",  "METTask_CaloMaxEtInHadTowers"  ,100,0,2000);
    hCaloMaxEtInHadTowers->setAxisTitle("Et(Max) in Had Tower [GeV]",1);

    hCaloHadEtInHB          = _dbe->book1D("METTask_CaloHadEtInHB","METTask_CaloHadEtInHB",100,0,2000);
    hCaloHadEtInHB->setAxisTitle("Had Et [GeV]",1);
    hCaloHadEtInHO          = _dbe->book1D("METTask_CaloHadEtInHO","METTask_CaloHadEtInHO",25,0,500);
    hCaloHadEtInHO->setAxisTitle("Had Et [GeV]",1);
    hCaloHadEtInHE          = _dbe->book1D("METTask_CaloHadEtInHE","METTask_CaloHadEtInHE",100,0,2000);
    hCaloHadEtInHE->setAxisTitle("Had Et [GeV]",1);
    hCaloHadEtInHF          = _dbe->book1D("METTask_CaloHadEtInHF","METTask_CaloHadEtInHF",50,0,1000);
    hCaloHadEtInHF->setAxisTitle("Had Et [GeV]",1);
    hCaloEmEtInHF           = _dbe->book1D("METTask_CaloEmEtInHF" ,"METTask_CaloEmEtInHF" ,25,0,500);
    hCaloEmEtInHF->setAxisTitle("EM Et [GeV]",1);
    hCaloEmEtInEE           = _dbe->book1D("METTask_CaloEmEtInEE" ,"METTask_CaloEmEtInEE" ,50,0,1000);
    hCaloEmEtInEE->setAxisTitle("EM Et [GeV]",1);
    hCaloEmEtInEB           = _dbe->book1D("METTask_CaloEmEtInEB" ,"METTask_CaloEmEtInEB" ,100,0,2000);
    hCaloEmEtInEB->setAxisTitle("EM Et [GeV]",1);

    hCaloEmMEx= _dbe->book1D("METTask_CaloEmMEx","METTask_CaloEmMEx",200,-500,500);
    hCaloEmMEx->setAxisTitle("EM MEx [GeV]",1);
    hCaloEmMEy= _dbe->book1D("METTask_CaloEmMEy","METTask_CaloEmMEy",200,-500,500);
    hCaloEmMEy->setAxisTitle("EM MEy [GeV]",1);
    hCaloEmMET= _dbe->book1D("METTask_CaloEmMET","METTask_CaloEmMET",200,0,1000);
    hCaloEmMET->setAxisTitle("EM MET [GeV]",1);
    hCaloEmMETPhi= _dbe->book1D("METTask_CaloEmMETPhi","METTask_CaloEmMETPhi", 60, -3.2, 3.2);
    hCaloEmMETPhi->setAxisTitle("EM METPhi [rad]",1);

    hCaloHaMEx= _dbe->book1D("METTask_CaloHaMEx","METTask_CaloHaMEx",200,-500,500);
    hCaloHaMEx->setAxisTitle("HA MEx [GeV]",1);
    hCaloHaMEy= _dbe->book1D("METTask_CaloHaMEy","METTask_CaloHaMEy",200,-500,500);
    hCaloHaMEy->setAxisTitle("HA MEy [GeV]",1);
    hCaloHaMET= _dbe->book1D("METTask_CaloHaMET","METTask_CaloHaMET",200,0,1000); 
    hCaloHaMET->setAxisTitle("HA MET [GeV]",1);
    hCaloHaMETPhi= _dbe->book1D("METTask_CaloHaMETPhi","METTask_CaloHaMETPhi", 60, -3.2, 3.2);
    hCaloHaMETPhi->setAxisTitle("HA METPhi [rad]",1);
  }
}


// ***********************************************************
void CaloMETAnalyzer::bookMonitorElementTriggered(std::string DirName, bool bLumiSecPlot=false)
{
  if (_verbose) std::cout << "bookMonitorElementTriggered " << DirName << std::endl;

  _dbe->setCurrentFolder(DirName);


  hCaloMEx        = _dbe->book1D("METTask_CaloMEx",        "METTask_CaloMEx",        200, -500,  500);
  hCaloMEy        = _dbe->book1D("METTask_CaloMEy",        "METTask_CaloMEy",        200, -500,  500); 
  hCaloMET        = _dbe->book1D("METTask_CaloMET",        "METTask_CaloMET",        200,    0, 1000); 
  hCaloMET1       = _dbe->book1D("METTask_CaloMET1",       "METTask_CaloMET1",        80,    0,  200);
  hCaloMETNoHF    = _dbe->book1D("METTask_CaloMETNoHF",    "METTask_CaloMETNoHF",    200,    0, 1000); 
  hCaloSumET      = _dbe->book1D("METTask_CaloSumET",      "METTask_CaloSumET",      400,    0, 4000); 
  hCaloMETSig     = _dbe->book1D("METTask_CaloMETSig",     "METTask_CaloMETSig",      51,    0,   51);
  hCaloMETPhi     = _dbe->book1D("METTask_CaloMETPhi",     "METTask_CaloMETPhi",      60, -3.2,  3.2); 
  hCaloMETPhi020  = _dbe->book1D("METTask_CaloMETPhi020",  "METTask_CaloMETPhi020",   60, -3.2,  3.2);
  hCaloMET_logx   = _dbe->book1D("METTask_CaloMET_logx",   "METTask_CaloMET_logx",    40,   -1,    7);
  hCaloSumET_logx = _dbe->book1D("METTask_CaloSumET_logx", "METTask_CaloSumET_logx",  40,   -1,    7);

  hCaloMEx       ->setAxisTitle("MEx [GeV]",                 1);
  hCaloMEy       ->setAxisTitle("MEy [GeV]",                 1);
  hCaloMET       ->setAxisTitle("MET [GeV]",                 1);
  hCaloMET1      ->setAxisTitle("MET [GeV]",                 1);
  hCaloMETNoHF   ->setAxisTitle("MET (No HF) [GeV]",         1);
  hCaloSumET     ->setAxisTitle("SumET [GeV]",               1);
  hCaloMETSig    ->setAxisTitle("METSig",                    1);
  hCaloMETPhi    ->setAxisTitle("METPhi [rad]",              1);
  hCaloMETPhi020 ->setAxisTitle("METPhi [rad] (MET>20 GeV)", 1);
  hCaloMET_logx  ->setAxisTitle("log(MET) [GeV]",            1);
  hCaloSumET_logx->setAxisTitle("log(SumET) [GeV]",          1);


  // Book NPV profiles
  //----------------------------------------------------------------------------
  hCaloMEx_profile     = _dbe->bookProfile("METTask_CaloMEx_profile",     "MEx [GeV]",     nbinsPV, PVlow, PVup, 200, -500,  500);
  hCaloMEy_profile     = _dbe->bookProfile("METTask_CaloMEy_profile",     "MEy [GeV]",     nbinsPV, PVlow, PVup, 200, -500,  500); 
  hCaloMET_profile     = _dbe->bookProfile("METTask_CaloMET_profile",     "MET [GeV]",     nbinsPV, PVlow, PVup, 200,    0, 1000); 
  hCaloMETNoHF_profile = _dbe->bookProfile("METTask_CaloMETNoHF_profile", "METNoHF [GeV]", nbinsPV, PVlow, PVup, 200,    0, 1000); 
  hCaloSumET_profile   = _dbe->bookProfile("METTask_CaloSumET_profile",   "SumET [GeV]",   nbinsPV, PVlow, PVup, 400,    0, 4000); 


  // Set NPV profiles x-axis title
  //----------------------------------------------------------------------------
  hCaloMEx_profile    ->setAxisTitle("nvtx", 1);
  hCaloMEy_profile    ->setAxisTitle("nvtx", 1);
  hCaloMET_profile    ->setAxisTitle("nvtx", 1);
  hCaloMETNoHF_profile->setAxisTitle("nvtx", 1);
  hCaloSumET_profile  ->setAxisTitle("nvtx", 1);


  if (_allhist){
    if (bLumiSecPlot){
      hCaloMExLS              = _dbe->book2D("METTask_CaloMEx_LS","METTask_CaloMEx_LS",200,-200,200,50,0.,500.);
      hCaloMExLS->setAxisTitle("MEx [GeV]",1);
      hCaloMExLS->setAxisTitle("Lumi Section",2);
      hCaloMEyLS              = _dbe->book2D("METTask_CaloMEy_LS","METTask_CaloMEy_LS",200,-200,200,50,0.,500.);
      hCaloMEyLS->setAxisTitle("MEy [GeV]",1);
      hCaloMEyLS->setAxisTitle("Lumi Section",2);
    }

    //hCaloMaxEtInEmTowers    = _dbe->book1D("METTask_CaloMaxEtInEmTowers",   "METTask_CaloMaxEtInEmTowers"   ,100,0,2000);
    //hCaloMaxEtInEmTowers->setAxisTitle("Et(Max) in EM Tower [GeV]",1);
    //hCaloMaxEtInHadTowers   = _dbe->book1D("METTask_CaloMaxEtInHadTowers",  "METTask_CaloMaxEtInHadTowers"  ,100,0,2000);
    //hCaloMaxEtInHadTowers->setAxisTitle("Et(Max) in Had Tower [GeV]",1);
    hCaloEtFractionHadronic = _dbe->book1D("METTask_CaloEtFractionHadronic","METTask_CaloEtFractionHadronic",100,0,1);
    hCaloEtFractionHadronic->setAxisTitle("Hadronic Et Fraction",1);
    hCaloEmEtFraction       = _dbe->book1D("METTask_CaloEmEtFraction",      "METTask_CaloEmEtFraction"      ,100,0,1);
    hCaloEmEtFraction->setAxisTitle("EM Et Fraction",1);

    //hCaloEmEtFraction002    = _dbe->book1D("METTask_CaloEmEtFraction002",   "METTask_CaloEmEtFraction002"      ,100,0,1);
    //hCaloEmEtFraction002->setAxisTitle("EM Et Fraction (MET>2 GeV)",1);
    //hCaloEmEtFraction010    = _dbe->book1D("METTask_CaloEmEtFraction010",   "METTask_CaloEmEtFraction010"      ,100,0,1);
    //hCaloEmEtFraction010->setAxisTitle("EM Et Fraction (MET>10 GeV)",1);
    hCaloEmEtFraction020    = _dbe->book1D("METTask_CaloEmEtFraction020",   "METTask_CaloEmEtFraction020"      ,100,0,1);
    hCaloEmEtFraction020->setAxisTitle("EM Et Fraction (MET>20 GeV)",1);
  }
  
  if (theCaloMETCollectionLabel.label() == "corMetGlobalMuons" ) {
    hCalomuPt    = _dbe->book1D("METTask_CalomuonPt", "METTask_CalomuonPt", 50, 0, 500);
    hCalomuEta   = _dbe->book1D("METTask_CalomuonEta", "METTask_CalomuonEta", 60, -3.0, 3.0);
    hCalomuNhits = _dbe->book1D("METTask_CalomuonNhits", "METTask_CalomuonNhits", 50, 0, 50);
    hCalomuChi2  = _dbe->book1D("METTask_CalomuonNormalizedChi2", "METTask_CalomuonNormalizedChi2", 20, 0, 20);
    hCalomuD0    = _dbe->book1D("METTask_CalomuonD0", "METTask_CalomuonD0", 50, -1, 1);
    hCaloMExCorrection       = _dbe->book1D("METTask_CaloMExCorrection", "METTask_CaloMExCorrection", 100, -500.0,500.0);
    hCaloMEyCorrection       = _dbe->book1D("METTask_CaloMEyCorrection", "METTask_CaloMEyCorrection", 100, -500.0,500.0);
    hCaloMuonCorrectionFlag  = _dbe->book1D("METTask_CaloCorrectionFlag","METTask_CaloCorrectionFlag", 5, -0.5, 4.5);
  }

}

// ***********************************************************
void CaloMETAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  if ( _HighPtJetEventFlag->on() ) _HighPtJetEventFlag->initRun( iRun, iSetup );
  if ( _LowPtJetEventFlag ->on() ) _LowPtJetEventFlag ->initRun( iRun, iSetup );
  if ( _MinBiasEventFlag  ->on() ) _MinBiasEventFlag  ->initRun( iRun, iSetup );
  if ( _HighMETEventFlag  ->on() ) _HighMETEventFlag  ->initRun( iRun, iSetup );
  //  if ( _LowMETEventFlag   ->on() ) _LowMETEventFlag   ->initRun( iRun, iSetup );
  if ( _EleEventFlag      ->on() ) _EleEventFlag      ->initRun( iRun, iSetup );
  if ( _MuonEventFlag     ->on() ) _MuonEventFlag     ->initRun( iRun, iSetup );

  if (_HighPtJetEventFlag->on() && _HighPtJetEventFlag->expressionsFromDB(_HighPtJetEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    highPtJetExpr_ = _HighPtJetEventFlag->expressionsFromDB(_HighPtJetEventFlag->hltDBKey(), iSetup);
  if (_LowPtJetEventFlag->on() && _LowPtJetEventFlag->expressionsFromDB(_LowPtJetEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    lowPtJetExpr_  = _LowPtJetEventFlag->expressionsFromDB(_LowPtJetEventFlag->hltDBKey(),   iSetup);
  if (_HighMETEventFlag->on() && _HighMETEventFlag->expressionsFromDB(_HighMETEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    highMETExpr_   = _HighMETEventFlag->expressionsFromDB(_HighMETEventFlag->hltDBKey(),     iSetup);
  //  if (_LowMETEventFlag->on() && _LowMETEventFlag->expressionsFromDB(_LowMETEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
  //    lowMETExpr_    = _LowMETEventFlag->expressionsFromDB(_LowMETEventFlag->hltDBKey(),       iSetup);
  if (_MuonEventFlag->on() && _MuonEventFlag->expressionsFromDB(_MuonEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    muonExpr_      = _MuonEventFlag->expressionsFromDB(_MuonEventFlag->hltDBKey(),           iSetup);
  if (_EleEventFlag->on() && _EleEventFlag->expressionsFromDB(_EleEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    elecExpr_      = _EleEventFlag->expressionsFromDB(_EleEventFlag->hltDBKey(),             iSetup);
  if (_MinBiasEventFlag->on() && _MinBiasEventFlag->expressionsFromDB(_MinBiasEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    minbiasExpr_   = _MinBiasEventFlag->expressionsFromDB(_MinBiasEventFlag->hltDBKey(),     iSetup);

}


// ***********************************************************
void CaloMETAnalyzer::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup, DQMStore * dbe)
{
  
  //
  //--- Check the time length of the Run from the lumi section plots

  std::string dirName = "JetMET/MET/"+_source+"/";
  _dbe->setCurrentFolder(dirName);

  TH1F* tlumisec;

  MonitorElement *meLumiSec = _dbe->get("aaa");
  meLumiSec = _dbe->get("JetMET/lumisec");

  int totlsec=0;
  double totltime=0.;
  if ( meLumiSec->getRootObject() ) {
    tlumisec = meLumiSec->getTH1F();
    for (int i=0; i<500; i++){
      if (tlumisec->GetBinContent(i+1)) totlsec++;
    }
    totltime = double(totlsec*90); // one lumi sec ~ 90 (sec)
  }

  if (totltime==0.) totltime=1.; 

  //
  //--- Make the integrated plots with rate (Hz)

  for (std::vector<std::string>::const_iterator ic = _FolderNames.begin(); ic != _FolderNames.end(); ic++)
    {

      std::string DirName;
      DirName = dirName+*ic;

      makeRatePlot(DirName,totltime);
      if ( _HighPtJetEventFlag->on() ) 
	makeRatePlot(DirName+"/"+"triggerName_HighJetPt",totltime);
      if ( _LowPtJetEventFlag->on() ) 
	makeRatePlot(DirName+"/"+"triggerName_LowJetPt",totltime);
      if ( _MinBiasEventFlag->on() ) 
	makeRatePlot(DirName+"/"+"triggerName_MinBias",totltime);
      if ( _HighMETEventFlag->on() ) 
	makeRatePlot(DirName+"/"+"triggerName_HighMET",totltime);
      //      if ( _LowMETEventFlag->on() ) 
      //	makeRatePlot(DirName+"/"+"triggerName_LowMET",totltime);
      if ( _EleEventFlag->on() ) 
	makeRatePlot(DirName+"/"+"triggerName_Ele",totltime);
      if ( _MuonEventFlag->on() ) 
	makeRatePlot(DirName+"/"+"triggerName_Muon",totltime);
    }
}


// ***********************************************************
void CaloMETAnalyzer::makeRatePlot(std::string DirName, double totltime)
{

  _dbe->setCurrentFolder(DirName);
  MonitorElement *meCaloMET = _dbe->get(DirName+"/"+"METTask_CaloMET");

  TH1F* tCaloMET;
  TH1F* tCaloMETRate;

  if ( meCaloMET )
    if ( meCaloMET->getRootObject() ) {
      tCaloMET     = meCaloMET->getTH1F();
      
      // Integral plot & convert number of events to rate (hz)
      tCaloMETRate = (TH1F*) tCaloMET->Clone("METTask_CaloMETRate");
      for (int i = tCaloMETRate->GetNbinsX()-1; i>=0; i--){
	tCaloMETRate->SetBinContent(i+1,tCaloMETRate->GetBinContent(i+2)+tCaloMET->GetBinContent(i+1));
      }
      for (int i = 0; i<tCaloMETRate->GetNbinsX(); i++){
	tCaloMETRate->SetBinContent(i+1,tCaloMETRate->GetBinContent(i+1)/double(totltime));
      }      

      tCaloMETRate->SetName("METTask_CaloMETRate");
      tCaloMETRate->SetTitle("METTask_CaloMETRate");
      hCaloMETRate = _dbe->book1D("METTask_CaloMETRate",tCaloMETRate);
      hCaloMETRate->setAxisTitle("MET Threshold [GeV]",1);
    }
}


// ***********************************************************
void CaloMETAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
			      const edm::TriggerResults& triggerResults) {

  if (_verbose) std::cout << "CaloMETAnalyzer analyze" << std::endl;

  std::string DirName = "JetMET/MET/"+_source;
  if (_print){
  std::cout << " " << std::endl;
  std::cout << "Event = " << iEvent.id().event() << std::endl;
  }

  LogTrace(metname)<<"[CaloMETAnalyzer] Analyze CaloMET";

  hmetME->Fill(1);

  // ==========================================================  
  // Trigger information 
  //
  _trig_JetMB=0;
  _trig_HighPtJet=0;
  _trig_LowPtJet=0;
  _trig_MinBias=0;
  _trig_HighMET=0;
  //  _trig_LowMET=0;
  _trig_Ele=0;
  _trig_Muon=0;
  _trig_PhysDec=0;
  if(&triggerResults) {   
    
    /////////// Analyzing HLT Trigger Results (TriggerResults) //////////
    
    //
    //
    // Check how many HLT triggers are in triggerResults 
    int ntrigs = triggerResults.size();
    if (_verbose) std::cout << "ntrigs=" << ntrigs << std::endl;
    
    //
    //
    // If index=ntrigs, this HLT trigger doesn't exist in the HLT table for this data.
    const edm::TriggerNames & triggerNames = iEvent.triggerNames(triggerResults);

    //
    //
    const unsigned int nTrig(triggerNames.size());
    for (unsigned int i=0;i<nTrig;++i)
      {
        if (triggerNames.triggerName(i).find(highPtJetExpr_[0].substr(0,highPtJetExpr_[0].rfind("_v")+2))!=std::string::npos && triggerResults.accept(i))
	  _trig_HighPtJet=true;
        else if (triggerNames.triggerName(i).find(lowPtJetExpr_[0].substr(0,lowPtJetExpr_[0].rfind("_v")+2))!=std::string::npos && triggerResults.accept(i))
	  _trig_LowPtJet=true;
        else if (triggerNames.triggerName(i).find(highMETExpr_[0].substr(0,highMETExpr_[0].rfind("_v")+2))!=std::string::npos && triggerResults.accept(i))
	  _trig_HighMET=true;
	//        else if (triggerNames.triggerName(i).find(lowMETExpr_[0].substr(0,lowMETExpr_[0].rfind("_v")+2))!=std::string::npos && triggerResults.accept(i))
	//	  _trig_LowMET=true;
        else if (triggerNames.triggerName(i).find(muonExpr_[0].substr(0,muonExpr_[0].rfind("_v")+2))!=std::string::npos && triggerResults.accept(i))
	  _trig_Muon=true;
        else if (triggerNames.triggerName(i).find(elecExpr_[0].substr(0,elecExpr_[0].rfind("_v")+2))!=std::string::npos && triggerResults.accept(i))
	  _trig_Ele=true;
        else if (triggerNames.triggerName(i).find(minbiasExpr_[0].substr(0,minbiasExpr_[0].rfind("_v")+2))!=std::string::npos && triggerResults.accept(i))
	  _trig_MinBias=true;
      }

    // count number of requested Jet or MB HLT paths which have fired
    for (unsigned int i=0; i!=HLTPathsJetMBByName_.size(); i++) {
      unsigned int triggerIndex = triggerNames.triggerIndex(HLTPathsJetMBByName_[i]);
      if (triggerIndex<triggerResults.size()) {
	if (triggerResults.accept(triggerIndex)) {
	  _trig_JetMB++;
	}
      }
    }
    // for empty input vectors (n==0), take all HLT triggers!
    if (HLTPathsJetMBByName_.size()==0) _trig_JetMB=triggerResults.size()-1;

    /*
      if ( _HighPtJetEventFlag->on() && _HighPtJetEventFlag->accept( iEvent, iSetup) )
      _trig_HighPtJet=1;
      
      if ( _LowPtJetEventFlag->on() && _LowPtJetEventFlag->accept( iEvent, iSetup) )
      _trig_LowPtJet=1;
      
      if ( _MinBiasEventFlag->on() && _MinBiasEventFlag->accept( iEvent, iSetup) )
      _trig_MinBias=1;
      
      if ( _HighMETEventFlag->on() && _HighMETEventFlag->accept( iEvent, iSetup) )
      _trig_HighMET=1;
      
      if ( _LowMETEventFlag->on() && _LowMETEventFlag->accept( iEvent, iSetup) )
      _trig_LowMET=1;
      
      if ( _EleEventFlag->on() && _EleEventFlag->accept( iEvent, iSetup) )
      _trig_Ele=1;
      
      if ( _MuonEventFlag->on() && _MuonEventFlag->accept( iEvent, iSetup) )
      _trig_Muon=1;
    */
      
    if (triggerNames.triggerIndex(_hlt_PhysDec)   != triggerNames.size() &&
	triggerResults.accept(triggerNames.triggerIndex(_hlt_PhysDec)))   _trig_PhysDec=1;
  } else {

    edm::LogInfo("CaloMetAnalyzer") << "TriggerResults::HLT not found, "
	"automatically select events"; 
    //
    // TriggerResults object not found. Look at all events.    
    _trig_JetMB=1;
    
  }
  
  // ==========================================================  
  // CaloMET information

  // **** Get the MET container  
  edm::Handle<reco::CaloMETCollection> calometcoll;
  iEvent.getByLabel(theCaloMETCollectionLabel, calometcoll);

  if(!calometcoll.isValid()) {
    std::cout<<"Unable to find MET results for CaloMET collection "<<theCaloMETCollectionLabel<<std::endl;
    return;
  }

  const reco::CaloMETCollection *calometcol = calometcoll.product();
  const reco::CaloMET *calomet;
  calomet = &(calometcol->front());
  
  LogTrace(metname)<<"[CaloMETAnalyzer] Call to the CaloMET analyzer";

  // **** Get the MET no HF container  
  edm::Handle<reco::CaloMETCollection> calometnohfcoll;
  iEvent.getByLabel("metNoHF", calometnohfcoll);

  if(!calometnohfcoll.isValid()) {
    std::cout<<"Unable to find MET results for CaloMETNoHF collection metNoHF"<<std::endl;
    return;
  }

  const reco::CaloMETCollection *calometnohfcol = calometnohfcoll.product();
  const reco::CaloMET *calometnohf;
  calometnohf = &(calometnohfcol->front());
  
  //LogTrace(metname)<<"[CaloMETAnalyzer] Call to the CaloMET analyzer";

  //Only for corMetGlobalMuons
  if (theCaloMETCollectionLabel.label() == "corMetGlobalMuons" ) {
    
    iEvent.getByLabel("muonMETValueMapProducer" , "muCorrData", corMetGlobalMuons_ValueMap_Handle);
    iEvent.getByLabel("muons", muon_h);
    iEvent.getByLabel(inputBeamSpotLabel, beamSpot_h);
    
    if(!beamSpot_h.isValid()) edm::LogInfo("OutputInfo") << "falied to retrieve beam spot data require by MET Task";
    
    bspot = ( beamSpot_h.isValid() ) ? beamSpot_h->position() : math::XYZPoint(0, 0, 0);
    
  }


  // ==========================================================
  //
  edm::Handle<reco::HcalNoiseRBXCollection> HRBXCollection;
  iEvent.getByLabel(HcalNoiseRBXCollectionTag,HRBXCollection);
  if (!HRBXCollection.isValid()) {
      LogDebug("") << "CaloMETAnalyzer: Could not find HcalNoiseRBX Collection" << std::endl;
      if (_verbose) std::cout << "CaloMETAnalyzer: Could not find HcalNoiseRBX Collection" << std::endl;
  }
  

  edm::Handle<bool> HBHENoiseFilterResultHandle;
  iEvent.getByLabel(HBHENoiseFilterResultTag, HBHENoiseFilterResultHandle);
  bool HBHENoiseFilterResult = *HBHENoiseFilterResultHandle;
  if (!HBHENoiseFilterResultHandle.isValid()) {
    LogDebug("") << "CaloMETAnalyzer: Could not find HBHENoiseFilterResult" << std::endl;
    if (_verbose) std::cout << "CaloMETAnalyzer: Could not find HBHENoiseFilterResult" << std::endl;
  }


  edm::Handle<reco::CaloJetCollection> caloJets;
  iEvent.getByLabel(theJetCollectionLabel, caloJets);
  if (!caloJets.isValid()) {
    LogDebug("") << "CaloMETAnalyzer: Could not find jet product" << std::endl;
    if (_verbose) std::cout << "CaloMETAnalyzer: Could not find jet product" << std::endl;
  }

  edm::Handle<edm::View<reco::Candidate> > towers;
  iEvent.getByLabel(theCaloTowersLabel, towers);
  if (!towers.isValid()) {
    LogDebug("") << "CaloMETAnalyzer: Could not find caltower product" << std::endl;
    if (_verbose) std::cout << "CaloMETAnalyzer: Could not find caltower product" << std::endl;
  }
 
  // ==========================================================
  // CaloMET sanity check

  if (_source=="CaloMET") validateMET(*calomet,towers);

  // ==========================================================

  if (_allhist) computeEmHaMET(towers);
    
  // ==========================================================
  // JetID 

  if (_verbose) std::cout << "JetID starts" << std::endl;
  
  //
  // --- Minimal cuts
  //
  bool bJetIDMinimal=true;
  int nj=0;
  for (reco::CaloJetCollection::const_iterator cal = caloJets->begin(); 
       cal!=caloJets->end(); ++cal){
    jetID->calculate(iEvent, *cal);
    if (_print && nj<=1) std::cout << "Jet pT = " << cal->pt() << " (GeV) "
				   << " eta = " << cal->eta() << " "
				   << " phi = " << cal->phi() << " "
				   << " emf = " << cal->emEnergyFraction() << std::endl;
    nj++;
    if (cal->pt()>10.){
      if (fabs(cal->eta())<=2.6 && 
	  cal->emEnergyFraction()<=0.01) bJetIDMinimal=false;
    }
  }

  //
  // --- Loose cuts
  //
  bool bJetIDLoose=true;
  for (reco::CaloJetCollection::const_iterator cal = caloJets->begin(); 
       cal!=caloJets->end(); ++cal){
    jetID->calculate(iEvent, *cal);
    if (_verbose) std::cout << jetID->n90Hits() << " " 
			    << jetID->restrictedEMF() << " "
			    << cal->pt() << std::endl;
    if (cal->pt()>10.){
      //
      // for all regions
      if (jetID->n90Hits()<2)  bJetIDLoose=false; 
      if (jetID->fHPD()>=0.98) bJetIDLoose=false; 
      //
      // for non-forward
      if (fabs(cal->eta())<2.55){
	if (cal->emEnergyFraction()<=0.01) bJetIDLoose=false; 
      }
      // for forward
      else {
	if (cal->emEnergyFraction()<=-0.9) bJetIDLoose=false; 
	if (cal->pt()>80.){
	if (cal->emEnergyFraction()>= 1.0) bJetIDLoose=false; 
	}
      } // forward vs non-forward
    }   // pt>10 GeV/c
  }     // calor-jets loop

  //
  // --- Tight cuts
  //
  bool bJetIDTight=true;
  bJetIDTight=bJetIDLoose;
  for (reco::CaloJetCollection::const_iterator cal = caloJets->begin(); 
       cal!=caloJets->end(); ++cal){
    jetID->calculate(iEvent, *cal);
    if (cal->pt()>25.){
      //
      // for all regions
      if (jetID->fHPD()>=0.95) bJetIDTight=false; 
      //
      // for 1.0<|eta|<1.75
      if (fabs(cal->eta())>=1.00 && fabs(cal->eta())<1.75){
	if (cal->pt()>80. && cal->emEnergyFraction()>=1.) bJetIDTight=false; 
      }
      //
      // for 1.75<|eta|<2.55
      else if (fabs(cal->eta())>=1.75 && fabs(cal->eta())<2.55){
	if (cal->pt()>80. && cal->emEnergyFraction()>=1.) bJetIDTight=false; 
      }
      //
      // for 2.55<|eta|<3.25
      else if (fabs(cal->eta())>=2.55 && fabs(cal->eta())<3.25){
	if (cal->pt()< 50.                   && cal->emEnergyFraction()<=-0.3) bJetIDTight=false; 
	if (cal->pt()>=50. && cal->pt()< 80. && cal->emEnergyFraction()<=-0.2) bJetIDTight=false; 
	if (cal->pt()>=80. && cal->pt()<340. && cal->emEnergyFraction()<=-0.1) bJetIDTight=false; 
	if (cal->pt()>=340.                  && cal->emEnergyFraction()<=-0.1 
                                             && cal->emEnergyFraction()>=0.95) bJetIDTight=false; 
      }
      //
      // for 3.25<|eta|
      else if (fabs(cal->eta())>=3.25){
	if (cal->pt()< 50.                   && cal->emEnergyFraction()<=-0.3
                                             && cal->emEnergyFraction()>=0.90) bJetIDTight=false; 
	if (cal->pt()>=50. && cal->pt()<130. && cal->emEnergyFraction()<=-0.2
                                             && cal->emEnergyFraction()>=0.80) bJetIDTight=false; 
	if (cal->pt()>=130.                  && cal->emEnergyFraction()<=-0.1 
                                             && cal->emEnergyFraction()>=0.70) bJetIDTight=false; 
      }
    }   // pt>10 GeV/c
  }     // calor-jets loop
  
  if (_verbose) std::cout << "JetID ends" << std::endl;
     
  // ==========================================================
  // HCAL Noise filter
  
  bool bHcalNoiseFilter = HBHENoiseFilterResult;

  // ==========================================================
  // Get BeamHaloSummary
  edm::Handle<reco::BeamHaloSummary> TheBeamHaloSummary ;
  iEvent.getByLabel(BeamHaloSummaryTag, TheBeamHaloSummary) ;

  bool bBeamHaloIDTightPass = true;
  bool bBeamHaloIDLoosePass = true;
  
  if(TheBeamHaloSummary.isValid()) {
    
    const reco::BeamHaloSummary TheSummary = (*TheBeamHaloSummary.product() );
    
    //   std::cout << TheSummary.EcalLooseHaloId() << " "
    // 	    << TheSummary.HcalLooseHaloId() << " "
    // 	    << TheSummary.CSCLooseHaloId()  << " "
    // 	    << TheSummary.GlobalLooseHaloId() << std::endl;
    
    if( TheSummary.EcalLooseHaloId()  || TheSummary.HcalLooseHaloId() || 
	TheSummary.CSCLooseHaloId()   || TheSummary.GlobalLooseHaloId() )
      bBeamHaloIDLoosePass = false;
    
    if( TheSummary.EcalTightHaloId()  || TheSummary.HcalTightHaloId() || 
	TheSummary.CSCTightHaloId()   || TheSummary.GlobalTightHaloId() )
      bBeamHaloIDTightPass = false;
    
  }
  
  if (_verbose) std::cout << "BeamHaloSummary ends" << std::endl;
  
  // ==========================================================
  //Vertex information

  _numPV = 0;
  bool bPrimaryVertex = true;
  if(_doPVCheck){
    bPrimaryVertex = false;
    Handle<reco::VertexCollection> vertexHandle;

    iEvent.getByLabel(vertexTag, vertexHandle);

    if (!vertexHandle.isValid()) {
      LogDebug("") << "CaloMETAnalyzer: Could not find vertex collection" << std::endl;
      if (_verbose) std::cout << "CaloMETAnalyzer: Could not find vertex collection" << std::endl;
    }
    
    if ( vertexHandle.isValid() ){
      reco::VertexCollection vertexCollection = *(vertexHandle.product());
      int vertex_number     = vertexCollection.size();
      reco::VertexCollection::const_iterator v = vertexCollection.begin();
      for ( ; v != vertexCollection.end(); ++v) {
	double vertex_chi2    = v->normalizedChi2();
	double vertex_ndof    = v->ndof();
	bool   fakeVtx        = v->isFake();
	double vertex_Z       = v->z();
	
	if (  !fakeVtx
	      && vertex_number>=_nvtx_min
	      && vertex_ndof   >_vtxndof_min
	      && vertex_chi2   <_vtxchi2_max
	      && fabs(vertex_Z)<_vtxz_max ) {
	  bPrimaryVertex = true;
	  ++_numPV;
	}
      }
    }
  }
  // ==========================================================

  edm::Handle< L1GlobalTriggerReadoutRecord > gtReadoutRecord;
  iEvent.getByLabel( gtTag, gtReadoutRecord);

  if (!gtReadoutRecord.isValid()) {
    LogDebug("") << "CaloMETAnalyzer: Could not find GT readout record" << std::endl;
    if (_verbose) std::cout << "CaloMETAnalyzer: Could not find GT readout record product" << std::endl;
  }
  
  bool bTechTriggers    = true;
  bool bTechTriggersAND = true;
  bool bTechTriggersOR  = false;
  bool bTechTriggersNOT = false;

  if (gtReadoutRecord.isValid()) {
    const TechnicalTriggerWord&  technicalTriggerWordBeforeMask = gtReadoutRecord->technicalTriggerWord();

    if (_techTrigsAND.size() == 0)
      bTechTriggersAND = true;
    else
      for (unsigned ttr = 0; ttr != _techTrigsAND.size(); ttr++) {
	bTechTriggersAND = bTechTriggersAND && technicalTriggerWordBeforeMask.at(_techTrigsAND.at(ttr));
      }
    
    if (_techTrigsAND.size() == 0)
      bTechTriggersOR = true;
    else
      for (unsigned ttr = 0; ttr != _techTrigsOR.size(); ttr++) {
	bTechTriggersOR = bTechTriggersOR || technicalTriggerWordBeforeMask.at(_techTrigsOR.at(ttr));
      }
    if (_techTrigsNOT.size() == 0)
      bTechTriggersNOT = false;
    else
      for (unsigned ttr = 0; ttr != _techTrigsNOT.size(); ttr++) {
	bTechTriggersNOT = bTechTriggersNOT || technicalTriggerWordBeforeMask.at(_techTrigsNOT.at(ttr));
      }
  }
  else
    {
      bTechTriggersAND = true;
      bTechTriggersOR  = true;
      bTechTriggersNOT = false;
    }
    
  if (_techTrigsAND.size()==0)
    bTechTriggersAND = true;
  if (_techTrigsOR.size()==0)
    bTechTriggersOR = true;
  if (_techTrigsNOT.size()==0)
    bTechTriggersNOT = false;
  
  bTechTriggers = bTechTriggersAND && bTechTriggersOR && !bTechTriggersNOT;
    
  // ==========================================================
  // Reconstructed MET Information - fill MonitorElements
  
  bool bHcalNoise  = bHcalNoiseFilter;
  bool bBeamHaloID = bBeamHaloIDLoosePass;
  bool bJetID      = bJetIDMinimal;

  bool bPhysicsDeclared = true;
  if(_doHLTPhysicsOn) bPhysicsDeclared =_trig_PhysDec;

  if      (_tightBHFiltering)       bBeamHaloID = bBeamHaloIDTightPass;

  if      (_tightJetIDFiltering==1)  bJetID      = bJetIDMinimal;
  else if (_tightJetIDFiltering==2)  bJetID      = bJetIDLoose;
  else if (_tightJetIDFiltering==3)  bJetID      = bJetIDTight;
  else if (_tightJetIDFiltering==-1) bJetID      = true;

  bool bBasicCleanup = bTechTriggers && bPrimaryVertex && bPhysicsDeclared;
  bool bExtraCleanup = bBasicCleanup && bHcalNoise && bJetID && bBeamHaloID;

  //std::string DirName = "JetMET/MET/"+_source;
  
  for (std::vector<std::string>::const_iterator ic = _FolderNames.begin(); 
       ic != _FolderNames.end(); ic++){
    if (*ic=="All")                                             fillMESet(iEvent, DirName+"/"+*ic, *calomet, *calometnohf);
    if (DCSFilter->filter(iEvent, iSetup)) {
      if (_cleanupSelection){
	if (*ic=="BasicCleanup"   && bBasicCleanup)                 fillMESet(iEvent, DirName+"/"+*ic, *calomet, *calometnohf);
	if (*ic=="ExtraCleanup"   && bExtraCleanup)                 fillMESet(iEvent, DirName+"/"+*ic, *calomet, *calometnohf);
      }
      if (_allSelection) {
	if (*ic=="HcalNoiseFilter"      && bHcalNoiseFilter )       fillMESet(iEvent, DirName+"/"+*ic, *calomet, *calometnohf);
	if (*ic=="JetIDMinimal"         && bJetIDMinimal)           fillMESet(iEvent, DirName+"/"+*ic, *calomet, *calometnohf);
	if (*ic=="JetIDLoose"           && bJetIDLoose)             fillMESet(iEvent, DirName+"/"+*ic, *calomet, *calometnohf);
	if (*ic=="JetIDTight"           && bJetIDTight)             fillMESet(iEvent, DirName+"/"+*ic, *calomet, *calometnohf);
	if (*ic=="BeamHaloIDTightPass"  && bBeamHaloIDTightPass)    fillMESet(iEvent, DirName+"/"+*ic, *calomet, *calometnohf);
	if (*ic=="BeamHaloIDLoosePass"  && bBeamHaloIDLoosePass)    fillMESet(iEvent, DirName+"/"+*ic, *calomet, *calometnohf);
	if (*ic=="Triggers"             && bTechTriggers)           fillMESet(iEvent, DirName+"/"+*ic, *calomet, *calometnohf);
	if (*ic=="PV"                   && bPrimaryVertex)          fillMESet(iEvent, DirName+"/"+*ic, *calomet, *calometnohf);
      }
    } // DCS
  }
} // CaloMETAnalyzer::analyze

// ***********************************************************
void CaloMETAnalyzer::computeEmHaMET(edm::Handle<edm::View<reco::Candidate> > towers)
{

  edm::View<reco::Candidate>::const_iterator towerCand = towers->begin();
  
  double sum_em_et = 0.0;
  double sum_em_ex = 0.0;
  double sum_em_ey = 0.0;
  double sum_em_ez = 0.0;
  
  double sum_ha_et = 0.0;
  double sum_ha_ex = 0.0;
  double sum_ha_ey = 0.0;
  double sum_ha_ez = 0.0;
  
  for ( ; towerCand != towers->end(); towerCand++)
    {
      const reco::Candidate* candidate = &(*towerCand);
      if (candidate)
	{
	  const CaloTower* calotower = dynamic_cast<const CaloTower*> (candidate);
	  if (calotower){
	    double Tower_ET = calotower->et();
	    if (Tower_ET>0.3) {
	      
	      double phi   = candidate->phi();
	      double theta = candidate->theta();
	      //double e     = candidate->energy();
	      double e_em  = calotower->emEnergy();
	      double e_ha  = calotower->hadEnergy();
	      double et_em = e_em*sin(theta);
	      double et_ha = e_ha*sin(theta);

	      sum_em_ez += e_em*cos(theta);
	      sum_em_et += et_em;
	      sum_em_ex += et_em*cos(phi);
	      sum_em_ey += et_em*sin(phi);

	      sum_ha_ez += e_ha*cos(theta);
	      sum_ha_et += et_ha;
	      sum_ha_ex += et_ha*cos(phi);
	      sum_ha_ey += et_ha*sin(phi);

	    } // Et>0.5
	  }   // calotower
	}     // candidate
    }         // loop
  
  //
  _EmMEx = -sum_em_ex;
  _EmMEy = -sum_em_ey;
  _EmMET = pow(_EmMEx*_EmMEx+_EmMEy*_EmMEy,0.5);
  _EmCaloEz = sum_em_ez;
  _EmSumEt  = sum_em_et;
  _EmMetPhi   = atan2( _EmMEy, _EmMEx ); 
  //
  _HaMEx = -sum_ha_ex;
  _HaMEy = -sum_ha_ey;
  _HaMET = pow(_HaMEx*_HaMEx+_HaMEy*_HaMEy,0.5);
  _HaCaloEz = sum_ha_ez;
  _HaSumEt  = sum_ha_et;
  _HaMetPhi   = atan2( _HaMEy, _HaMEx ); 
  
}
// ***********************************************************
void CaloMETAnalyzer::validateMET(const reco::CaloMET& calomet, 
				  edm::Handle<edm::View<reco::Candidate> > towers)
{

  edm::View<reco::Candidate>::const_iterator towerCand = towers->begin();
  
  double sum_et = 0.0;
  double sum_ex = 0.0;
  double sum_ey = 0.0;
  double sum_ez = 0.0;
  
  for ( ; towerCand != towers->end(); towerCand++)
    {
      const reco::Candidate* candidate = &(*towerCand);
      if (candidate)
	{
	  const CaloTower* calotower = dynamic_cast<const CaloTower*> (candidate);
	  if (calotower){
	    double Tower_ET = calotower->et();
	    if (Tower_ET>0.3) {
	      
	      double phi   = candidate->phi();
	      double theta = candidate->theta();
	      double e     = candidate->energy();
	      double et    = e*sin(theta);
	      sum_ez += e*cos(theta);
	      sum_et += et;
	      sum_ex += et*cos(phi);
	      sum_ey += et*sin(phi);

	    } // Et>0.5
	  }   // calotower
	}     // candidate
    }         // loop
  
  double Mex   = -sum_ex;
  double Mey   = -sum_ey;
  //double Mez   = -sum_ez;
  double Met   = sqrt( sum_ex*sum_ex + sum_ey*sum_ey );
  double Sumet = sum_et;
  //double MetPhi   = atan2( -sum_ey, -sum_ex ); // since MET is now a candidate,
  
  if (_verbose){
    if (Sumet!=calomet.sumEt() || Mex!=calomet.px() || Mey!=calomet.py() || Met!=calomet.pt() ){
      std::cout << _source << std::endl;
      std::cout << "SUMET" << Sumet << " METBlock" << calomet.sumEt() << std::endl;
      std::cout << "MEX"   << Mex   << " METBlock" << calomet.px()    << std::endl;
      std::cout << "MEY"   << Mey   << " METBlock" << calomet.py()    << std::endl;
      std::cout << "MET"   << Met   << " METBlock" << calomet.pt()    << std::endl;
    }
  }  

  if (_print){
    std::cout << "SUMET = " << calomet.sumEt() << " (GeV) "
	      << "MEX"   << calomet.px() << " (GeV) "
	      << "MEY"   << calomet.py() << " (GeV) " 
	      << "MET"   << calomet.pt() << " (GeV) " << std::endl;
  }

}

// ***********************************************************
void CaloMETAnalyzer::fillMESet(const edm::Event& iEvent, std::string DirName, 
				const reco::CaloMET& calomet,
				const reco::CaloMET& calometnohf)
{

  _dbe->setCurrentFolder(DirName);

  bool bLumiSecPlot=false;
  if (DirName.find("All")) bLumiSecPlot=true;

  if (_trig_JetMB)
    fillMonitorElement(iEvent,DirName,"",calomet,calometnohf, bLumiSecPlot);
  if (_trig_HighPtJet)
    fillMonitorElementTriggered(iEvent,DirName,"HighPtJet",calomet,calometnohf,false);
  if (_trig_LowPtJet)
    fillMonitorElementTriggered(iEvent,DirName,"LowPtJet",calomet,calometnohf,false);
  if (_trig_MinBias)
    fillMonitorElementTriggered(iEvent,DirName,"MinBias",calomet,calometnohf,false);
  if (_trig_HighMET)
    fillMonitorElementTriggered(iEvent,DirName,"HighMET",calomet,calometnohf,false);
  //  if (_trig_LowMET)
  //    fillMonitorElementTriggered(iEvent,DirName,"LowMET",calomet,calometnohf,false);
  if (_trig_Ele)
    fillMonitorElementTriggered(iEvent,DirName,"Ele",calomet,calometnohf,false);
  if (_trig_Muon) {
    fillMonitorElementTriggered(iEvent,DirName,"Muon",calomet,calometnohf,false);
  }
}

// ***********************************************************
void CaloMETAnalyzer::fillMonitorElement(const edm::Event& iEvent, std::string DirName, 
					 std::string TriggerTypeName, 
					 const reco::CaloMET& calomet,
					 const reco::CaloMET& calometnohf,
					 bool bLumiSecPlot)
{
  fillMonitorElementTriggered(iEvent, DirName, TriggerTypeName, calomet, calometnohf, bLumiSecPlot);

  double caloSumET = calomet.sumEt();

  /*
  double caloMETSig = calomet.mEtSig();
  //double caloEz     = calomet.e_longitudinal();
  double caloMET    = calomet.pt();
  double caloMEx    = calomet.px();
  double caloMEy    = calomet.py();
  double caloMETPhi = calomet.phi();

  double caloMETNoHF    = calometnohf.pt();
  //double caloMETNoHFPhi = calometnohf.phi();

  if (_verbose) std::cout << _source << " " << caloMET << std::endl;

  double caloEtFractionHadronic = calomet.etFractionHadronic();
  double caloEmEtFraction       = calomet.emEtFraction();
  */
  double caloMaxEtInEMTowers    = calomet.maxEtInEmTowers();
  double caloMaxEtInHadTowers   = calomet.maxEtInHadTowers();

  double caloHadEtInHB = calomet.hadEtInHB();
  double caloHadEtInHO = calomet.hadEtInHO();
  double caloHadEtInHE = calomet.hadEtInHE();
  double caloHadEtInHF = calomet.hadEtInHF();
  double caloEmEtInEB  = calomet.emEtInEB();
  double caloEmEtInEE  = calomet.emEtInEE();
  double caloEmEtInHF  = calomet.emEtInHF();
  //
  //  int myLuminosityBlock;
  //  myLuminosityBlock = (evtCounter++)/1000;
  //  myLuminosityBlock = iEvent.luminosityBlock();
  //

  if (TriggerTypeName!="") DirName = DirName +"/"+TriggerTypeName;

  if (_verbose) std::cout << "_etThreshold = " << _etThreshold << std::endl;
  if (caloSumET>_etThreshold){
  /*
    hCaloMEx    = _dbe->get(DirName+"/"+"METTask_CaloMEx");    if (hCaloMEx     && hCaloMEx->getRootObject() )    hCaloMEx->Fill(caloMEx);
    hCaloMEy    = _dbe->get(DirName+"/"+"METTask_CaloMEy");    if (hCaloMEy     && hCaloMEy->getRootObject() )    hCaloMEy->Fill(caloMEy);
    hCaloMET    = _dbe->get(DirName+"/"+"METTask_CaloMET");    if (hCaloMET     && hCaloMET->getRootObject() )    hCaloMET->Fill(caloMET);
    hCaloMET1   = _dbe->get(DirName+"/"+"METTask_CaloMET1");   if (hCaloMET1    && hCaloMET1->getRootObject() )   hCaloMET1->Fill(caloMET);
    hCaloMETPhi = _dbe->get(DirName+"/"+"METTask_CaloMETPhi"); if (hCaloMETPhi  && hCaloMETPhi->getRootObject() ) hCaloMETPhi->Fill(caloMETPhi);
    hCaloSumET  = _dbe->get(DirName+"/"+"METTask_CaloSumET");  if (hCaloSumET   && hCaloSumET->getRootObject() )  hCaloSumET->Fill(caloSumET);
    hCaloMETSig = _dbe->get(DirName+"/"+"METTask_CaloMETSig"); if (hCaloMETSig  && hCaloMETSig->getRootObject() ) hCaloMETSig->Fill(caloMETSig);
    //hCaloEz     = _dbe->get(DirName+"/"+"METTask_CaloEz");     if (hCaloEz      && hCaloEz->getRootObject() )     hCaloEz->Fill(caloEz);
    
    hCaloMETNoHF    = _dbe->get(DirName+"/"+"METTask_CaloMETNoHF");    if (hCaloMETNoHF     && hCaloMETNoHF->getRootObject() )    hCaloMETNoHF->Fill(caloMETNoHF);
    
    hCaloMET_logx   = _dbe->get(DirName+"/"+"METTask_CaloMET_logx");      if (hCaloMET_logx    && hCaloMET_logx->getRootObject() )   hCaloMET_logx->Fill(log10(caloMET));
    hCaloSumET_logx = _dbe->get(DirName+"/"+"METTask_CaloSumET_logx");    if (hCaloSumET_logx  && hCaloSumET_logx->getRootObject() ) hCaloSumET_logx->Fill(log10(caloSumET));
    
    //  hCaloMETIonFeedbck = _dbe->get(DirName+"/"+"METTask_CaloMETIonFeedbck"); if (hCaloMETIonFeedbck  && hCaloMETIonFeedbck->getRootObject() ) hCaloMETIonFeedbck->Fill(caloMET);
    //  hCaloMETHPDNoise   = _dbe->get(DirName+"/"+"METTask_CaloMETHPDNoise");   if (hCaloMETHPDNoise    && hCaloMETHPDNoise->getRootObject() )   hCaloMETHPDNoise->Fill(caloMET);

    //hCaloMETPhi002 = _dbe->get(DirName+"/"+"METTask_CaloMETPhi002");    if (caloMET>  2. && hCaloMETPhi002  &&  hCaloMETPhi002->getRootObject()) { hCaloMETPhi002->Fill(caloMETPhi);}
    //hCaloMETPhi010 = _dbe->get(DirName+"/"+"METTask_CaloMETPhi010");    if (caloMET> 10. && hCaloMETPhi010  &&  hCaloMETPhi010->getRootObject()) { hCaloMETPhi010->Fill(caloMETPhi);}
    hCaloMETPhi020 = _dbe->get(DirName+"/"+"METTask_CaloMETPhi020");    if (caloMET> 20. && hCaloMETPhi020  &&  hCaloMETPhi020->getRootObject()) { hCaloMETPhi020->Fill(caloMETPhi);}
  */
  
    if (_allhist){
      /*
      if (bLumiSecPlot){
	hCaloMExLS = _dbe->get(DirName+"/"+"METTask_CaloMEx_LS");   if (hCaloMExLS  &&  hCaloMExLS->getRootObject())    hCaloMExLS->Fill(caloMEx,myLuminosityBlock);
	hCaloMEyLS = _dbe->get(DirName+"/"+"METTask_CaloMEy_LS");   if (hCaloMEyLS  &&  hCaloMEyLS->getRootObject())    hCaloMEyLS->Fill(caloMEy,myLuminosityBlock);
      }
      hCaloEtFractionHadronic = _dbe->get(DirName+"/"+"METTask_CaloEtFractionHadronic"); if (hCaloEtFractionHadronic && hCaloEtFractionHadronic->getRootObject())  hCaloEtFractionHadronic->Fill(caloEtFractionHadronic);
      hCaloEmEtFraction       = _dbe->get(DirName+"/"+"METTask_CaloEmEtFraction");       if (hCaloEmEtFraction       && hCaloEmEtFraction->getRootObject())        hCaloEmEtFraction->Fill(caloEmEtFraction);
      
      //hCaloEmEtFraction002 = _dbe->get(DirName+"/"+"METTask_CaloEmEtFraction002");       if (caloMET>  2.  &&  hCaloEmEtFraction002    && hCaloEmEtFraction002->getRootObject()) hCaloEmEtFraction002->Fill(caloEmEtFraction);
      //hCaloEmEtFraction010 = _dbe->get(DirName+"/"+"METTask_CaloEmEtFraction010");       if (caloMET> 10.  &&  hCaloEmEtFraction010    && hCaloEmEtFraction010->getRootObject()) hCaloEmEtFraction010->Fill(caloEmEtFraction);
      hCaloEmEtFraction020 = _dbe->get(DirName+"/"+"METTask_CaloEmEtFraction020");       if (caloMET> 20.  &&  hCaloEmEtFraction020    && hCaloEmEtFraction020->getRootObject()) hCaloEmEtFraction020->Fill(caloEmEtFraction);
      */

      hCaloMaxEtInEmTowers  = _dbe->get(DirName+"/"+"METTask_CaloMaxEtInEmTowers");   if (hCaloMaxEtInEmTowers  && hCaloMaxEtInEmTowers->getRootObject())   hCaloMaxEtInEmTowers->Fill(caloMaxEtInEMTowers);
      hCaloMaxEtInHadTowers = _dbe->get(DirName+"/"+"METTask_CaloMaxEtInHadTowers");  if (hCaloMaxEtInHadTowers && hCaloMaxEtInHadTowers->getRootObject())  hCaloMaxEtInHadTowers->Fill(caloMaxEtInHadTowers);

      hCaloHadEtInHB = _dbe->get(DirName+"/"+"METTask_CaloHadEtInHB");  if (hCaloHadEtInHB  &&  hCaloHadEtInHB->getRootObject())  hCaloHadEtInHB->Fill(caloHadEtInHB);
      hCaloHadEtInHO = _dbe->get(DirName+"/"+"METTask_CaloHadEtInHO");  if (hCaloHadEtInHO  &&  hCaloHadEtInHO->getRootObject())  hCaloHadEtInHO->Fill(caloHadEtInHO);
      hCaloHadEtInHE = _dbe->get(DirName+"/"+"METTask_CaloHadEtInHE");  if (hCaloHadEtInHE  &&  hCaloHadEtInHE->getRootObject())  hCaloHadEtInHE->Fill(caloHadEtInHE);
      hCaloHadEtInHF = _dbe->get(DirName+"/"+"METTask_CaloHadEtInHF");  if (hCaloHadEtInHF  &&  hCaloHadEtInHF->getRootObject())  hCaloHadEtInHF->Fill(caloHadEtInHF);
      hCaloEmEtInEB  = _dbe->get(DirName+"/"+"METTask_CaloEmEtInEB");   if (hCaloEmEtInEB   &&  hCaloEmEtInEB->getRootObject())   hCaloEmEtInEB->Fill(caloEmEtInEB);
      hCaloEmEtInEE  = _dbe->get(DirName+"/"+"METTask_CaloEmEtInEE");   if (hCaloEmEtInEE   &&  hCaloEmEtInEE->getRootObject())   hCaloEmEtInEE->Fill(caloEmEtInEE);
      hCaloEmEtInHF  = _dbe->get(DirName+"/"+"METTask_CaloEmEtInHF");   if (hCaloEmEtInHF   &&  hCaloEmEtInHF->getRootObject())   hCaloEmEtInHF->Fill(caloEmEtInHF);

      hCaloEmMEx    = _dbe->get(DirName+"/"+"METTask_CaloEmMEx");     if (hCaloEmMEx    && hCaloEmMEx->getRootObject())     hCaloEmMEx->Fill(_EmMEx);
      hCaloEmMEy    = _dbe->get(DirName+"/"+"METTask_CaloEmMEy");     if (hCaloEmMEy    && hCaloEmMEy->getRootObject())     hCaloEmMEy->Fill(_EmMEy);
      //hCaloEmEz     = _dbe->get(DirName+"/"+"METTask_CaloEmEz");      if (hCaloEmEz     && hCaloEmEz->getRootObject())      hCaloEmEz->Fill(_EmCaloEz);
      hCaloEmMET    = _dbe->get(DirName+"/"+"METTask_CaloEmMET");     if (hCaloEmMET    && hCaloEmMET->getRootObject())     hCaloEmMET->Fill(_EmMET);
      hCaloEmMETPhi = _dbe->get(DirName+"/"+"METTask_CaloEmMETPhi");  if (hCaloEmMETPhi && hCaloEmMETPhi->getRootObject())  hCaloEmMETPhi->Fill(_EmMetPhi);
      //hCaloEmSumET  = _dbe->get(DirName+"/"+"METTask_CaloEmSumET");   if (hCaloEmSumET  && hCaloEmSumET->getRootObject())   hCaloEmSumET->Fill(_EmSumEt);

      hCaloHaMEx    = _dbe->get(DirName+"/"+"METTask_CaloHaMEx");     if (hCaloHaMEx    && hCaloHaMEx->getRootObject())     hCaloHaMEx->Fill(_HaMEx);
      hCaloHaMEy    = _dbe->get(DirName+"/"+"METTask_CaloHaMEy");     if (hCaloHaMEy    && hCaloHaMEy->getRootObject())     hCaloHaMEy->Fill(_HaMEy);
      //hCaloHaEz     = _dbe->get(DirName+"/"+"METTask_CaloHaEz");      if (hCaloHaEz     && hCaloHaEz->getRootObject())      hCaloHaEz->Fill(_HaCaloEz);
      hCaloHaMET    = _dbe->get(DirName+"/"+"METTask_CaloHaMET");     if (hCaloHaMET    && hCaloHaMET->getRootObject())     hCaloHaMET->Fill(_HaMET);
      hCaloHaMETPhi = _dbe->get(DirName+"/"+"METTask_CaloHaMETPhi");  if (hCaloHaMETPhi && hCaloHaMETPhi->getRootObject())  hCaloHaMETPhi->Fill(_HaMetPhi);
      //hCaloHaSumET  = _dbe->get(DirName+"/"+"METTask_CaloHaSumET");   if (hCaloHaSumET  && hCaloHaSumET->getRootObject())   hCaloHaSumET->Fill(_HaSumEt);

    } // _allhist
    /*
    if (theCaloMETCollectionLabel.label() == "corMetGlobalMuons" ) {

      for( reco::MuonCollection::const_iterator muonit = muon_h->begin(); muonit != muon_h->end(); muonit++ ) {
	const reco::TrackRef siTrack = muonit->innerTrack();
	hCalomuPt    = _dbe->get(DirName+"/"+"METTask_CalomuPt");     if (hCalomuPt    && hCalomuPt->getRootObject())     hCalomuPt->Fill( muonit->p4().pt() );
	hCalomuEta   = _dbe->get(DirName+"/"+"METTask_CalomuEta");    if (hCalomuEta   && hCalomuEta->getRootObject())    hCalomuEta->Fill( muonit->p4().eta() );
	hCalomuNhits = _dbe->get(DirName+"/"+"METTask_CalomuNhits");  if (hCalomuNhits && hCalomuNhits->getRootObject())  hCalomuNhits->Fill( siTrack.isNonnull() ? siTrack->numberOfValidHits() : -999 );
	hCalomuChi2  = _dbe->get(DirName+"/"+"METTask_CalomuChi2");   if (hCalomuChi2  && hCalomuChi2->getRootObject())   hCalomuChi2->Fill( siTrack.isNonnull() ? siTrack->chi2()/siTrack->ndof() : -999 );
	double d0 = siTrack.isNonnull() ? -1 * siTrack->dxy( bspot) : -999;
	hCalomuD0    = _dbe->get(DirName+"/"+"METTask_CalomuD0");     if (hCalomuD0    && hCalomuD0->getRootObject())  hCalomuD0->Fill( d0 );
      }
      
      const unsigned int nMuons = muon_h->size();      
      for( unsigned int mus = 0; mus < nMuons; mus++ ) {
	reco::MuonRef muref( muon_h, mus);
	reco::MuonMETCorrectionData muCorrData = (*corMetGlobalMuons_ValueMap_Handle)[muref];
 	hCaloMExCorrection      = _dbe->get(DirName+"/"+"METTask_CaloMExCorrection");       if (hCaloMExCorrection      && hCaloMExCorrection->getRootObject())       hCaloMExCorrection-> Fill(muCorrData.corrY());
 	hCaloMEyCorrection      = _dbe->get(DirName+"/"+"METTask_CaloMEyCorrection");       if (hCaloMEyCorrection      && hCaloMEyCorrection->getRootObject())       hCaloMEyCorrection-> Fill(muCorrData.corrX());
 	hCaloMuonCorrectionFlag = _dbe->get(DirName+"/"+"METTask_CaloMuonCorrectionFlag");  if (hCaloMuonCorrectionFlag && hCaloMuonCorrectionFlag->getRootObject())  hCaloMuonCorrectionFlag-> Fill(muCorrData.type());
      }
    }    
    */
  } // et threshold cut

}

// ***********************************************************
void CaloMETAnalyzer::fillMonitorElementTriggered(const edm::Event& iEvent, std::string DirName, 
						  std::string TriggerTypeName, 
						  const reco::CaloMET& calomet,
						  const reco::CaloMET& calometnohf,
						  bool bLumiSecPlot)
{

  if (TriggerTypeName=="HighPtJet") {
    if (!selectHighPtJetEvent(iEvent)) return;
  }
  else if (TriggerTypeName=="LowPtJet") {
    if (!selectLowPtJetEvent(iEvent)) return;
  }
  else if (TriggerTypeName=="HighMET") {
    if (calomet.pt()<_highMETThreshold) return;
  }
  //  else if (TriggerTypeName=="LowMET") {
  //    if (calomet.pt()<_lowMETThreshold) return;
  //  }
  else if (TriggerTypeName=="Ele") {
    if (!selectWElectronEvent(iEvent)) return;
  }
  else if (TriggerTypeName=="Muon") {
    if (!selectWMuonEvent(iEvent)) return;
  }

  double caloSumET  = calomet.sumEt();
  double caloMETSig = calomet.mEtSig();
  //double caloEz     = calomet.e_longitudinal();
  double caloMET    = calomet.pt();
  double caloMEx    = calomet.px();
  double caloMEy    = calomet.py();
  double caloMETPhi = calomet.phi();

  double caloMETNoHF    = calometnohf.pt();
  //double caloMETNoHFPhi = calometnohf.phi();

  if (_verbose) std::cout << _source << " " << caloMET << std::endl;

  double caloEtFractionHadronic = calomet.etFractionHadronic();
  double caloEmEtFraction       = calomet.emEtFraction();

  //double caloMaxEtInEMTowers    = calomet.maxEtInEmTowers();
  //double caloMaxEtInHadTowers   = calomet.maxEtInHadTowers();

  //double caloHadEtInHB = calomet.hadEtInHB();
  //double caloHadEtInHO = calomet.hadEtInHO();
  //double caloHadEtInHE = calomet.hadEtInHE();
  //double caloHadEtInHF = calomet.hadEtInHF();
  //double caloEmEtInEB  = calomet.emEtInEB();
  //double caloEmEtInEE  = calomet.emEtInEE();
  //double caloEmEtInHF  = calomet.emEtInHF();

  //
  int myLuminosityBlock;
  //  myLuminosityBlock = (evtCounter++)/1000;
  myLuminosityBlock = iEvent.luminosityBlock();
  //

  if (TriggerTypeName!="") DirName = DirName +"/"+TriggerTypeName;

  if (_verbose) std::cout << "_etThreshold = " << _etThreshold << std::endl;
  if (caloSumET>_etThreshold){
    hCaloMEx    = _dbe->get(DirName+"/"+"METTask_CaloMEx");    if (hCaloMEx     && hCaloMEx->getRootObject() )    hCaloMEx->Fill(caloMEx);
    hCaloMEy    = _dbe->get(DirName+"/"+"METTask_CaloMEy");    if (hCaloMEy     && hCaloMEy->getRootObject() )    hCaloMEy->Fill(caloMEy);
    hCaloMET    = _dbe->get(DirName+"/"+"METTask_CaloMET");    if (hCaloMET     && hCaloMET->getRootObject() )    hCaloMET->Fill(caloMET);
    hCaloMET1   = _dbe->get(DirName+"/"+"METTask_CaloMET1");   if (hCaloMET1    && hCaloMET1->getRootObject() )   hCaloMET1->Fill(caloMET);
    hCaloMETPhi = _dbe->get(DirName+"/"+"METTask_CaloMETPhi"); if (hCaloMETPhi  && hCaloMETPhi->getRootObject() ) hCaloMETPhi->Fill(caloMETPhi);
    hCaloSumET  = _dbe->get(DirName+"/"+"METTask_CaloSumET");  if (hCaloSumET   && hCaloSumET->getRootObject() )  hCaloSumET->Fill(caloSumET);
    hCaloMETSig = _dbe->get(DirName+"/"+"METTask_CaloMETSig"); if (hCaloMETSig  && hCaloMETSig->getRootObject() ) hCaloMETSig->Fill(caloMETSig);
    //hCaloEz     = _dbe->get(DirName+"/"+"METTask_CaloEz");     if (hCaloEz      && hCaloEz->getRootObject() )     hCaloEz->Fill(caloEz);

    hCaloMETNoHF    = _dbe->get(DirName+"/"+"METTask_CaloMETNoHF");    if (hCaloMETNoHF     && hCaloMETNoHF->getRootObject() )    hCaloMETNoHF->Fill(caloMETNoHF);

    hCaloMET_logx   = _dbe->get(DirName+"/"+"METTask_CaloMET_logx");      if (hCaloMET_logx    && hCaloMET_logx->getRootObject() )   hCaloMET_logx->Fill(log10(caloMET));
    hCaloSumET_logx = _dbe->get(DirName+"/"+"METTask_CaloSumET_logx");    if (hCaloSumET_logx  && hCaloSumET_logx->getRootObject() ) hCaloSumET_logx->Fill(log10(caloSumET));


    // Fill NPV profiles
    //--------------------------------------------------------------------------
    hCaloMEx_profile     = _dbe->get(DirName + "/METTask_CaloMEx_profile");
    hCaloMEy_profile     = _dbe->get(DirName + "/METTask_CaloMEy_profile");
    hCaloMET_profile     = _dbe->get(DirName + "/METTask_CaloMET_profile");
    hCaloMETNoHF_profile = _dbe->get(DirName + "/METTask_CaloMETNoHF_profile");
    hCaloSumET_profile   = _dbe->get(DirName + "/METTask_CaloSumET_profile");
    
    if (hCaloMEx_profile     && hCaloMEx_profile    ->getRootObject()) hCaloMEx_profile    ->Fill(_numPV, caloMEx);
    if (hCaloMEy_profile     && hCaloMEy_profile    ->getRootObject()) hCaloMEy_profile    ->Fill(_numPV, caloMEy);
    if (hCaloMET_profile     && hCaloMET_profile    ->getRootObject()) hCaloMET_profile    ->Fill(_numPV, caloMET);
    if (hCaloMETNoHF_profile && hCaloMETNoHF_profile->getRootObject()) hCaloMETNoHF_profile->Fill(_numPV, caloMETNoHF);
    if (hCaloSumET_profile   && hCaloSumET_profile  ->getRootObject()) hCaloSumET_profile  ->Fill(_numPV, caloSumET);


    //hCaloEz     = _dbe->get(DirName+"/"+"METTask_CaloEz_profile");     if (hCaloEz      && hCaloEz_profile->getRootObject() )     hCaloEz_profile->Fill(caloEz);
    //  hCaloMETIonFeedbck = _dbe->get(DirName+"/"+"METTask_CaloMETIonFeedbck"); if (hCaloMETIonFeedbck  && hCaloMETIonFeedbck->getRootObject() ) hCaloMETIonFeedbck->Fill(caloMET);
    //  haloMETHPDNoise   = _dbe->get(DirName+"/"+"METTask_CaloMETHPDNoise");   if (hCaloMETHPDNoise    && hCaloMETHPDNoise->getRootObject() )   hCaloMETHPDNoise->Fill(caloMET);

    //hCaloMETPhi002 = _dbe->get(DirName+"/"+"METTask_CaloMETPhi002");    if (caloMET>  2. && hCaloMETPhi002  &&  hCaloMETPhi002->getRootObject()) { hCaloMETPhi002->Fill(caloMETPhi);}
    //hCaloMETPhi010 = _dbe->get(DirName+"/"+"METTask_CaloMETPhi010");    if (caloMET> 10. && hCaloMETPhi010  &&  hCaloMETPhi010->getRootObject()) { hCaloMETPhi010->Fill(caloMETPhi);}
    hCaloMETPhi020 = _dbe->get(DirName+"/"+"METTask_CaloMETPhi020");    if (caloMET> 20. && hCaloMETPhi020  &&  hCaloMETPhi020->getRootObject()) { hCaloMETPhi020->Fill(caloMETPhi);}

    if (_allhist){
      if (bLumiSecPlot){
	hCaloMExLS = _dbe->get(DirName+"/"+"METTask_CaloMEx_LS");   if (hCaloMExLS  &&  hCaloMExLS->getRootObject())    hCaloMExLS->Fill(caloMEx,myLuminosityBlock);
	hCaloMEyLS = _dbe->get(DirName+"/"+"METTask_CaloMEy_LS");   if (hCaloMEyLS  &&  hCaloMEyLS->getRootObject())    hCaloMEyLS->Fill(caloMEy,myLuminosityBlock);
      }
      
      hCaloEtFractionHadronic = _dbe->get(DirName+"/"+"METTask_CaloEtFractionHadronic"); if (hCaloEtFractionHadronic && hCaloEtFractionHadronic->getRootObject())  hCaloEtFractionHadronic->Fill(caloEtFractionHadronic);
      hCaloEmEtFraction       = _dbe->get(DirName+"/"+"METTask_CaloEmEtFraction");       if (hCaloEmEtFraction       && hCaloEmEtFraction->getRootObject())        hCaloEmEtFraction->Fill(caloEmEtFraction);
      
      //hCaloEmEtFraction002 = _dbe->get(DirName+"/"+"METTask_CaloEmEtFraction002");       if (caloMET>  2.  &&  hCaloEmEtFraction002    && hCaloEmEtFraction002->getRootObject()) hCaloEmEtFraction002->Fill(caloEmEtFraction);
      //hCaloEmEtFraction010 = _dbe->get(DirName+"/"+"METTask_CaloEmEtFraction010");       if (caloMET> 10.  &&  hCaloEmEtFraction010    && hCaloEmEtFraction010->getRootObject()) hCaloEmEtFraction010->Fill(caloEmEtFraction);
      hCaloEmEtFraction020 = _dbe->get(DirName+"/"+"METTask_CaloEmEtFraction020");       if (caloMET> 20.  &&  hCaloEmEtFraction020    && hCaloEmEtFraction020->getRootObject()) hCaloEmEtFraction020->Fill(caloEmEtFraction);

      //hCaloMaxEtInEmTowers  = _dbe->get(DirName+"/"+"METTask_CaloMaxEtInEmTowers");   if (hCaloMaxEtInEmTowers  && hCaloMaxEtInEmTowers->getRootObject())   hCaloMaxEtInEmTowers->Fill(caloMaxEtInEMTowers);
      //hCaloMaxEtInHadTowers = _dbe->get(DirName+"/"+"METTask_CaloMaxEtInHadTowers");  if (hCaloMaxEtInHadTowers && hCaloMaxEtInHadTowers->getRootObject())  hCaloMaxEtInHadTowers->Fill(caloMaxEtInHadTowers);
      //
      //hCaloHadEtInHB = _dbe->get(DirName+"/"+"METTask_CaloHadEtInHB");  if (hCaloHadEtInHB  &&  hCaloHadEtInHB->getRootObject())  hCaloHadEtInHB->Fill(caloHadEtInHB);
      //hCaloHadEtInHO = _dbe->get(DirName+"/"+"METTask_CaloHadEtInHO");  if (hCaloHadEtInHO  &&  hCaloHadEtInHO->getRootObject())  hCaloHadEtInHO->Fill(caloHadEtInHO);
      //hCaloHadEtInHE = _dbe->get(DirName+"/"+"METTask_CaloHadEtInHE");  if (hCaloHadEtInHE  &&  hCaloHadEtInHE->getRootObject())  hCaloHadEtInHE->Fill(caloHadEtInHE);
      //hCaloHadEtInHF = _dbe->get(DirName+"/"+"METTask_CaloHadEtInHF");  if (hCaloHadEtInHF  &&  hCaloHadEtInHF->getRootObject())  hCaloHadEtInHF->Fill(caloHadEtInHF);
      //hCaloEmEtInEB  = _dbe->get(DirName+"/"+"METTask_CaloEmEtInEB");   if (hCaloEmEtInEB   &&  hCaloEmEtInEB->getRootObject())   hCaloEmEtInEB->Fill(caloEmEtInEB);
      //hCaloEmEtInEE  = _dbe->get(DirName+"/"+"METTask_CaloEmEtInEE");   if (hCaloEmEtInEE   &&  hCaloEmEtInEE->getRootObject())   hCaloEmEtInEE->Fill(caloEmEtInEE);
      //hCaloEmEtInHF  = _dbe->get(DirName+"/"+"METTask_CaloEmEtInHF");   if (hCaloEmEtInHF   &&  hCaloEmEtInHF->getRootObject())   hCaloEmEtInHF->Fill(caloEmEtInHF);
      //
      //hCaloEmMEx    = _dbe->get(DirName+"/"+"METTask_CaloEmMEx");     if (hCaloEmMEx    && hCaloEmMEx->getRootObject())     hCaloEmMEx->Fill(_EmMEx);
      //hCaloEmMEy    = _dbe->get(DirName+"/"+"METTask_CaloEmMEy");     if (hCaloEmMEy    && hCaloEmMEy->getRootObject())     hCaloEmMEy->Fill(_EmMEy);
      ////hCaloEmEz     = _dbe->get(DirName+"/"+"METTask_CaloEmEz");      if (hCaloEmEz     && hCaloEmEz->getRootObject())      hCaloEmEz->Fill(_EmCaloEz);
      //hCaloEmMET    = _dbe->get(DirName+"/"+"METTask_CaloEmMET");     if (hCaloEmMET    && hCaloEmMET->getRootObject())     hCaloEmMET->Fill(_EmMET);
      //hCaloEmMETPhi = _dbe->get(DirName+"/"+"METTask_CaloEmMETPhi");  if (hCaloEmMETPhi && hCaloEmMETPhi->getRootObject())  hCaloEmMETPhi->Fill(_EmMetPhi);
      ////hCaloEmSumET  = _dbe->get(DirName+"/"+"METTask_CaloEmSumET");   if (hCaloEmSumET  && hCaloEmSumET->getRootObject())   hCaloEmSumET->Fill(_EmSumEt);
      //
      //hCaloHaMEx    = _dbe->get(DirName+"/"+"METTask_CaloHaMEx");     if (hCaloHaMEx    && hCaloHaMEx->getRootObject())     hCaloHaMEx->Fill(_HaMEx);
      //hCaloHaMEy    = _dbe->get(DirName+"/"+"METTask_CaloHaMEy");     if (hCaloHaMEy    && hCaloHaMEy->getRootObject())     hCaloHaMEy->Fill(_HaMEy);
      ////hCaloHaEz     = _dbe->get(DirName+"/"+"METTask_CaloHaEz");      if (hCaloHaEz     && hCaloHaEz->getRootObject())      hCaloHaEz->Fill(_HaCaloEz);
      //hCaloHaMET    = _dbe->get(DirName+"/"+"METTask_CaloHaMET");     if (hCaloHaMET    && hCaloHaMET->getRootObject())     hCaloHaMET->Fill(_HaMET);
      //hCaloHaMETPhi = _dbe->get(DirName+"/"+"METTask_CaloHaMETPhi");  if (hCaloHaMETPhi && hCaloHaMETPhi->getRootObject())  hCaloHaMETPhi->Fill(_HaMetPhi);
      ////hCaloHaSumET  = _dbe->get(DirName+"/"+"METTask_CaloHaSumET");   if (hCaloHaSumET  && hCaloHaSumET->getRootObject())   hCaloHaSumET->Fill(_HaSumEt);

    } // _allhist
    if (theCaloMETCollectionLabel.label() == "corMetGlobalMuons" ) {

      for( reco::MuonCollection::const_iterator muonit = muon_h->begin(); muonit != muon_h->end(); muonit++ ) {
	const reco::TrackRef siTrack = muonit->innerTrack();
	hCalomuPt    = _dbe->get(DirName+"/"+"METTask_CalomuPt");     if (hCalomuPt    && hCalomuPt->getRootObject())     hCalomuPt->Fill( muonit->p4().pt() );
	hCalomuEta   = _dbe->get(DirName+"/"+"METTask_CalomuEta");    if (hCalomuEta   && hCalomuEta->getRootObject())    hCalomuEta->Fill( muonit->p4().eta() );
	hCalomuNhits = _dbe->get(DirName+"/"+"METTask_CalomuNhits");  if (hCalomuNhits && hCalomuNhits->getRootObject())  hCalomuNhits->Fill( siTrack.isNonnull() ? siTrack->numberOfValidHits() : -999 );
	hCalomuChi2  = _dbe->get(DirName+"/"+"METTask_CalomuChi2");   if (hCalomuChi2  && hCalomuChi2->getRootObject())   hCalomuChi2->Fill( siTrack.isNonnull() ? siTrack->chi2()/siTrack->ndof() : -999 );
	double d0 = siTrack.isNonnull() ? -1 * siTrack->dxy( bspot) : -999;
	hCalomuD0    = _dbe->get(DirName+"/"+"METTask_CalomuD0");     if (hCalomuD0    && hCalomuD0->getRootObject())  hCalomuD0->Fill( d0 );
      }
      
      const unsigned int nMuons = muon_h->size();      
      for( unsigned int mus = 0; mus < nMuons; mus++ ) {
	reco::MuonRef muref( muon_h, mus);
	reco::MuonMETCorrectionData muCorrData = (*corMetGlobalMuons_ValueMap_Handle)[muref];
 	hCaloMExCorrection      = _dbe->get(DirName+"/"+"METTask_CaloMExCorrection");       if (hCaloMExCorrection      && hCaloMExCorrection->getRootObject())       hCaloMExCorrection-> Fill(muCorrData.corrY());
 	hCaloMEyCorrection      = _dbe->get(DirName+"/"+"METTask_CaloMEyCorrection");       if (hCaloMEyCorrection      && hCaloMEyCorrection->getRootObject())       hCaloMEyCorrection-> Fill(muCorrData.corrX());
 	hCaloMuonCorrectionFlag = _dbe->get(DirName+"/"+"METTask_CaloMuonCorrectionFlag");  if (hCaloMuonCorrectionFlag && hCaloMuonCorrectionFlag->getRootObject())  hCaloMuonCorrectionFlag-> Fill(muCorrData.type());
      }
    }    
  } // et threshold cut

}

// ***********************************************************
bool CaloMETAnalyzer::selectHighPtJetEvent(const edm::Event& iEvent){

  bool return_value=false;

  edm::Handle<reco::CaloJetCollection> caloJets;
  iEvent.getByLabel(theJetCollectionLabel, caloJets);
  if (!caloJets.isValid()) {
    LogDebug("") << "CaloMETAnalyzer: Could not find jet product" << std::endl;
    if (_verbose) std::cout << "CaloMETAnalyzer: Could not find jet product" << std::endl;
  }

  for (reco::CaloJetCollection::const_iterator cal = caloJets->begin(); 
       cal!=caloJets->end(); ++cal){
    if (cal->pt()>_highPtJetThreshold){
      return_value=true;
    }
  }

  return return_value;

}

// ***********************************************************
bool CaloMETAnalyzer::selectLowPtJetEvent(const edm::Event& iEvent){

  bool return_value=false;

  edm::Handle<reco::CaloJetCollection> caloJets;
  iEvent.getByLabel(theJetCollectionLabel, caloJets);
  if (!caloJets.isValid()) {
    LogDebug("") << "CaloMETAnalyzer: Could not find jet product" << std::endl;
    if (_verbose) std::cout << "CaloMETAnalyzer: Could not find jet product" << std::endl;
  }

  for (reco::CaloJetCollection::const_iterator cal = caloJets->begin(); 
       cal!=caloJets->end(); ++cal){
    if (cal->pt()>_lowPtJetThreshold){
      return_value=true;
    }
  }

  return return_value;

}

// ***********************************************************
bool CaloMETAnalyzer::selectWElectronEvent(const edm::Event& iEvent){

  bool return_value=true;

  /*
    W-electron event selection comes here
   */

  return return_value;

}

// ***********************************************************
bool CaloMETAnalyzer::selectWMuonEvent(const edm::Event& iEvent){

  bool return_value=true;

  /*
    W-muon event selection comes here
   */

  return return_value;

}


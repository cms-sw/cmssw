/*
 *  See header file for a description of this class.
 *
 *  \author A.Apresyan - Caltech
 *          K.Hatakeyama - Baylor
 */

#include "DQMOffline/JetMET/interface/METAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include <string>
using namespace edm;
using namespace reco;
using namespace math;

// ***********************************************************
METAnalyzer::METAnalyzer(const edm::ParameterSet& pSet) {

  parameters = pSet;

  edm::ParameterSet highptjetparms = parameters.getParameter<edm::ParameterSet>("highPtJetTrigger");
  edm::ParameterSet lowptjetparms  = parameters.getParameter<edm::ParameterSet>("lowPtJetTrigger" );
  edm::ParameterSet minbiasparms   = parameters.getParameter<edm::ParameterSet>("minBiasTrigger"  );
  edm::ParameterSet highmetparms   = parameters.getParameter<edm::ParameterSet>("highMETTrigger"  );
  //  edm::ParameterSet lowmetparms    = parameters.getParameter<edm::ParameterSet>("lowMETTrigger"   );
  edm::ParameterSet eleparms       = parameters.getParameter<edm::ParameterSet>("eleTrigger"      );
  edm::ParameterSet muonparms      = parameters.getParameter<edm::ParameterSet>("muonTrigger"     );

  mOutputFile   = parameters.getParameter<std::string>("OutputFile");
  MetType = parameters.getUntrackedParameter<std::string>("METType");

  theTriggerResultsLabel        = parameters.getParameter<edm::InputTag>("TriggerResultsLabel");
  triggerResultsToken_= consumes<edm::TriggerResults>(edm::InputTag(theTriggerResultsLabel));

  isCaloMet = (std::string("calo")==MetType);
  isTCMet = (std::string("tc") ==MetType);
  isPFMet = (std::string("pf") ==MetType);

  // MET information
  theMETCollectionLabel       = parameters.getParameter<edm::InputTag>("METCollectionLabel");

  if(isPFMet){
    pfMetToken_= consumes<reco::PFMETCollection>(edm::InputTag(theMETCollectionLabel));
  }
 if(isCaloMet){
    caloMetToken_= consumes<reco::CaloMETCollection>(edm::InputTag(theMETCollectionLabel));
  }
 if(isTCMet){
    tcMetToken_= consumes<reco::METCollection>(edm::InputTag(theMETCollectionLabel));
  }

  //jet cleanup parameters
  theCleaningParameters = pSet.getParameter<ParameterSet>("CleaningParameters");

  //Vertex requirements
  _doPVCheck          = theCleaningParameters.getParameter<bool>("doPrimaryVertexCheck");
  vertexTag  = theCleaningParameters.getParameter<edm::InputTag>("vertexLabel");
  vertexToken_= consumes<std::vector<reco::Vertex> >(edm::InputTag(vertexTag));

  //Trigger parameters
  gtTag          = theCleaningParameters.getParameter<edm::InputTag>("gtLabel");
  gtToken_= consumes<L1GlobalTriggerReadoutRecord>(edm::InputTag(gtTag));

  inputTrackLabel         = parameters.getParameter<edm::InputTag>("InputTrackLabel");
  inputMuonLabel          = parameters.getParameter<edm::InputTag>("InputMuonLabel");
  inputElectronLabel      = parameters.getParameter<edm::InputTag>("InputElectronLabel");
  inputBeamSpotLabel      = parameters.getParameter<edm::InputTag>("InputBeamSpotLabel");
  inputTCMETValueMap      = parameters.getParameter<edm::InputTag>("InputTCMETValueMap");
  TrackToken_= consumes<edm::View <reco::Track> >(inputTrackLabel);
  MuonToken_= consumes<reco::MuonCollection>(inputMuonLabel);
  ElectronToken_= consumes<edm::View<reco::GsfElectron> >(inputElectronLabel);
  BeamspotToken_= consumes<reco::BeamSpot>(inputBeamSpotLabel);
  tcMET_ValueMapToken_= consumes< edm::ValueMap<reco::MuonMETCorrectionData> >(inputTCMETValueMap);
 



  // Other data collections
  theJetCollectionLabel       = parameters.getParameter<edm::InputTag>("JetCollectionLabel");
  if (isCaloMet) caloJetsToken_ = consumes<reco::CaloJetCollection>(theJetCollectionLabel);
  if (isTCMet) jptJetsToken_ = consumes<reco::JPTJetCollection>(theJetCollectionLabel);
  if (isPFMet) pfJetsToken_ = consumes<reco::PFJetCollection>(theJetCollectionLabel);


  HcalNoiseRBXCollectionTag   = parameters.getParameter<edm::InputTag>("HcalNoiseRBXCollection");
  HcalNoiseRBXToken_ = consumes<reco::HcalNoiseRBXCollection>(HcalNoiseRBXCollectionTag);

  BeamHaloSummaryTag          = parameters.getParameter<edm::InputTag>("BeamHaloSummaryLabel");
  BeamHaloSummaryToken_       = consumes<BeamHaloSummary>(BeamHaloSummaryTag); 
  HBHENoiseFilterResultTag    = parameters.getParameter<edm::InputTag>("HBHENoiseFilterResultLabel");
  HBHENoiseFilterResultToken_=consumes<bool>(HBHENoiseFilterResultTag);

  // 
  nbinsPV = parameters.getParameter<int>("pVBin");
  PVlow   = parameters.getParameter<double>("pVMin");
  PVup  = parameters.getParameter<double>("pVMax");

  //genericTriggerEventFlag_( new GenericTriggerEventFlag( conf_, consumesCollector() ) );
  _HighPtJetEventFlag = new GenericTriggerEventFlag( highptjetparms, consumesCollector() );
  _LowPtJetEventFlag  = new GenericTriggerEventFlag( lowptjetparms, consumesCollector() );
  _MinBiasEventFlag   = new GenericTriggerEventFlag( minbiasparms , consumesCollector() );
  _HighMETEventFlag   = new GenericTriggerEventFlag( highmetparms , consumesCollector() );
  //  _LowMETEventFlag    = new GenericTriggerEventFlag( lowmetparms  , consumesCollector() );
  _EleEventFlag       = new GenericTriggerEventFlag( eleparms     , consumesCollector() );
  _MuonEventFlag      = new GenericTriggerEventFlag( muonparms    , consumesCollector() );

  highPtJetExpr_ = highptjetparms.getParameter<std::vector<std::string> >("hltPaths");
  lowPtJetExpr_  = lowptjetparms .getParameter<std::vector<std::string> >("hltPaths");
  highMETExpr_   = highmetparms  .getParameter<std::vector<std::string> >("hltPaths");
  //  lowMETExpr_    = lowmetparms   .getParameter<std::vector<std::string> >("hltPaths");
  muonExpr_      = muonparms     .getParameter<std::vector<std::string> >("hltPaths");
  elecExpr_      = eleparms      .getParameter<std::vector<std::string> >("hltPaths");
  minbiasExpr_   = minbiasparms  .getParameter<std::vector<std::string> >("hltPaths");




}

// ***********************************************************
METAnalyzer::~METAnalyzer() {

  delete _HighPtJetEventFlag;
  delete _LowPtJetEventFlag;
  delete _MinBiasEventFlag;
  delete _HighMETEventFlag;
  //  delete _LowMETEventFlag;
  delete _EleEventFlag;
  delete _MuonEventFlag;

}

void METAnalyzer::beginJob(/*MStore * dbe*/){

  evtCounter = 0;

  // trigger information
  HLTPathsJetMBByName_ = parameters.getParameter<std::vector<std::string > >("HLTPathsJetMB");

  theCleaningParameters = parameters.getParameter<ParameterSet>("CleaningParameters"),

  //_theGTLabel         = theCleaningParameters.getParameter<edm::InputTag>("gtLabel");
  //gtToken_= consumes<L1GlobalTriggerReadoutRecord>(edm::InputTag(_theGTLabel));

  _doHLTPhysicsOn = theCleaningParameters.getParameter<bool>("doHLTPhysicsOn");
  //_hlt_PhysDec    = theCleaningParameters.getParameter<std::string>("HLT_PhysDec");

  _tightBHFiltering     = theCleaningParameters.getParameter<bool>("tightBHFiltering");
  _tightJetIDFiltering  = theCleaningParameters.getParameter<int>("tightJetIDFiltering");

  // ==========================================================
  //DCS information
  // ==========================================================
  DCSFilter = new JetMETDQMDCSFilter(parameters.getParameter<ParameterSet>("DCSFilter"), iC);


  if (_doPVCheck) {
    _nvtx_min        = theCleaningParameters.getParameter<int>("nvtx_min");
    _nvtxtrks_min    = theCleaningParameters.getParameter<int>("nvtxtrks_min");
    _vtxndof_min     = theCleaningParameters.getParameter<int>("vtxndof_min");
    _vtxchi2_max     = theCleaningParameters.getParameter<double>("vtxchi2_max");
    _vtxz_max        = theCleaningParameters.getParameter<double>("vtxz_max");
  }

  // misc
  _verbose      = parameters.getParameter<int>("verbose");
  _etThreshold  = parameters.getParameter<double>("etThreshold"); // MET threshold
  _allhist      = parameters.getParameter<bool>("allHist");       // Full set of monitoring histograms
  _allSelection = parameters.getParameter<bool>("allSelection");  // Plot with all sets of event selection
  _cleanupSelection = parameters.getParameter<bool>("cleanupSelection");  // Plot with all sets of event selection

  _FolderName              = parameters.getUntrackedParameter<std::string>("FolderName");

  _highPtJetThreshold = parameters.getParameter<double>("HighPtJetThreshold"); // High Pt Jet threshold
  _lowPtJetThreshold  = parameters.getParameter<double>("LowPtJetThreshold");   // Low Pt Jet threshold
  _highMETThreshold   = parameters.getParameter<double>("HighMETThreshold");     // High MET threshold
  //  _lowMETThreshold    = parameters.getParameter<double>("LowMETThreshold");       // Low MET threshold

  //
  if(isCaloMet || isTCMet){
    jetID = new reco::helper::JetIDHelper(parameters.getParameter<ParameterSet>("JetIDParams"));
  }

  // DQStore stuff
  dbe = edm::Service<DQMStore>().operator->();
  LogTrace(metname)<<"[METAnalyzer] Parameters initialization";
  std::string DirName = "JetMET/MET/"+theMETCollectionLabel.label();
  dbe->setCurrentFolder(DirName);

  hmetME = dbe->book1D("metReco", "metReco", 4, 1, 5);
  if(isTCMet){
    hmetME->setBinLabel(2,"tcMet",1);
  }
 if(isPFMet){
    hmetME->setBinLabel(3,"pfMet",1);
  }
 if(isCaloMet){
    hmetME->setBinLabel(1,"tcMet",1);
  }
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
    if (*ic=="All")                  bookMESet(DirName+"/"+*ic);
    if (_cleanupSelection){
    if (*ic=="BasicCleanup")         bookMESet(DirName+"/"+*ic);
    if (*ic=="ExtraCleanup")         bookMESet(DirName+"/"+*ic);
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

  // MET information



}

// ***********************************************************
void METAnalyzer::endJob() {
  if(isTCMet || isCaloMet){
    delete jetID;
  }
  delete DCSFilter;

 if(!mOutputFile.empty() && &*edm::Service<DQMStore>()){
      //dbe->save(mOutputFile);
    edm::Service<DQMStore>()->save(mOutputFile);
  }

}

// ***********************************************************
void METAnalyzer::bookMESet(std::string DirName)
{

  bool bLumiSecPlot=false;
  if (DirName.find("All")!=std::string::npos) bLumiSecPlot=true;

  bookMonitorElement(DirName,bLumiSecPlot);

  if ( _HighPtJetEventFlag->on() ) {
    bookMonitorElement(DirName+"/"+"HighPtJet",false);
    hTriggerName_HighPtJet = _dbe->bookString("triggerName_HighPtJet", highPtJetExpr_[0]);
  }

  if ( _LowPtJetEventFlag->on() ) {
    bookMonitorElement(DirName+"/"+"LowPtJet",false);
    hTriggerName_LowPtJet = _dbe->bookString("triggerName_LowPtJet", lowPtJetExpr_[0]);
  }

  if ( _MinBiasEventFlag->on() ) {
    bookMonitorElement(DirName+"/"+"MinBias",false);
    hTriggerName_MinBias = _dbe->bookString("triggerName_MinBias", minbiasExpr_[0]);
    if (_verbose) std::cout << "_MinBiasEventFlag is on, folder created\n";
  }

  if ( _HighMETEventFlag->on() ) {
    bookMonitorElement(DirName+"/"+"HighMET",false);
    hTriggerName_HighMET = _dbe->bookString("triggerName_HighMET", highMETExpr_[0]);
  }

  //  if ( _LowMETEventFlag->on() ) {
  //    bookMonitorElement(DirName+"/"+"LowMET",false);
  //    hTriggerName_LowMET = _dbe->bookString("triggerName_LowMET", lowMETExpr_[0]);
  //  }

  if ( _EleEventFlag->on() ) {
    bookMonitorElement(DirName+"/"+"Ele",false);
    hTriggerName_Ele = _dbe->bookString("triggerName_Ele", elecExpr_[0]);
    if (_verbose) std::cout << "_EleEventFlag is on, folder created\n";
  }

  if ( _MuonEventFlag->on() ) {
    bookMonitorElement(DirName+"/"+"Muon",false);
    hTriggerName_Muon = _dbe->bookString("triggerName_Muon", muonExpr_[0]);
    if (_verbose) std::cout << "_MuonEventFlag is on, folder created\n";
  }
}

// ***********************************************************
void METAnalyzer::bookMonitorElement(std::string DirName, bool bLumiSecPlot=false)
{
  if (_verbose) std::cout << "bookMonitorElement " << DirName << std::endl;

  _dbe->setCurrentFolder(DirName);

  hMEx        = _dbe->book1D("METTask_MEx",        "METTask_MEx",        200, -500,  500);
  hMEy        = _dbe->book1D("METTask_MEy",        "METTask_MEy",        200, -500,  500);
  hMET        = _dbe->book1D("METTask_MET",        "METTask_MET",        200,    0, 1000);
  hSumET      = _dbe->book1D("METTask_SumET",      "METTask_SumET",      400,    0, 4000);
  hMETSig     = _dbe->book1D("METTask_METSig",     "METTask_METSig",      51,    0,   51);
  hMETPhi     = _dbe->book1D("METTask_METPhi",     "METTask_METPhi",      60, -3.2,  3.2);
  hMET_logx   = _dbe->book1D("METTask_MET_logx",   "METTask_MET_logx",    40,   -1,    7);
  hSumET_logx = _dbe->book1D("METTask_SumET_logx", "METTask_SumET_logx",  40,   -1,    7);

  hMEx       ->setAxisTitle("MEx [GeV]",        1);
  hMEy       ->setAxisTitle("MEy [GeV]",        1);
  hMET       ->setAxisTitle("MET [GeV]",        1);
  hSumET     ->setAxisTitle("SumET [GeV]",      1);
  hMETSig    ->setAxisTitle("METSig",       1);
  hMETPhi    ->setAxisTitle("METPhi [rad]",     1);
  hMET_logx  ->setAxisTitle("log(MET) [GeV]",   1);
  hSumET_logx->setAxisTitle("log(SumET) [GeV]", 1);

  // Book NPV profiles --> would some of these profiles be interesting for other MET types too
  //----------------------------------------------------------------------------
  meMEx_profile   = _dbe->bookProfile("METTask_MEx_profile",   "met.px()",    nbinsPV, PVlow, PVup, 200, -500,  500);
  meMEy_profile   = _dbe->bookProfile("METTask_MEy_profile",   "met.py()",    nbinsPV, PVlow, PVup, 200, -500,  500);
  meMET_profile   = _dbe->bookProfile("METTask_MET_profile",   "met.pt()",    nbinsPV, PVlow, PVup, 200,    0, 1000);
  meSumET_profile = _dbe->bookProfile("METTask_SumET_profile", "met.sumEt()", nbinsPV, PVlow, PVup, 400,    0, 4000);
  // Set NPV profiles x-axis title
  //----------------------------------------------------------------------------
  meMEx_profile  ->setAxisTitle("nvtx", 1);
  meMEy_profile  ->setAxisTitle("nvtx", 1);
  meMET_profile  ->setAxisTitle("nvtx", 1);
  meSumET_profile->setAxisTitle("nvtx", 1);
    
  if(isCaloMet){
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

    hCaloMETPhi020  = _dbe->book1D("METTask_CaloMETPhi020",  "METTask_CaloMETPhi020",   60, -3.2,  3.2);
    hCaloMETPhi020 ->setAxisTitle("METPhi [rad] (MET>20 GeV)", 1);

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

    if (theMETCollectionLabel.label() == "corMetGlobalMuons" ) {
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

  if(isPFMet){
    mePhotonEtFraction        = _dbe->book1D("METTask_PfPhotonEtFraction",        "pfmet.photonEtFraction()",         50, 0,    1);
    mePhotonEt                = _dbe->book1D("METTask_PfPhotonEt",                "pfmet.photonEt()",                100, 0, 1000);
    meNeutralHadronEtFraction = _dbe->book1D("METTask_PfNeutralHadronEtFraction", "pfmet.neutralHadronEtFraction()",  50, 0,    1);
    meNeutralHadronEt         = _dbe->book1D("METTask_PfNeutralHadronEt",         "pfmet.neutralHadronEt()",         100, 0, 1000);
    meElectronEtFraction      = _dbe->book1D("METTask_PfElectronEtFraction",      "pfmet.electronEtFraction()",       50, 0,    1);
    meElectronEt              = _dbe->book1D("METTask_PfElectronEt",              "pfmet.electronEt()",              100, 0, 1000);
    meChargedHadronEtFraction = _dbe->book1D("METTask_PfChargedHadronEtFraction", "pfmet.chargedHadronEtFraction()",  50, 0,    1);
    meChargedHadronEt         = _dbe->book1D("METTask_PfChargedHadronEt",         "pfmet.chargedHadronEt()",         100, 0, 1000);
    meMuonEtFraction          = _dbe->book1D("METTask_PfMuonEtFraction",          "pfmet.muonEtFraction()",           50, 0,    1);
    meMuonEt                  = _dbe->book1D("METTask_PfMuonEt",                  "pfmet.muonEt()",                  100, 0, 1000);
    meHFHadronEtFraction      = _dbe->book1D("METTask_PfHFHadronEtFraction",      "pfmet.HFHadronEtFraction()",       50, 0,    1);
    meHFHadronEt              = _dbe->book1D("METTask_PfHFHadronEt",              "pfmet.HFHadronEt()",              100, 0, 1000);
    meHFEMEtFraction          = _dbe->book1D("METTask_PfHFEMEtFraction",          "pfmet.HFEMEtFraction()",           50, 0,    1);
    meHFEMEt                  = _dbe->book1D("METTask_PfHFEMEt",                  "pfmet.HFEMEt()",                  100, 0, 1000);
    
    mePhotonEtFraction_profile        = _dbe->bookProfile("METTask_PfPhotonEtFraction_profile",        "pfmet.photonEtFraction()",        nbinsPV, PVlow, PVup,  50, 0,    1);
    mePhotonEt_profile                = _dbe->bookProfile("METTask_PfPhotonEt_profile",                "pfmet.photonEt()",                nbinsPV, PVlow, PVup, 100, 0, 1000);
    meNeutralHadronEtFraction_profile = _dbe->bookProfile("METTask_PfNeutralHadronEtFraction_profile", "pfmet.neutralHadronEtFraction()", nbinsPV, PVlow, PVup,  50, 0,    1);
    meNeutralHadronEt_profile         = _dbe->bookProfile("METTask_PfNeutralHadronEt_profile",         "pfmet.neutralHadronEt()",         nbinsPV, PVlow, PVup, 100, 0, 1000);
    meElectronEtFraction_profile      = _dbe->bookProfile("METTask_PfElectronEtFraction_profile",      "pfmet.electronEtFraction()",      nbinsPV, PVlow, PVup,  50, 0,    1);
    meElectronEt_profile              = _dbe->bookProfile("METTask_PfElectronEt_profile",              "pfmet.electronEt()",              nbinsPV, PVlow, PVup, 100, 0, 1000);
    meChargedHadronEtFraction_profile = _dbe->bookProfile("METTask_PfChargedHadronEtFraction_profile", "pfmet.chargedHadronEtFraction()", nbinsPV, PVlow, PVup,  50, 0,    1);
    meChargedHadronEt_profile         = _dbe->bookProfile("METTask_PfChargedHadronEt_profile",         "pfmet.chargedHadronEt()",         nbinsPV, PVlow, PVup, 100, 0, 1000);
    meMuonEtFraction_profile          = _dbe->bookProfile("METTask_PfMuonEtFraction_profile",          "pfmet.muonEtFraction()",          nbinsPV, PVlow, PVup,  50, 0,    1);
    meMuonEt_profile                  = _dbe->bookProfile("METTask_PfMuonEt_profile",                  "pfmet.muonEt()",                  nbinsPV, PVlow, PVup, 100, 0, 1000);
    meHFHadronEtFraction_profile      = _dbe->bookProfile("METTask_PfHFHadronEtFraction_profile",      "pfmet.HFHadronEtFraction()",      nbinsPV, PVlow, PVup,  50, 0,    1);
    meHFHadronEt_profile              = _dbe->bookProfile("METTask_PfHFHadronEt_profile",              "pfmet.HFHadronEt()",              nbinsPV, PVlow, PVup, 100, 0, 1000);
    meHFEMEtFraction_profile          = _dbe->bookProfile("METTask_PfHFEMEtFraction_profile",          "pfmet.HFEMEtFraction()",          nbinsPV, PVlow, PVup,  50, 0,    1);
    meHFEMEt_profile                  = _dbe->bookProfile("METTask_PfHFEMEt_profile",                  "pfmet.HFEMEt()",                  nbinsPV, PVlow, PVup, 100, 0, 1000);
    
    mePhotonEtFraction_profile       ->setAxisTitle("nvtx", 1);
    mePhotonEt_profile               ->setAxisTitle("nvtx", 1);
    meNeutralHadronEtFraction_profile->setAxisTitle("nvtx", 1);
    meNeutralHadronEt_profile        ->setAxisTitle("nvtx", 1);
    meElectronEtFraction_profile     ->setAxisTitle("nvtx", 1);
    meElectronEt_profile             ->setAxisTitle("nvtx", 1);
    meChargedHadronEtFraction_profile->setAxisTitle("nvtx", 1);
    meChargedHadronEt_profile        ->setAxisTitle("nvtx", 1);
    meMuonEtFraction_profile         ->setAxisTitle("nvtx", 1);
    meMuonEt_profile                 ->setAxisTitle("nvtx", 1);
    meHFHadronEtFraction_profile     ->setAxisTitle("nvtx", 1);
    meHFHadronEt_profile             ->setAxisTitle("nvtx", 1);
    meHFEMEtFraction_profile         ->setAxisTitle("nvtx", 1);
    meHFEMEt_profile                 ->setAxisTitle("nvtx", 1);
  }



  if (_allhist){
    if (bLumiSecPlot){
      hMExLS = _dbe->book2D("METTask_MEx_LS","METTask_MEx_LS",200,-200,200,50,0.,500.);
      hMExLS->setAxisTitle("MEx [GeV]",1);
      hMExLS->setAxisTitle("Lumi Section",2);
      hMEyLS = _dbe->book2D("METTask_MEy_LS","METTask_MEy_LS",200,-200,200,50,0.,500.);
      hMEyLS->setAxisTitle("MEy [GeV]",1);
      hMEyLS->setAxisTitle("Lumi Section",2);
    }
  }

  if (isTCMet) {
    htrkPt    = _dbe->book1D("METTask_trackPt", "METTask_trackPt", 50, 0, 500);
    htrkEta   = _dbe->book1D("METTask_trackEta", "METTask_trackEta", 60, -3.0, 3.0);
    htrkNhits = _dbe->book1D("METTask_trackNhits", "METTask_trackNhits", 50, 0, 50);
    htrkChi2  = _dbe->book1D("METTask_trackNormalizedChi2", "METTask_trackNormalizedChi2", 20, 0, 20);
    htrkD0    = _dbe->book1D("METTask_trackD0", "METTask_trackd0", 50, -1, 1);
    helePt    = _dbe->book1D("METTask_electronPt", "METTask_electronPt", 50, 0, 500);
    heleEta   = _dbe->book1D("METTask_electronEta", "METTask_electronEta", 60, -3.0, 3.0);
    heleHoE   = _dbe->book1D("METTask_electronHoverE", "METTask_electronHoverE", 25, 0, 0.5);
    hmuPt     = _dbe->book1D("METTask_muonPt", "METTask_muonPt", 50, 0, 500);
    hmuEta    = _dbe->book1D("METTask_muonEta", "METTask_muonEta", 60, -3.0, 3.0);
    hmuNhits  = _dbe->book1D("METTask_muonNhits", "METTask_muonNhits", 50, 0, 50);
    hmuChi2   = _dbe->book1D("METTask_muonNormalizedChi2", "METTask_muonNormalizedChi2", 20, 0, 20);
    hmuD0     = _dbe->book1D("METTask_muonD0", "METTask_muonD0", 50, -1, 1);

    hMETIonFeedbck      = _dbe->book1D("METTask_METIonFeedbck", "METTask_METIonFeedbck" ,200,0,1000);
    hMETHPDNoise        = _dbe->book1D("METTask_METHPDNoise",   "METTask_METHPDNoise"   ,200,0,1000);
    hMETRBXNoise        = _dbe->book1D("METTask_METRBXNoise",   "METTask_METRBXNoise"   ,200,0,1000);
    hMExCorrection       = _dbe->book1D("METTask_MExCorrection", "METTask_MExCorrection", 100, -500.0,500.0);
    hMEyCorrection       = _dbe->book1D("METTask_MEyCorrection", "METTask_MEyCorrection", 100, -500.0,500.0);
    hMuonCorrectionFlag  = _dbe->book1D("METTask_CorrectionFlag","METTask_CorrectionFlag", 5, -0.5, 4.5);
  }



}

// ***********************************************************
void METAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
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
void METAnalyzer::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup, DQMStore * dbe)
{

  //
  //--- Check the time length of the Run from the lumi section plots

  std::string dirName = _FolderName+theMETCollectionLabel.label()+"/";
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
  /* check if JPT and PFMET is better
  for (std::vector<std::string>::const_iterator ic = _FolderNames.begin(); ic != _FolderNames.end(); ic++)
    {

      std::string DirName;
      DirName = dirName+*ic;

      makeRatePlot(DirName,totltime);
      if (_hlt_HighPtJet.size()) makeRatePlot(DirName+"/"+_hlt_HighPtJet,totltime);
      if (_hlt_LowPtJet.size())  makeRatePlot(DirName+"/"+_hlt_LowPtJet,totltime);
      if (_hlt_HighMET.size())   makeRatePlot(DirName+"/"+_hlt_HighMET,totltime);
      //      if (_hlt_LowMET.size())    makeRatePlot(DirName+"/"+_hlt_LowMET,totltime);
      if (_hlt_Ele.size())       makeRatePlot(DirName+"/"+_hlt_Ele,totltime);
      if (_hlt_Muon.size())      makeRatePlot(DirName+"/"+_hlt_Muon,totltime);

    }
  */
  //below is the original METAnalyzer formulation
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
void METAnalyzer::makeRatePlot(std::string DirName, double totltime)
{

  _dbe->setCurrentFolder(DirName);
  MonitorElement *meMET = _dbe->get(DirName+"/"+"METTask_MET");

  TH1F* tMET;
  TH1F* tMETRate;

  if ( meMET )
    if ( meMET->getRootObject() ) {
      tMET     = meMET->getTH1F();

      // Integral plot & convert number of events to rate (hz)
      tMETRate = (TH1F*) tMET->Clone("METTask_METRate");
      for (int i = tMETRate->GetNbinsX()-1; i>=0; i--){
	tMETRate->SetBinContent(i+1,tMETRate->GetBinContent(i+2)+tMET->GetBinContent(i+1));
      }
      for (int i = 0; i<tMETRate->GetNbinsX(); i++){
	tMETRate->SetBinContent(i+1,tMETRate->GetBinContent(i+1)/double(totltime));
      }

      tMETRate->SetName("METTask_METRate");
      tMETRate->SetTitle("METTask_METRate");
      hMETRate      = _dbe->book1D("METTask_METRate",tMETRate);
    }
}

// ***********************************************************
void METAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  if (_verbose) std::cout << "METAnalyzer analyze" << std::endl;

  std::string DirName = _FolderName+theMETCollectionLabel.label();


  hmetME->Fill(2);

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

  // **** Get the TriggerResults container
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(triggerResultsToken_, triggerResults);

  if( triggerResults.isValid()) {

    /////////// Analyzing HLT Trigger Results (TriggerResults) //////////

    //
    //
    // Check how many HLT triggers are in triggerResults
    int ntrigs = (*triggerResults).size();
    if (_verbose) std::cout << "ntrigs=" << ntrigs << std::endl;

    //
    //
    // If index=ntrigs, this HLT trigger doesn't exist in the HLT table for this data.
    const edm::TriggerNames & triggerNames = iEvent.triggerNames(*triggerResults);

    //
    //
    const unsigned int nTrig(triggerNames.size());
    for (unsigned int i=0;i<nTrig;++i)
      {
        if (triggerNames.triggerName(i).find(highPtJetExpr_[0].substr(0,highPtJetExpr_[0].rfind("_v")+2))!=std::string::npos && (*triggerResults).accept(i))
	  _trig_HighPtJet=true;
        else if (triggerNames.triggerName(i).find(lowPtJetExpr_[0].substr(0,lowPtJetExpr_[0].rfind("_v")+2))!=std::string::npos && (*triggerResults).accept(i))
	  _trig_LowPtJet=true;
        else if (triggerNames.triggerName(i).find(highMETExpr_[0].substr(0,highMETExpr_[0].rfind("_v")+2))!=std::string::npos && (*triggerResults).accept(i))
	  _trig_HighMET=true;
	//        else if (triggerNames.triggerName(i).find(lowMETExpr_[0].substr(0,lowMETExpr_[0].rfind("_v")+2))!=std::string::npos && (*triggerResults).accept(i))
	//	  _trig_LowMET=true;
        else if (triggerNames.triggerName(i).find(muonExpr_[0].substr(0,muonExpr_[0].rfind("_v")+2))!=std::string::npos && (*triggerResults).accept(i))
	  _trig_Muon=true;
        else if (triggerNames.triggerName(i).find(elecExpr_[0].substr(0,elecExpr_[0].rfind("_v")+2))!=std::string::npos && (*triggerResults).accept(i))
	  _trig_Ele=true;
        else if (triggerNames.triggerName(i).find(minbiasExpr_[0].substr(0,minbiasExpr_[0].rfind("_v")+2))!=std::string::npos && (*triggerResults).accept(i))
	  _trig_MinBias=true;
      }

    // count number of requested Jet or MB HLT paths which have fired
    for (unsigned int i=0; i!=HLTPathsJetMBByName_.size(); i++) {
      unsigned int triggerIndex = triggerNames.triggerIndex(HLTPathsJetMBByName_[i]);
      if (triggerIndex<(*triggerResults).size()) {
	if ((*triggerResults).accept(triggerIndex)) {
	  _trig_JetMB++;
	}
      }
    }
    // for empty input vectors (n==0), take all HLT triggers!
    if (HLTPathsJetMBByName_.size()==0) _trig_JetMB=(*triggerResults).size()-1;

    /*that is what we had in TCMet
    //
    if (_verbose) std::cout << "triggerNames size" << " " << triggerNames.size() << std::endl;
    if (_verbose) std::cout << _hlt_HighPtJet << " " << triggerNames.triggerIndex(_hlt_HighPtJet) << std::endl;
    if (_verbose) std::cout << _hlt_LowPtJet  << " " << triggerNames.triggerIndex(_hlt_LowPtJet)  << std::endl;
    if (_verbose) std::cout << _hlt_HighMET   << " " << triggerNames.triggerIndex(_hlt_HighMET)   << std::endl;
    //    if (_verbose) std::cout << _hlt_LowMET    << " " << triggerNames.triggerIndex(_hlt_LowMET)    << std::endl;
    if (_verbose) std::cout << _hlt_Ele       << " " << triggerNames.triggerIndex(_hlt_Ele)       << std::endl;
    if (_verbose) std::cout << _hlt_Muon      << " " << triggerNames.triggerIndex(_hlt_Muon)      << std::endl;

    if (triggerNames.triggerIndex(_hlt_HighPtJet) != triggerNames.size() &&
	(*triggerResults).accept(triggerNames.triggerIndex(_hlt_HighPtJet))) _trig_HighPtJet=1;

    if (triggerNames.triggerIndex(_hlt_LowPtJet)  != triggerNames.size() &&
	(*triggerResults).accept(triggerNames.triggerIndex(_hlt_LowPtJet)))  _trig_LowPtJet=1;

    if (triggerNames.triggerIndex(_hlt_HighMET)   != triggerNames.size() &&
        (*triggerResults).accept(triggerNames.triggerIndex(_hlt_HighMET)))   _trig_HighMET=1;

    //    if (triggerNames.triggerIndex(_hlt_LowMET)    != triggerNames.size() &&
    //        (*triggerResults).accept(triggerNames.triggerIndex(_hlt_LowMET)))    _trig_LowMET=1;

    if (triggerNames.triggerIndex(_hlt_Ele)       != triggerNames.size() &&
        (*triggerResults).accept(triggerNames.triggerIndex(_hlt_Ele)))       _trig_Ele=1;

    if (triggerNames.triggerIndex(_hlt_Muon)      != triggerNames.size() &&
        (*triggerResults).accept(triggerNames.triggerIndex(_hlt_Muon)))      _trig_Muon=1;
    */



    /* commented out in METAnalyzer
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
	(*triggerResults).accept(triggerNames.triggerIndex(_hlt_PhysDec)))   _trig_PhysDec=1;
  } else {

    edm::LogInfo("MetAnalyzer") << "TriggerResults::HLT not found, "
      "automatsdically select events";

    // TriggerResults object not found. Look at all events.
    _trig_JetMB=1;
  }

  // ==========================================================
  // MET information

  // **** Get the MET container
  edm::Handle<reco::METCollection> tcmetcoll;
  edm::Handle<reco::CaloMETCollection> calometcoll;
  edm::Handle<reco::PFMETCollection> pfmetcoll;

  if(isTCMet){
    iEvent.getByToken(tcMetToken_, tcmetcoll);
    if(isTCMet && !tcmetcoll.isValid()) return;
  }
  if(isCaloMet){
    iEvent.getByToken(caloMetToken_, calometcoll);
    if(isCaloMet && !calometcoll.isValid()) return;
  }
  if(isPFMet){
    iEvent.getByToken(pfMetToken_, pfmetcoll);
    if(isPFMet && !pfmetcoll.isValid()) return;
  }

  const MET *met=NULL;
  const PFMET *pfmet=NULL;
  const CaloMET *calomet=NULL;
  if(isTCMet){
    met=&(tcmetcoll->front());
  }
  if(isPFMet){
    met=&(pfmetcoll->front());
    pfmet=&(pfmetcoll->front());
  }
  if(isCaloMet){
    met=&(calometcoll->front());
    calomet=&(calometcoll->front());
  }

  LogTrace(metname)<<"[METAnalyzer] Call to the MET analyzer";

  // ==========================================================
  // TCMET

  

  if (isTCMet || (isCaloMet && theMETCollectionLabel.label() == "corMetGlobalMuons")) {

    iEvent.getByToken(MuonToken_, muon_h);
    iEvent.getByToken(TrackToken_, track_h);
    iEvent.getByToken(ElectronToken_, electron_h);
    iEvent.getByToken(BeamspotToken_, beamSpot_h);
    iEvent.getByToken(tcMET_ValueMapToken_,tcMet_ValueMap_Handle);

    if(!muon_h.isValid())     edm::LogInfo("OutputInfo") << "falied to retrieve muon data require by MET Task";
    if(!track_h.isValid())    edm::LogInfo("OutputInfo") << "falied to retrieve track data require by MET Task";
    if(!electron_h.isValid()) edm::LogInfo("OutputInfo") << "falied to retrieve electron data require by MET Task";
    if(!beamSpot_h.isValid()) edm::LogInfo("OutputInfo") << "falied to retrieve beam spot data require by MET Task";

    bspot = ( beamSpot_h.isValid() ) ? beamSpot_h->position() : math::XYZPoint(0, 0, 0);

  }

  // ==========================================================
  //

  edm::Handle<HcalNoiseRBXCollection> HRBXCollection;
  iEvent.getByToken(HcalNoiseRBXToken_,HRBXCollection);
  if (!HRBXCollection.isValid()) {
    LogDebug("") << "METAnalyzer: Could not find HcalNoiseRBX Collection" << std::endl;
    if (_verbose) std::cout << "METAnalyzer: Could not find HcalNoiseRBX Collection" << std::endl;
  }


  edm::Handle<bool> HBHENoiseFilterResultHandle;
  iEvent.getByToken(HBHENoiseFilterResultToken_, HBHENoiseFilterResultHandle);
  bool HBHENoiseFilterResult = *HBHENoiseFilterResultHandle;
  if (!HBHENoiseFilterResultHandle.isValid()) {
    LogDebug("") << "METAnalyzer: Could not find HBHENoiseFilterResult" << std::endl;
    if (_verbose) std::cout << "METAnalyzer: Could not find HBHENoiseFilterResult" << std::endl;
  }

  bool bJetIDMinimal=true;
  bool bJetIDLoose=true;
  bool bJetIDTight=true;

  if(isCaloMet){
    edm::Handle<reco::CaloJetCollection> caloJets;
    iEvent.getByToken(caloJetsToken_, caloJets);
    if (!caloJets.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
      if (_verbose) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
    }
    

    // JetID
    
    if (_verbose) std::cout << "caloJet JetID starts" << std::endl;
    
    //
    // --- Minimal cuts
    //

 
    for (reco::CaloJetCollection::const_iterator cal = caloJets->begin();
	 cal!=caloJets->end(); ++cal){
      jetID->calculate(iEvent, *cal);
      if (cal->pt()>10.){
	if (fabs(cal->eta())<=2.6 &&
	    cal->emEnergyFraction()<=0.01) bJetIDMinimal=false;
      }
    }
    
    //
    // --- Loose cuts, not  specific for now!
    //
    
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
	//if (jetID->restrictedEMF()<0.01) bJetIDLoose=false;
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
    
    if (_verbose) std::cout << "caloJet JetID ends" << std::endl;
  }
  if(isTCMet){
    edm::Handle<reco::JPTJetCollection> jptJets;
    iEvent.getByToken(jptJetsToken_, jptJets);
    if (!jptJets.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find JPT jet product" << std::endl;
      if (_verbose) std::cout << "METAnalyzer: Could not find JPT jet product" << std::endl;
    }

    for (reco::JPTJetCollection::const_iterator jpt = jptJets->begin();
	 jpt!=jptJets->end(); ++jpt){
      const edm::RefToBase<reco::Jet>&  rawJet = jpt->getCaloJetRef();
      const reco::CaloJet *rawCaloJet = dynamic_cast<const reco::CaloJet*>(&*rawJet);
      
      jetID->calculate(iEvent, *rawCaloJet);
      if (jpt->pt()>10.){
	//ID of JPT jets depend 
	if (fabs(rawCaloJet->eta())<=2.6 &&
	    rawCaloJet->emEnergyFraction()<=0.01) bJetIDMinimal=false;
      }
    }
    
    //
    // --- Loose cuts, not  specific for now!
    //
    
    for (reco::JPTJetCollection::const_iterator jpt = jptJets->begin();
	 jpt!=jptJets->end(); ++jpt){
      const edm::RefToBase<reco::Jet>&  rawJet = jpt->getCaloJetRef();
      const reco::CaloJet *rawCaloJet = dynamic_cast<const reco::CaloJet*>(&*rawJet);
      jetID->calculate(iEvent,*rawCaloJet);
      if (_verbose) std::cout << jetID->n90Hits() << " "
			      << jetID->restrictedEMF() << " "
			      << jpt->pt() << std::endl;
      //calculate ID for JPT jets above 10
      if (jpt->pt()>10.){
	//for ID itself calojet is the relevant quantity
	// for all regions
	if (jetID->n90Hits()<2)  bJetIDLoose=false;
	if (jetID->fHPD()>=0.98) bJetIDLoose=false;
	//if (jetID->restrictedEMF()<0.01) bJetIDLoose=false;
	//
	// for non-forward
	if (fabs(rawCaloJet->eta())<2.55){
	  if (rawCaloJet->emEnergyFraction()<=0.01) bJetIDLoose=false;
	}
	// for forward
	else {
	  if (rawCaloJet->emEnergyFraction()<=-0.9) bJetIDLoose=false;
	  if (rawCaloJet->pt()>80.){
	    if (rawCaloJet->emEnergyFraction()>= 1.0) bJetIDLoose=false;
	  }
	} // forward vs non-forward
      }   // pt>10 GeV/c
    }     // rawCaloJetor-jets loop
    
    //
    // --- Tight cuts
    //
    bJetIDTight=bJetIDLoose;
    for (reco::JPTJetCollection::const_iterator jpt = jptJets->begin();
	 jpt!=jptJets->end(); ++jpt){
      const edm::RefToBase<reco::Jet>&  rawJet = jpt->getCaloJetRef();
      const reco::CaloJet *rawCaloJet = dynamic_cast<const reco::CaloJet*>(&*rawJet);
      jetID->calculate(iEvent, *rawCaloJet);
      if (jpt->pt()>25.){
	//
	// for all regions
	if (jetID->fHPD()>=0.95) bJetIDTight=false;
	//
	// for 1.0<|eta|<1.75
	if (fabs(rawCaloJet->eta())>=1.00 && fabs(rawCaloJet->eta())<1.75){
	  if (rawCaloJet->pt()>80. && rawCaloJet->emEnergyFraction()>=1.) bJetIDTight=false;
	}
	//
	// for 1.75<|eta|<2.55
	else if (fabs(rawCaloJet->eta())>=1.75 && fabs(rawCaloJet->eta())<2.55){
	  if (rawCaloJet->pt()>80. && rawCaloJet->emEnergyFraction()>=1.) bJetIDTight=false;
	}
	//
	// for 2.55<|eta|<3.25
	else if (fabs(rawCaloJet->eta())>=2.55 && fabs(rawCaloJet->eta())<3.25){
	  if (rawCaloJet->pt()< 50.                   && rawCaloJet->emEnergyFraction()<=-0.3) bJetIDTight=false;
	  if (rawCaloJet->pt()>=50. && rawCaloJet->pt()< 80. && rawCaloJet->emEnergyFraction()<=-0.2) bJetIDTight=false;
	  if (rawCaloJet->pt()>=80. && rawCaloJet->pt()<340. && rawCaloJet->emEnergyFraction()<=-0.1) bJetIDTight=false;
	  if (rawCaloJet->pt()>=340.                  && rawCaloJet->emEnergyFraction()<=-0.1
	      && rawCaloJet->emEnergyFraction()>=0.95) bJetIDTight=false;
	}
	//
	// for 3.25<|eta|
	else if (fabs(rawCaloJet->eta())>=3.25){
	  if (rawCaloJet->pt()< 50.                   && rawCaloJet->emEnergyFraction()<=-0.3
	      && rawCaloJet->emEnergyFraction()>=0.90) bJetIDTight=false;
	  if (rawCaloJet->pt()>=50. && rawCaloJet->pt()<130. && rawCaloJet->emEnergyFraction()<=-0.2
	      && rawCaloJet->emEnergyFraction()>=0.80) bJetIDTight=false;
	  if (rawCaloJet->pt()>=130.                  && rawCaloJet->emEnergyFraction()<=-0.1
	      && rawCaloJet->emEnergyFraction()>=0.70) bJetIDTight=false;
	}
      }   // pt>10 GeV/c
    }     // rawCaloJetor-jets loop
    if(isPFMet){
      edm::Handle<reco::PFJetCollection> pfJets;
      iEvent.getByToken(pfJetsToken_, pfJets);
      if (!pfJets.isValid()) {
	LogDebug("") << "METAnalyzer: Could not find PF jet product" << std::endl;
	if (_verbose) std::cout << "METAnalyzer: Could not find PF jet product" << std::endl;
      }
      
      for (reco::PFJetCollection::const_iterator pf = pfJets->begin();
	   pf!=pfJets->end(); ++pf){
	if (pf->pt()>10.){
	  //ID of JPT jets depend 
	  if((pf->neutralHadronEnergyFraction() + pf->HFHadronEnergyFraction())>=1)bJetIDMinimal=false;
	  if((pf->photonEnergyFraction() + pf->HFEMEnergyFraction())>=1)bJetIDMinimal=false;
	  if(pf->nConstituents()<2)bJetIDMinimal=false;
	  if (fabs(pf->eta())<=2.4){
	    if(pf->electronEnergyFraction()>=1)bJetIDMinimal=false;
	    if( pf->chargedHadronEnergyFraction ()<=0) bJetIDMinimal=false;
	  }
	}
      }
    //
    // --- Loose cuts, not  specific for now!
    //

     for (reco::PFJetCollection::const_iterator pf = pfJets->begin();
	   pf!=pfJets->end(); ++pf){
	if (pf->pt()>10.){
	  //ID of JPT jets depend 
	  if((pf->neutralHadronEnergyFraction() + pf->HFHadronEnergyFraction())>=0.99)bJetIDLoose=false;
	  if((pf->photonEnergyFraction() + pf->HFEMEnergyFraction())>=0.99)bJetIDLoose=false;
	  if(pf->nConstituents()<2)bJetIDLoose=false;
	  if (fabs(pf->eta())<=2.4){
	    if(pf->electronEnergyFraction()>=0.99)bJetIDLoose=false;
	    if( pf->chargedHadronEnergyFraction ()<=0) bJetIDLoose=false;
	  }
	}
      }
    
    //
    // --- Tight cuts
    //
    bJetIDTight=bJetIDLoose;
       for (reco::PFJetCollection::const_iterator pf = pfJets->begin();
	   pf!=pfJets->end(); ++pf){
	if (pf->pt()>10.){
	  //ID of JPT jets depend 
	  if((pf->neutralHadronEnergyFraction() + pf->HFHadronEnergyFraction())>=0.90)bJetIDTight=false;
	  if((pf->photonEnergyFraction() + pf->HFEMEnergyFraction())>=0.90)bJetIDTight=false;
	  if(pf->nConstituents()<2)bJetIDTight=false;
	  if (fabs(pf->eta())<=2.4){
	    if(pf->electronEnergyFraction()>=0.99)bJetIDTight=false;
	    if( pf->chargedHadronEnergyFraction ()<=0) bJetIDTight=false;
	  }
	}
      }
    if (_verbose) std::cout << "TCMET JetID ends" << std::endl;
    }
  }

  // ==========================================================
  // HCAL Noise filter

  bool bHcalNoiseFilter = HBHENoiseFilterResult;

  // ==========================================================
  // Get BeamHaloSummary
  edm::Handle<BeamHaloSummary> TheBeamHaloSummary ;
  iEvent.getByToken(BeamHaloSummaryToken_, TheBeamHaloSummary) ;

  if (!TheBeamHaloSummary.isValid()) {
    std::cout << "BeamHaloSummary doesn't exist" << std::endl;
  }

  bool bBeamHaloIDTightPass = true;
  bool bBeamHaloIDLoosePass = true;

  if(!TheBeamHaloSummary.isValid()) {

  const BeamHaloSummary TheSummary = (*TheBeamHaloSummary.product() );

  if( !TheSummary.EcalLooseHaloId()  && !TheSummary.HcalLooseHaloId() &&
      !TheSummary.CSCLooseHaloId()   && !TheSummary.GlobalLooseHaloId() )
    bBeamHaloIDLoosePass = false;

  if( !TheSummary.EcalTightHaloId()  && !TheSummary.HcalTightHaloId() &&
      !TheSummary.CSCTightHaloId()   && !TheSummary.GlobalTightHaloId() )
    bBeamHaloIDTightPass = false;

  }

  // ==========================================================
  //Vertex information
  _numPV = 0;
  bool bPrimaryVertex = true;
  if(_doPVCheck){
    bPrimaryVertex = false;
    Handle<VertexCollection> vertexHandle;

    iEvent.getByToken(vertexToken_, vertexHandle);

    if (!vertexHandle.isValid()) {
      LogDebug("") << "CaloMETAnalyzer: Could not find vertex collection" << std::endl;
      if (_verbose) std::cout << "CaloMETAnalyzer: Could not find vertex collection" << std::endl;
    }

    if ( vertexHandle.isValid() ){
      VertexCollection vertexCollection = *(vertexHandle.product());
      int vertex_number     = vertexCollection.size();
      VertexCollection::const_iterator v = vertexCollection.begin();
      for ( ; v != vertexCollection.end(); ++v) {
	double vertex_chi2    = v->normalizedChi2();
	double vertex_ndof    = v->ndof();
	bool   fakeVtx        = v->isFake();
	double vertex_Z       = v->z();

	if (  !fakeVtx
	      && vertex_number>=_nvtx_min
	      && vertex_ndof   >_vtxndof_min
	      && vertex_chi2   <_vtxchi2_max
	      && fabs(vertex_Z)<_vtxz_max )
	  bPrimaryVertex = true;
	++_numPV;
      }
    }
  }

  // ==========================================================

  edm::Handle< L1GlobalTriggerReadoutRecord > gtReadoutRecord;
  //iEvent.getByLabel( gtTag, gtReadoutRecord);
  iEvent.getByToken( gtToken_, gtReadoutRecord);

  if (!gtReadoutRecord.isValid()) {
    LogDebug("") << "CaloMETAnalyzer: Could not find GT readout record" << std::endl;
    if (_verbose) std::cout << "CaloMETAnalyzer: Could not find GT readout record product" << std::endl;
  }
  // ==========================================================
  // Reconstructed MET Information - fill MonitorElements

  bool bHcalNoise   = bHcalNoiseFilter;
  bool bBeamHaloID  = bBeamHaloIDLoosePass;
  bool bJetID       = true;



  if      (_tightBHFiltering)       bBeamHaloID = bBeamHaloIDTightPass;

  if      (_tightJetIDFiltering==1)  bJetID      = bJetIDMinimal;
  else if (_tightJetIDFiltering==2)  bJetID      = bJetIDLoose;
  else if (_tightJetIDFiltering==3)  bJetID      = bJetIDTight;
  else if (_tightJetIDFiltering==-1) bJetID      = true;

  bool bBasicCleanup = bPrimaryVertex;
  bool bExtraCleanup = bBasicCleanup && bHcalNoise && bJetID && bBeamHaloID;


  for (std::vector<std::string>::const_iterator ic = _FolderNames.begin();
       ic != _FolderNames.end(); ic++){
    if (*ic=="All")                                             fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet);
    if (DCSFilter->filter(iEvent, iSetup)) {
      if (_cleanupSelection){
	if (*ic=="BasicCleanup" && bBasicCleanup)                   fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet);
	if (*ic=="ExtraCleanup" && bExtraCleanup)                   fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet);
      }
      if (_allSelection) {
	if (*ic=="HcalNoiseFilter"      && bHcalNoiseFilter )       fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet);
	if (*ic=="JetIDMinimal"         && bJetIDMinimal)           fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet);
	if (*ic=="JetIDLoose"           && bJetIDLoose)             fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet);
	if (*ic=="JetIDTight"           && bJetIDTight)             fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet);
	if (*ic=="BeamHaloIDTightPass"  && bBeamHaloIDTightPass)    fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet);
	if (*ic=="BeamHaloIDLoosePass"  && bBeamHaloIDLoosePass)    fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet);
	if (*ic=="PV"                   && bPrimaryVertex)          fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet);
      }
    } // DCS
  }
}


// ***********************************************************
void METAnalyzer::fillMESet(const edm::Event& iEvent, std::string DirName,
			    const reco::MET& met, const reco::PFMET& pfmet, const reco::CaloMET& calomet)
{

  _dbe->setCurrentFolder(DirName);

  bool bLumiSecPlot=false;
  if (DirName.find("All")) bLumiSecPlot=true;

  if (_trig_JetMB)
    fillMonitorElement(iEvent,DirName,"",met,pfmet,calomet, bLumiSecPlot);
  if (_trig_HighPtJet)
    fillMonitorElement(iEvent,DirName,"HighPtJet",met,pfmet,calomet,false);
  if (_trig_LowPtJet)
    fillMonitorElement(iEvent,DirName,"LowPtJet",met,pfmet,calomet,false);
  if (_trig_MinBias)
    fillMonitorElement(iEvent,DirName,"MinBias",met,pfmet,calomet,false);
  if (_trig_HighMET)
    fillMonitorElement(iEvent,DirName,"HighMET",met,pfmet,calomet,false);
  //  if (_trig_LowMET)
  //    fillMonitorElement(iEvent,DirName,"LowMET",met,pfmet,calomet,false);
  if (_trig_Ele)
    fillMonitorElement(iEvent,DirName,"Ele",met,pfmet,calomet,false);
  if (_trig_Muon)
    fillMonitorElement(iEvent,DirName,"Muon",met,pfmet,calomet,false);
}

// ***********************************************************
void METAnalyzer::fillMonitorElement(const edm::Event& iEvent, std::string DirName,
					 std::string TriggerTypeName,
				     const reco::MET& met, const reco::PFMET & pfmet, const reco::CaloMET &calomet, bool bLumiSecPlot)
{

  if (TriggerTypeName=="HighPtJet") {
    if (!selectHighPtJetEvent(iEvent)) return;
  }
  else if (TriggerTypeName=="LowPtJet") {
    if (!selectLowPtJetEvent(iEvent)) return;
  }
  else if (TriggerTypeName=="HighMET") {
    if (met.pt()<_highMETThreshold) return;
  }
  //  else if (TriggerTypeName=="LowMET") {
  //    if (met.pt()<_lowMETThreshold) return;
  //  }
  else if (TriggerTypeName=="Ele") {
    if (!selectWElectronEvent(iEvent)) return;
  }
  else if (TriggerTypeName=="Muon") {
    if (!selectWMuonEvent(iEvent)) return;
  }

// Reconstructed MET Information
  double SumET  = met.sumEt();
  double METSig = met.mEtSig();
  //double Ez     = met.e_longitudinal();
  double MET    = met.pt();
  double MEx    = met.px();
  double MEy    = met.py();
  double METPhi = met.phi();
  //
  int myLuminosityBlock;
  //  myLuminosityBlock = (evtCounter++)/1000;
  myLuminosityBlock = iEvent.luminosityBlock();
  //

  if (TriggerTypeName!="") DirName = DirName +"/"+TriggerTypeName;

  if (_verbose) std::cout << "_etThreshold = " << _etThreshold << std::endl;

  if (SumET>_etThreshold){
    hMEx    = _dbe->get(DirName+"/"+"METTask_MEx");     if (hMEx           && hMEx->getRootObject()){     hMEx          ->Fill(MEx);}
    hMEy    = _dbe->get(DirName+"/"+"METTask_MEy");     if (hMEy           && hMEy->getRootObject())     hMEy          ->Fill(MEy);
    hMET    = _dbe->get(DirName+"/"+"METTask_MET");     if (hMET           && hMET->getRootObject())     hMET          ->Fill(MET);
    hMETPhi = _dbe->get(DirName+"/"+"METTask_METPhi");  if (hMETPhi        && hMETPhi->getRootObject())  hMETPhi       ->Fill(METPhi);
    hSumET  = _dbe->get(DirName+"/"+"METTask_SumET");   if (hSumET         && hSumET->getRootObject())   hSumET        ->Fill(SumET);
    hMETSig = _dbe->get(DirName+"/"+"METTask_METSig");  if (hMETSig        && hMETSig->getRootObject())  hMETSig       ->Fill(METSig);
    //hEz     = _dbe->get(DirName+"/"+"METTask_Ez");      if (hEz            && hEz->getRootObject())      hEz           ->Fill(Ez);

    hMET_logx   = _dbe->get(DirName+"/"+"METTask_MET_logx");    if (hMET_logx      && hMET_logx->getRootObject())    hMET_logx->Fill(log10(MET));
    hSumET_logx = _dbe->get(DirName+"/"+"METTask_SumET_logx");  if (hSumET_logx    && hSumET_logx->getRootObject())  hSumET_logx->Fill(log10(SumET));

    // Fill NPV profiles
      //--------------------------------------------------------------------------
    meMEx_profile   = _dbe->get(DirName + "/METTask_MEx_profile");
    meMEy_profile   = _dbe->get(DirName + "/METTask_MEy_profile");
    meMET_profile   = _dbe->get(DirName + "/METTask_MET_profile");
    meSumET_profile = _dbe->get(DirName + "/METTask_SumET_profile");
    
    if (meMEx_profile   && meMEx_profile  ->getRootObject()) meMEx_profile  ->Fill(_numPV, MEx);
    if (meMEy_profile   && meMEy_profile  ->getRootObject()) meMEy_profile  ->Fill(_numPV, MEy);
    if (meMET_profile   && meMET_profile  ->getRootObject()) meMET_profile  ->Fill(_numPV, MET);
    if (meSumET_profile && meSumET_profile->getRootObject()) meSumET_profile->Fill(_numPV, SumET);
 

    //hMETIonFeedbck = _dbe->get(DirName+"/"+"METTask_METIonFeedbck");  if (hMETIonFeedbck && hMETIonFeedbck->getRootObject())  hMETIonFeedbck->Fill(MET);
    //hMETHPDNoise   = _dbe->get(DirName+"/"+"METTask_METHPDNoise");    if (hMETHPDNoise   && hMETHPDNoise->getRootObject())    hMETHPDNoise->Fill(MET);
    //comment out like already done before for TcMET and PFMET
    if(isTCMet || theMETCollectionLabel.label() == "corMetGlobalMuons"){
      hMETIonFeedbck = _dbe->get(DirName+"/"+"METTask_METIonFeedbck");  if (hMETIonFeedbck && hMETIonFeedbck->getRootObject()) hMETIonFeedbck->Fill(MET);
      hMETHPDNoise   = _dbe->get(DirName+"/"+"METTask_METHPDNoise");    if (hMETHPDNoise   && hMETHPDNoise->getRootObject())   hMETHPDNoise->Fill(MET);
      hMETRBXNoise   = _dbe->get(DirName+"/"+"METTask_METRBXNoise");    if (hMETRBXNoise   && hMETRBXNoise->getRootObject())   hMETRBXNoise->Fill(MET);
    }


    if(isCaloMet){
      //const reco::CaloMETCollection *calometcol = calometcoll.product();
      //const reco::CaloMET *calomet;
      //calomet = &(calometcol->front());
      
      double caloEtFractionHadronic = calomet.etFractionHadronic();
      double caloEmEtFraction       = calomet.emEtFraction();

      double caloMaxEtInEMTowers    = calomet.maxEtInEmTowers();
      double caloMaxEtInHadTowers   = calomet.maxEtInHadTowers();
      
      double caloHadEtInHB = calomet.hadEtInHB();
      double caloHadEtInHO = calomet.hadEtInHO();
      double caloHadEtInHE = calomet.hadEtInHE();
      double caloHadEtInHF = calomet.hadEtInHF();
      double caloEmEtInEB  = calomet.emEtInEB();
      double caloEmEtInEE  = calomet.emEtInEE();
      double caloEmEtInHF  = calomet.emEtInHF();

      hCaloMaxEtInEmTowers  = _dbe->get(DirName+"/"+"METTask_CaloMaxEtInEmTowers");   if (hCaloMaxEtInEmTowers  && hCaloMaxEtInEmTowers->getRootObject())   hCaloMaxEtInEmTowers->Fill(caloMaxEtInEMTowers);
      hCaloMaxEtInHadTowers = _dbe->get(DirName+"/"+"METTask_CaloMaxEtInHadTowers");  if (hCaloMaxEtInHadTowers && hCaloMaxEtInHadTowers->getRootObject())  hCaloMaxEtInHadTowers->Fill(caloMaxEtInHadTowers);
      
      hCaloHadEtInHB = _dbe->get(DirName+"/"+"METTask_CaloHadEtInHB");  if (hCaloHadEtInHB  &&  hCaloHadEtInHB->getRootObject())  hCaloHadEtInHB->Fill(caloHadEtInHB);
      hCaloHadEtInHO = _dbe->get(DirName+"/"+"METTask_CaloHadEtInHO");  if (hCaloHadEtInHO  &&  hCaloHadEtInHO->getRootObject())  hCaloHadEtInHO->Fill(caloHadEtInHO);
      hCaloHadEtInHE = _dbe->get(DirName+"/"+"METTask_CaloHadEtInHE");  if (hCaloHadEtInHE  &&  hCaloHadEtInHE->getRootObject())  hCaloHadEtInHE->Fill(caloHadEtInHE);
      hCaloHadEtInHF = _dbe->get(DirName+"/"+"METTask_CaloHadEtInHF");  if (hCaloHadEtInHF  &&  hCaloHadEtInHF->getRootObject())  hCaloHadEtInHF->Fill(caloHadEtInHF);
      hCaloEmEtInEB  = _dbe->get(DirName+"/"+"METTask_CaloEmEtInEB");   if (hCaloEmEtInEB   &&  hCaloEmEtInEB->getRootObject())   hCaloEmEtInEB->Fill(caloEmEtInEB);
      hCaloEmEtInEE  = _dbe->get(DirName+"/"+"METTask_CaloEmEtInEE");   if (hCaloEmEtInEE   &&  hCaloEmEtInEE->getRootObject())   hCaloEmEtInEE->Fill(caloEmEtInEE);
      hCaloEmEtInHF  = _dbe->get(DirName+"/"+"METTask_CaloEmEtInHF");   if (hCaloEmEtInHF   &&  hCaloEmEtInHF->getRootObject())   hCaloEmEtInHF->Fill(caloEmEtInHF);

      hCaloMETPhi020 = _dbe->get(DirName+"/"+"METTask_CaloMETPhi020");    if (MET> 20. && hCaloMETPhi020  &&  hCaloMETPhi020->getRootObject()) { hCaloMETPhi020->Fill(METPhi);}


      hCaloEtFractionHadronic = _dbe->get(DirName+"/"+"METTask_CaloEtFractionHadronic"); if (hCaloEtFractionHadronic && hCaloEtFractionHadronic->getRootObject())  hCaloEtFractionHadronic->Fill(caloEtFractionHadronic);
      hCaloEmEtFraction       = _dbe->get(DirName+"/"+"METTask_CaloEmEtFraction");       if (hCaloEmEtFraction       && hCaloEmEtFraction->getRootObject())        hCaloEmEtFraction->Fill(caloEmEtFraction);
      hCaloEmEtFraction020 = _dbe->get(DirName+"/"+"METTask_CaloEmEtFraction020");       if (MET> 20.  &&  hCaloEmEtFraction020    && hCaloEmEtFraction020->getRootObject()) hCaloEmEtFraction020->Fill(caloEmEtFraction);
      if (theMETCollectionLabel.label() == "corMetGlobalMuons" ) {
	
	for( reco::MuonCollection::const_iterator muonit = muon_h->begin(); muonit != muon_h->end(); muonit++ ) {
	  const reco::TrackRef siTrack = muonit->innerTrack();
	  hCalomuPt    = _dbe->get(DirName+"/"+"METTask_CalomuonPt");  
	  if (hCalomuPt    && hCalomuPt->getRootObject())   hCalomuPt->Fill( muonit->p4().pt() );
	  hCalomuEta   = _dbe->get(DirName+"/"+"METTask_CalomuonEta");    if (hCalomuEta   && hCalomuEta->getRootObject())    hCalomuEta->Fill( muonit->p4().eta() );
	  hCalomuNhits = _dbe->get(DirName+"/"+"METTask_CalomuonNhits");  if (hCalomuNhits && hCalomuNhits->getRootObject())  hCalomuNhits->Fill( siTrack.isNonnull() ? siTrack->numberOfValidHits() : -999 );
	  hCalomuChi2  = _dbe->get(DirName+"/"+"METTask_CalomuonNormalizedChi2");   if (hCalomuChi2  && hCalomuChi2->getRootObject())   hCalomuChi2->Fill( siTrack.isNonnull() ? siTrack->chi2()/siTrack->ndof() : -999 );
	  double d0 = siTrack.isNonnull() ? -1 * siTrack->dxy( bspot) : -999;
	  hCalomuD0    = _dbe->get(DirName+"/"+"METTask_CalomuonD0");     if (hCalomuD0    && hCalomuD0->getRootObject())  hCalomuD0->Fill( d0 );
	}
	
	const unsigned int nMuons = muon_h->size();
	for( unsigned int mus = 0; mus < nMuons; mus++ ) {
	  reco::MuonRef muref( muon_h, mus);
	  reco::MuonMETCorrectionData muCorrData = (*tcMet_ValueMap_Handle)[muref];
	  hCaloMExCorrection      = _dbe->get(DirName+"/"+"METTask_CaloMExCorrection");       if (hCaloMExCorrection      && hCaloMExCorrection->getRootObject())       hCaloMExCorrection-> Fill(muCorrData.corrY());
	  hCaloMEyCorrection      = _dbe->get(DirName+"/"+"METTask_CaloMEyCorrection");       if (hCaloMEyCorrection      && hCaloMEyCorrection->getRootObject())       hCaloMEyCorrection-> Fill(muCorrData.corrX());
	  hCaloMuonCorrectionFlag = _dbe->get(DirName+"/"+"METTask_CaloMuonCorrectionFlag");  if (hCaloMuonCorrectionFlag && hCaloMuonCorrectionFlag->getRootObject())  hCaloMuonCorrectionFlag-> Fill(muCorrData.type());
	}
      } 
    }

    if(isPFMet){
      // **** Get the MET container
      //const PFMETCollection *pfmetcol = pfmetcoll.product();
      //const PFMET *pfmet;
      //pfmet = &(pfmetcol->front());
      
      // PFMET getters
      //----------------------------------------------------------------------------
      double pfPhotonEtFraction        = pfmet.photonEtFraction();
      double pfPhotonEt                = pfmet.photonEt();
      double pfNeutralHadronEtFraction = pfmet.neutralHadronEtFraction();
      double pfNeutralHadronEt         = pfmet.neutralHadronEt();
      double pfElectronEtFraction      = pfmet.electronEtFraction();
      double pfElectronEt              = pfmet.electronEt();
      double pfChargedHadronEtFraction = pfmet.chargedHadronEtFraction();
      double pfChargedHadronEt         = pfmet.chargedHadronEt();
      double pfMuonEtFraction          = pfmet.muonEtFraction();
      double pfMuonEt                  = pfmet.muonEt();
      double pfHFHadronEtFraction      = pfmet.HFHadronEtFraction();
      double pfHFHadronEt              = pfmet.HFHadronEt();
      double pfHFEMEtFraction          = pfmet.HFEMEtFraction();
      double pfHFEMEt                  = pfmet.HFEMEt();
      
      mePhotonEtFraction        = _dbe->get(DirName + "/METTask_PfPhotonEtFraction");
      mePhotonEt                = _dbe->get(DirName + "/METTask_PfPhotonEt");
      meNeutralHadronEtFraction = _dbe->get(DirName + "/METTask_PfNeutralHadronEtFraction");
      meNeutralHadronEt         = _dbe->get(DirName + "/METTask_PfNeutralHadronEt");
      meElectronEtFraction      = _dbe->get(DirName + "/METTask_PfElectronEtFraction");
      meElectronEt              = _dbe->get(DirName + "/METTask_PfElectronEt");
      meChargedHadronEtFraction = _dbe->get(DirName + "/METTask_PfChargedHadronEtFraction");
      meChargedHadronEt         = _dbe->get(DirName + "/METTask_PfChargedHadronEt");
      meMuonEtFraction          = _dbe->get(DirName + "/METTask_PfMuonEtFraction");
      meMuonEt                  = _dbe->get(DirName + "/METTask_PfMuonEt");
      meHFHadronEtFraction      = _dbe->get(DirName + "/METTask_PfHFHadronEtFraction");
      meHFHadronEt              = _dbe->get(DirName + "/METTask_PfHFHadronEt");
      meHFEMEtFraction          = _dbe->get(DirName + "/METTask_PfHFEMEtFraction");
      meHFEMEt                  = _dbe->get(DirName + "/METTask_PfHFEMEt");
      
      if (mePhotonEtFraction        && mePhotonEtFraction       ->getRootObject()) mePhotonEtFraction       ->Fill(pfPhotonEtFraction);
      if (mePhotonEt                && mePhotonEt               ->getRootObject()) mePhotonEt               ->Fill(pfPhotonEt);
      if (meNeutralHadronEtFraction && meNeutralHadronEtFraction->getRootObject()) meNeutralHadronEtFraction->Fill(pfNeutralHadronEtFraction);
      if (meNeutralHadronEt         && meNeutralHadronEt        ->getRootObject()) meNeutralHadronEt        ->Fill(pfNeutralHadronEt);
      if (meElectronEtFraction      && meElectronEtFraction     ->getRootObject()) meElectronEtFraction     ->Fill(pfElectronEtFraction);
      if (meElectronEt              && meElectronEt             ->getRootObject()) meElectronEt             ->Fill(pfElectronEt);
      if (meChargedHadronEtFraction && meChargedHadronEtFraction->getRootObject()) meChargedHadronEtFraction->Fill(pfChargedHadronEtFraction);
      if (meChargedHadronEt         && meChargedHadronEt        ->getRootObject()) meChargedHadronEt        ->Fill(pfChargedHadronEt);
      if (meMuonEtFraction          && meMuonEtFraction         ->getRootObject()) meMuonEtFraction         ->Fill(pfMuonEtFraction);
      if (meMuonEt                  && meMuonEt                 ->getRootObject()) meMuonEt                 ->Fill(pfMuonEt);
      if (meHFHadronEtFraction      && meHFHadronEtFraction     ->getRootObject()) meHFHadronEtFraction     ->Fill(pfHFHadronEtFraction);
      if (meHFHadronEt              && meHFHadronEt             ->getRootObject()) meHFHadronEt             ->Fill(pfHFHadronEt);
      if (meHFEMEtFraction          && meHFEMEtFraction         ->getRootObject()) meHFEMEtFraction         ->Fill(pfHFEMEtFraction);
      if (meHFEMEt                  && meHFEMEt                 ->getRootObject()) meHFEMEt                 ->Fill(pfHFEMEt);
      

      //NPV profiles     
      
      mePhotonEtFraction_profile        = _dbe->get(DirName + "/METTask_PfPhotonEtFraction_profile");
      mePhotonEt_profile                = _dbe->get(DirName + "/METTask_PfPhotonEt_profile");
      meNeutralHadronEtFraction_profile = _dbe->get(DirName + "/METTask_PfNeutralHadronEtFraction_profile");
      meNeutralHadronEt_profile         = _dbe->get(DirName + "/METTask_PfNeutralHadronEt_profile");
      meElectronEtFraction_profile      = _dbe->get(DirName + "/METTask_PfElectronEtFraction_profile");
      meElectronEt_profile              = _dbe->get(DirName + "/METTask_PfElectronEt_profile");
      meChargedHadronEtFraction_profile = _dbe->get(DirName + "/METTask_PfChargedHadronEtFraction_profile");
      meChargedHadronEt_profile         = _dbe->get(DirName + "/METTask_PfChargedHadronEt_profile");
      meMuonEtFraction_profile          = _dbe->get(DirName + "/METTask_PfMuonEtFraction_profile");
      meMuonEt_profile                  = _dbe->get(DirName + "/METTask_PfMuonEt_profile");
      meHFHadronEtFraction_profile      = _dbe->get(DirName + "/METTask_PfHFHadronEtFraction_profile");
      meHFHadronEt_profile              = _dbe->get(DirName + "/METTask_PfHFHadronEt_profile");
      meHFEMEtFraction_profile          = _dbe->get(DirName + "/METTask_PfHFEMEtFraction_profile");
      meHFEMEt_profile                  = _dbe->get(DirName + "/METTask_PfHFEMEt_profile");
      
      if (mePhotonEtFraction_profile        && mePhotonEtFraction_profile       ->getRootObject()) mePhotonEtFraction_profile       ->Fill(_numPV, pfPhotonEtFraction);
      if (mePhotonEt_profile                && mePhotonEt_profile               ->getRootObject()) mePhotonEt_profile               ->Fill(_numPV, pfPhotonEt);
      if (meNeutralHadronEtFraction_profile && meNeutralHadronEtFraction_profile->getRootObject()) meNeutralHadronEtFraction_profile->Fill(_numPV, pfNeutralHadronEtFraction);
      if (meNeutralHadronEt_profile         && meNeutralHadronEt_profile        ->getRootObject()) meNeutralHadronEt_profile        ->Fill(_numPV, pfNeutralHadronEt);
      if (meElectronEtFraction_profile      && meElectronEtFraction_profile     ->getRootObject()) meElectronEtFraction_profile     ->Fill(_numPV, pfElectronEtFraction);
      if (meElectronEt_profile              && meElectronEt_profile             ->getRootObject()) meElectronEt_profile             ->Fill(_numPV, pfElectronEt);
      if (meChargedHadronEtFraction_profile && meChargedHadronEtFraction_profile->getRootObject()) meChargedHadronEtFraction_profile->Fill(_numPV, pfChargedHadronEtFraction);
      if (meChargedHadronEt_profile         && meChargedHadronEt_profile        ->getRootObject()) meChargedHadronEt_profile        ->Fill(_numPV, pfChargedHadronEt);
      if (meMuonEtFraction_profile          && meMuonEtFraction_profile         ->getRootObject()) meMuonEtFraction_profile         ->Fill(_numPV, pfMuonEtFraction);
      if (meMuonEt_profile                  && meMuonEt_profile                 ->getRootObject()) meMuonEt_profile                 ->Fill(_numPV, pfMuonEt);
      if (meHFHadronEtFraction_profile      && meHFHadronEtFraction_profile     ->getRootObject()) meHFHadronEtFraction_profile     ->Fill(_numPV, pfHFHadronEtFraction);
      if (meHFHadronEt_profile              && meHFHadronEt_profile             ->getRootObject()) meHFHadronEt_profile             ->Fill(_numPV, pfHFHadronEt);
      if (meHFEMEtFraction_profile          && meHFEMEtFraction_profile         ->getRootObject()) meHFEMEtFraction_profile         ->Fill(_numPV, pfHFEMEtFraction);
      if (meHFEMEt_profile                  && meHFEMEt_profile                 ->getRootObject()) meHFEMEt_profile                 ->Fill(_numPV, pfHFEMEt);
    }

    if (_allhist){
      if (bLumiSecPlot){
	hMExLS = _dbe->get(DirName+"/"+"METTask_MExLS"); if (hMExLS  &&  hMExLS->getRootObject())   hMExLS->Fill(MEx,myLuminosityBlock);
	hMEyLS = _dbe->get(DirName+"/"+"METTask_MEyLS"); if (hMEyLS  &&  hMEyLS->getRootObject())   hMEyLS->Fill(MEy,myLuminosityBlock);
      }
    } // _allhist

    ////////////////////////////////////
    if (isTCMet) {

      if(track_h.isValid()) {
	for( edm::View<reco::Track>::const_iterator trkit = track_h->begin(); trkit != track_h->end(); trkit++ ) {
	  htrkPt    = _dbe->get(DirName+"/"+"METTask_trackPt");     if (htrkPt    && htrkPt->getRootObject())     htrkPt->Fill( trkit->pt() );
	  htrkEta   = _dbe->get(DirName+"/"+"METTask_trackEta");    if (htrkEta   && htrkEta->getRootObject())    htrkEta->Fill( trkit->eta() );
	  htrkNhits = _dbe->get(DirName+"/"+"METTask_trackNhits");  if (htrkNhits && htrkNhits->getRootObject())  htrkNhits->Fill( trkit->numberOfValidHits() );
	  htrkChi2  = _dbe->get(DirName+"/"+"METTask_trackNormalizedChi2");  
	  if (htrkChi2  && htrkChi2->getRootObject())   htrkChi2->Fill( trkit->chi2() / trkit->ndof() );
	  double d0 = -1 * trkit->dxy( bspot );
	  htrkD0    = _dbe->get(DirName+"/"+"METTask_trackD0");     if (htrkD0 && htrkD0->getRootObject())        htrkD0->Fill( d0 );
	}
      }else{std::cout<<"tracks not valid"<<std::endl;}

      if(electron_h.isValid()) {
	for( edm::View<reco::GsfElectron>::const_iterator eleit = electron_h->begin(); eleit != electron_h->end(); eleit++ ) {
	  helePt  = _dbe->get(DirName+"/"+"METTask_electronPt");   if (helePt  && helePt->getRootObject())   helePt->Fill( eleit->p4().pt() );
	  heleEta = _dbe->get(DirName+"/"+"METTask_electronEta");  if (heleEta && heleEta->getRootObject())  heleEta->Fill( eleit->p4().eta() );
	  heleHoE = _dbe->get(DirName+"/"+"METTask_electronHoverE");  if (heleHoE && heleHoE->getRootObject())  heleHoE->Fill( eleit->hadronicOverEm() );
	}
      }else{
	std::cout<<"electrons not valid"<<std::endl;
      }

      if(muon_h.isValid()) {
	for( reco::MuonCollection::const_iterator muonit = muon_h->begin(); muonit != muon_h->end(); muonit++ ) {
	  const reco::TrackRef siTrack = muonit->innerTrack();
	  hmuPt    = _dbe->get(DirName+"/"+"METTask_muonPt");     if (hmuPt    && hmuPt->getRootObject())  hmuPt   ->Fill( muonit->p4().pt() );
	  hmuEta   = _dbe->get(DirName+"/"+"METTask_muonEta");    if (hmuEta   && hmuEta->getRootObject())  hmuEta  ->Fill( muonit->p4().eta() );
	  hmuNhits = _dbe->get(DirName+"/"+"METTask_muonNhits");  if (hmuNhits && hmuNhits->getRootObject())  hmuNhits->Fill( siTrack.isNonnull() ? siTrack->numberOfValidHits() : -999 );
	  hmuChi2  = _dbe->get(DirName+"/"+"METTask_muonNormalizedChi2");   if (hmuChi2  && hmuChi2->getRootObject())  hmuChi2 ->Fill( siTrack.isNonnull() ? siTrack->chi2()/siTrack->ndof() : -999 );
	  double d0 = siTrack.isNonnull() ? -1 * siTrack->dxy( bspot) : -999;
	  hmuD0    = _dbe->get(DirName+"/"+"METTask_muonD0");     if (hmuD0    && hmuD0->getRootObject())  hmuD0->Fill( d0 );
	}
	const unsigned int nMuons = muon_h->size();
	for( unsigned int mus = 0; mus < nMuons; mus++ ) {
	  reco::MuonRef muref( muon_h, mus);
	  reco::MuonMETCorrectionData muCorrData = (*tcMet_ValueMap_Handle)[muref];
	  hMExCorrection      = _dbe->get(DirName+"/"+"METTask_MExCorrection");       if (hMExCorrection      && hMExCorrection->getRootObject())       hMExCorrection-> Fill(muCorrData.corrY());
	  hMEyCorrection      = _dbe->get(DirName+"/"+"METTask_MEyCorrection");       if (hMEyCorrection      && hMEyCorrection->getRootObject())       hMEyCorrection-> Fill(muCorrData.corrX());
	  hMuonCorrectionFlag = _dbe->get(DirName+"/"+"METTask_CorrectionFlag");  if (hMuonCorrectionFlag && hMuonCorrectionFlag->getRootObject())  hMuonCorrectionFlag-> Fill(muCorrData.type());
	}
      }else{
	std::cout<<"muons not valid"<<std::endl;
      }
    }
  } // et threshold cut

}

// ***********************************************************
bool METAnalyzer::selectHighPtJetEvent(const edm::Event& iEvent){

  bool return_value=false;

  if(isCaloMet){
    edm::Handle<reco::CaloJetCollection> caloJets;
    iEvent.getByToken(caloJetsToken_, caloJets);
    if (!caloJets.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
      if (_verbose) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
    }
    
    for (reco::CaloJetCollection::const_iterator cal = caloJets->begin();
	 cal!=caloJets->end(); ++cal){
      if (cal->pt()>_highPtJetThreshold){
	return_value=true;
      }
    }
  }
  if(isTCMet){
    edm::Handle<reco::JPTJetCollection> jptJets;
    iEvent.getByToken(jptJetsToken_, jptJets);
    if (!jptJets.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
      if (_verbose) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
    }
    
    for (reco::JPTJetCollection::const_iterator cal = jptJets->begin();
	 cal!=jptJets->end(); ++cal){
      if (cal->pt()>_highPtJetThreshold){
	return_value=true;
      }
    }
  }
  if(isPFMet){
    edm::Handle<reco::PFJetCollection> PFJets;
    iEvent.getByToken(pfJetsToken_, PFJets);
    if (!PFJets.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
      if (_verbose) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
    }
    for (reco::PFJetCollection::const_iterator cal = PFJets->begin();
	 cal!=PFJets->end(); ++cal){
      if (cal->pt()>_highPtJetThreshold){
	return_value=true;
      }
    }
  }


  return return_value;
}

// // ***********************************************************
bool METAnalyzer::selectLowPtJetEvent(const edm::Event& iEvent){

  bool return_value=false;
  if(isCaloMet){
    edm::Handle<reco::CaloJetCollection> caloJets;
    iEvent.getByToken(caloJetsToken_, caloJets);
    if (!caloJets.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
      if (_verbose) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
    }
    
    for (reco::CaloJetCollection::const_iterator cal = caloJets->begin();
	 cal!=caloJets->end(); ++cal){
      if (cal->pt()>_lowPtJetThreshold){
	return_value=true;
      }
    }
  }
  if(isTCMet){
    edm::Handle<reco::JPTJetCollection> jptJets;
    iEvent.getByToken(jptJetsToken_, jptJets);
    if (!jptJets.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
      if (_verbose) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
    }
    
    for (reco::JPTJetCollection::const_iterator cal = jptJets->begin();
	 cal!=jptJets->end(); ++cal){
      if (cal->pt()>_lowPtJetThreshold){
	return_value=true;
      }
    }
  }
  if(isPFMet){
    edm::Handle<reco::PFJetCollection> PFJets;
    iEvent.getByToken(pfJetsToken_, PFJets);
    if (!PFJets.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
      if (_verbose) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
    }
    for (reco::PFJetCollection::const_iterator cal = PFJets->begin();
	 cal!=PFJets->end(); ++cal){
      if (cal->pt()>_lowPtJetThreshold){
	return_value=true;
      }
    }
  }
  return return_value;

}


// ***********************************************************
bool METAnalyzer::selectWElectronEvent(const edm::Event& iEvent){

  bool return_value=true;

  /*
    W-electron event selection comes here
   */

  return return_value;

}

// ***********************************************************
bool METAnalyzer::selectWMuonEvent(const edm::Event& iEvent){

  bool return_value=true;

  /*
    W-muon event selection comes here
   */

  return return_value;

}


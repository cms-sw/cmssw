/** \class JetMETAnalyzer
 *
 *  DQM jetMET analysis monitoring
 *
 *  \author F. Chlebana - Fermilab
 *          K. Hatakeyama - Rockefeller University
 *
 *          Jan. '14: modified by
 *
 *          M. Artur Weber
 *          R. Schoefbeck
 *          V. Sordini
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

  mOutputFile_   = parameters.getParameter<std::string>("OutputFile");
  MetType_ = parameters.getUntrackedParameter<std::string>("METType");

  triggerResultsLabel_        = parameters.getParameter<edm::InputTag>("TriggerResultsLabel");
  triggerResultsToken_= consumes<edm::TriggerResults>(edm::InputTag(triggerResultsLabel_));

  isCaloMet_ = (std::string("calo")==MetType_);
  isTCMet_ = (std::string("tc") ==MetType_);
  isPFMet_ = (std::string("pf") ==MetType_);

  // MET information
  metCollectionLabel_       = parameters.getParameter<edm::InputTag>("METCollectionLabel");

  if(isPFMet_){
    pfMetToken_= consumes<reco::PFMETCollection>(edm::InputTag(metCollectionLabel_));
  }
 if(isCaloMet_){
    caloMetToken_= consumes<reco::CaloMETCollection>(edm::InputTag(metCollectionLabel_));
  }
 if(isTCMet_){
    tcMetToken_= consumes<reco::METCollection>(edm::InputTag(metCollectionLabel_));
  }

  //jet cleanup parameters
  cleaningParameters_ = pSet.getParameter<ParameterSet>("CleaningParameters");

  //Vertex requirements
  bypassAllPVChecks_    = cleaningParameters_.getParameter<bool>("bypassAllPVChecks");
  vertexTag_    = cleaningParameters_.getParameter<edm::InputTag>("vertexCollection");
  vertexToken_  = consumes<std::vector<reco::Vertex> >(edm::InputTag(vertexTag_));

  //Trigger parameters
  gtTag_          = cleaningParameters_.getParameter<edm::InputTag>("gtLabel");
  gtToken_= consumes<L1GlobalTriggerReadoutRecord>(edm::InputTag(gtTag_));

  inputTrackLabel_         = parameters.getParameter<edm::InputTag>("InputTrackLabel");
  inputMuonLabel_          = parameters.getParameter<edm::InputTag>("InputMuonLabel");
  inputElectronLabel_      = parameters.getParameter<edm::InputTag>("InputElectronLabel");
  inputBeamSpotLabel_      = parameters.getParameter<edm::InputTag>("InputBeamSpotLabel");
  inputTCMETValueMap_      = parameters.getParameter<edm::InputTag>("InputTCMETValueMap");
  TrackToken_= consumes<edm::View <reco::Track> >(inputTrackLabel_);
  MuonToken_= consumes<reco::MuonCollection>(inputMuonLabel_);
  ElectronToken_= consumes<edm::View<reco::GsfElectron> >(inputElectronLabel_);
  BeamspotToken_= consumes<reco::BeamSpot>(inputBeamSpotLabel_);
  tcMETValueMapToken_= consumes< edm::ValueMap<reco::MuonMETCorrectionData> >(inputTCMETValueMap_);

  // Other data collections
  jetCollectionLabel_       = parameters.getParameter<edm::InputTag>("JetCollectionLabel");
  if (isCaloMet_) caloJetsToken_ = consumes<reco::CaloJetCollection>(jetCollectionLabel_);
  if (isTCMet_) jptJetsToken_ = consumes<reco::JPTJetCollection>(jetCollectionLabel_);
  if (isPFMet_) pfJetsToken_ = consumes<reco::PFJetCollection>(jetCollectionLabel_);

  hcalNoiseRBXCollectionTag_   = parameters.getParameter<edm::InputTag>("HcalNoiseRBXCollection");
  HcalNoiseRBXToken_ = consumes<reco::HcalNoiseRBXCollection>(hcalNoiseRBXCollectionTag_);

  beamHaloSummaryTag_          = parameters.getParameter<edm::InputTag>("BeamHaloSummaryLabel");
  beamHaloSummaryToken_       = consumes<BeamHaloSummary>(beamHaloSummaryTag_); 
  hbheNoiseFilterResultTag_    = parameters.getParameter<edm::InputTag>("HBHENoiseFilterResultLabel");
  hbheNoiseFilterResultToken_=consumes<bool>(hbheNoiseFilterResultTag_);

  // 
  nbinsPV_ = parameters.getParameter<int>("pVBin");
  nPVMin_   = parameters.getParameter<double>("pVMin");
  nPVMax_  = parameters.getParameter<double>("pVMax");

  //genericTriggerEventFlag_( new GenericTriggerEventFlag( conf_, consumesCollector() ) );
  highPtJetEventFlag_ = new GenericTriggerEventFlag( highptjetparms, consumesCollector() );
  lowPtJetEventFlag_  = new GenericTriggerEventFlag( lowptjetparms, consumesCollector() );
  minBiasEventFlag_   = new GenericTriggerEventFlag( minbiasparms , consumesCollector() );
  highMETEventFlag_   = new GenericTriggerEventFlag( highmetparms , consumesCollector() );
  //  _LowMETEventFlag    = new GenericTriggerEventFlag( lowmetparms  , consumesCollector() );
  eleEventFlag_       = new GenericTriggerEventFlag( eleparms     , consumesCollector() );
  muonEventFlag_      = new GenericTriggerEventFlag( muonparms    , consumesCollector() );

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

  delete highPtJetEventFlag_;
  delete lowPtJetEventFlag_;
  delete minBiasEventFlag_;
  delete highMETEventFlag_;
  //  delete _LowMETEventFlag;
  delete eleEventFlag_;
  delete muonEventFlag_;

}

void METAnalyzer::beginJob(){

  // trigger information
  HLTPathsJetMBByName_ = parameters.getParameter<std::vector<std::string > >("HLTPathsJetMB");

  cleaningParameters_ = parameters.getParameter<ParameterSet>("CleaningParameters"),

  //_theGTLabel         = cleaningParameters_.getParameter<edm::InputTag>("gtLabel");
  //gtToken_= consumes<L1GlobalTriggerReadoutRecord>(edm::InputTag(_theGTLabel));

//  doHkLTPhysicsOn_ = cleaningParameters_.getParameter<bool>("doHLTPhysicsOn");

  tightBHFiltering_     = cleaningParameters_.getParameter<bool>("tightBHFiltering");
  tightJetIDFiltering_  = cleaningParameters_.getParameter<int>("tightJetIDFiltering");

  // ==========================================================
  //DCS information
  // ==========================================================
  DCSFilter_ = new JetMETDQMDCSFilter(parameters.getParameter<ParameterSet>("DCSFilter"));

  // misc
  verbose_      = parameters.getParameter<int>("verbose");
  etThreshold_  = parameters.getParameter<double>("etThreshold"); // MET threshold
//  allSelection_ = parameters.getParameter<bool>("allSelection");  // Plot with all sets of event selection
  cleanupSelection_ = parameters.getParameter<bool>("cleanupSelection");  // Plot with all sets of event selection

  FolderName_              = parameters.getUntrackedParameter<std::string>("FolderName");

  highPtJetThreshold_ = parameters.getParameter<double>("HighPtJetThreshold"); // High Pt Jet threshold
  lowPtJetThreshold_  = parameters.getParameter<double>("LowPtJetThreshold");   // Low Pt Jet threshold
  highMETThreshold_   = parameters.getParameter<double>("HighMETThreshold");     // High MET threshold
  //  _lowMETThreshold    = parameters.getParameter<double>("LowMETThreshold");       // Low MET threshold

  //
  if(isCaloMet_ || isTCMet_){
    jetID_ = new reco::helper::JetIDHelper(parameters.getParameter<ParameterSet>("JetIDParams"));
  }

  // DQStore stuff
  dbe_ = edm::Service<DQMStore>().operator->();
  LogTrace(metname)<<"[METAnalyzer] Parameters initialization";
  std::string DirName = "JetMET/MET/"+metCollectionLabel_.label();
  dbe_->setCurrentFolder(DirName);

  hmetME = dbe_->book1D("metReco", "metReco", 4, 1, 5);
  if(isTCMet_){
    hmetME->setBinLabel(2,"tcMet",1);
  }
 if(isPFMet_){
    hmetME->setBinLabel(3,"pfMet",1);
  }
 if(isCaloMet_){
    hmetME->setBinLabel(1,"tcMet",1);
  }

  folderNames_.push_back("All");
  folderNames_.push_back("BasicCleanup");
  folderNames_.push_back("ExtraCleanup");
  folderNames_.push_back("HcalNoiseFilter");
  folderNames_.push_back("JetIDMinimal");
  folderNames_.push_back("JetIDLoose");
  folderNames_.push_back("JetIDTight");
  folderNames_.push_back("BeamHaloIDTightPass");
  folderNames_.push_back("BeamHaloIDLoosePass");
  folderNames_.push_back("Triggers");
  folderNames_.push_back("PV");

  for (std::vector<std::string>::const_iterator ic = folderNames_.begin();
       ic != folderNames_.end(); ic++){
    if (*ic=="All")                  bookMESet(DirName+"/"+*ic);
    if (cleanupSelection_){
    if (*ic=="BasicCleanup")         bookMESet(DirName+"/"+*ic);
    if (*ic=="ExtraCleanup")         bookMESet(DirName+"/"+*ic);
    }
//    if (allSelection_){
//      if (*ic=="HcalNoiseFilter")      bookMESet(DirName+"/"+*ic);
//      if (*ic=="JetIDMinimal")         bookMESet(DirName+"/"+*ic);
//      if (*ic=="JetIDLoose")           bookMESet(DirName+"/"+*ic);
//      if (*ic=="JetIDTight")           bookMESet(DirName+"/"+*ic);
//      if (*ic=="BeamHaloIDTightPass")  bookMESet(DirName+"/"+*ic);
//      if (*ic=="BeamHaloIDLoosePass")  bookMESet(DirName+"/"+*ic);
//      if (*ic=="Triggers")             bookMESet(DirName+"/"+*ic);
//      if (*ic=="PV")                   bookMESet(DirName+"/"+*ic);
//    }
  }

  // MET information



}

// ***********************************************************
void METAnalyzer::endJob() {
  if(isTCMet_ || isCaloMet_){
    delete jetID_;
  }
  delete DCSFilter_;

 if(!mOutputFile_.empty() && &*edm::Service<DQMStore>()){
      //dbe->save(mOutputFile_);
    edm::Service<DQMStore>()->save(mOutputFile_);
  }

}

// ***********************************************************
void METAnalyzer::bookMESet(std::string DirName)
{

  bool bLumiSecPlot=false;
  if (DirName.find("All")!=std::string::npos) bLumiSecPlot=true;

  bookMonitorElement(DirName,bLumiSecPlot);

  if ( highPtJetEventFlag_->on() ) {
    bookMonitorElement(DirName+"/"+"HighPtJet",false);
    hTriggerName_HighPtJet = dbe_->bookString("triggerName_HighPtJet", highPtJetExpr_[0]);
  }

  if ( lowPtJetEventFlag_->on() ) {
    bookMonitorElement(DirName+"/"+"LowPtJet",false);
    hTriggerName_LowPtJet = dbe_->bookString("triggerName_LowPtJet", lowPtJetExpr_[0]);
  }

  if ( minBiasEventFlag_->on() ) {
    bookMonitorElement(DirName+"/"+"MinBias",false);
    hTriggerName_MinBias = dbe_->bookString("triggerName_MinBias", minbiasExpr_[0]);
    if (verbose_) std::cout << "minBiasEventFlag_ is on, folder created\n";
  }

  if ( highMETEventFlag_->on() ) {
    bookMonitorElement(DirName+"/"+"HighMET",false);
    hTriggerName_HighMET = dbe_->bookString("triggerName_HighMET", highMETExpr_[0]);
  }

  //  if ( _LowMETEventFlag->on() ) {
  //    bookMonitorElement(DirName+"/"+"LowMET",false);
  //    hTriggerName_LowMET = dbe_->bookString("triggerName_LowMET", lowMETExpr_[0]);
  //  }

  if ( eleEventFlag_->on() ) {
    bookMonitorElement(DirName+"/"+"Ele",false);
    hTriggerName_Ele = dbe_->bookString("triggerName_Ele", elecExpr_[0]);
    if (verbose_) std::cout << "eleEventFlag is on, folder created\n";
  }

  if ( muonEventFlag_->on() ) {
    bookMonitorElement(DirName+"/"+"Muon",false);
    hTriggerName_Muon = dbe_->bookString("triggerName_Muon", muonExpr_[0]);
    if (verbose_) std::cout << "muonEventFlag is on, folder created\n";
  }
}

// ***********************************************************
void METAnalyzer::bookMonitorElement(std::string DirName, bool bLumiSecPlot=false)
{
  if (verbose_) std::cout << "bookMonitorElement " << DirName << std::endl;

  dbe_->setCurrentFolder(DirName);

  hMEx        = dbe_->book1D("METTask_MEx",        "METTask_MEx",        200, -500,  500);
  hMEy        = dbe_->book1D("METTask_MEy",        "METTask_MEy",        200, -500,  500);
  hMET        = dbe_->book1D("METTask_MET",        "METTask_MET",        200,    0, 1000);
  hSumET      = dbe_->book1D("METTask_SumET",      "METTask_SumET",      400,    0, 4000);
  hMETSig     = dbe_->book1D("METTask_METSig",     "METTask_METSig",      51,    0,   51);
  hMETPhi     = dbe_->book1D("METTask_METPhi",     "METTask_METPhi",      60, -3.2,  3.2);
  hMET_logx   = dbe_->book1D("METTask_MET_logx",   "METTask_MET_logx",    40,   -1,    7);
  hSumET_logx = dbe_->book1D("METTask_SumET_logx", "METTask_SumET_logx",  40,   -1,    7);

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
  meMEx_profile   = dbe_->bookProfile("METTask_MEx_profile",   "met.px()",    nbinsPV_, nPVMin_, nPVMax_, 200, -500,  500);
  meMEy_profile   = dbe_->bookProfile("METTask_MEy_profile",   "met.py()",    nbinsPV_, nPVMin_, nPVMax_, 200, -500,  500);
  meMET_profile   = dbe_->bookProfile("METTask_MET_profile",   "met.pt()",    nbinsPV_, nPVMin_, nPVMax_, 200,    0, 1000);
  meSumET_profile = dbe_->bookProfile("METTask_SumET_profile", "met.sumEt()", nbinsPV_, nPVMin_, nPVMax_, 400,    0, 4000);
  // Set NPV profiles x-axis title
  //----------------------------------------------------------------------------
  meMEx_profile  ->setAxisTitle("nvtx", 1);
  meMEy_profile  ->setAxisTitle("nvtx", 1);
  meMET_profile  ->setAxisTitle("nvtx", 1);
  meSumET_profile->setAxisTitle("nvtx", 1);
    
  if(isCaloMet_){
    hCaloMaxEtInEmTowers    = dbe_->book1D("METTask_CaloMaxEtInEmTowers",   "METTask_CaloMaxEtInEmTowers"   ,100,0,2000);
    hCaloMaxEtInEmTowers->setAxisTitle("Et(Max) in EM Tower [GeV]",1);
    hCaloMaxEtInHadTowers   = dbe_->book1D("METTask_CaloMaxEtInHadTowers",  "METTask_CaloMaxEtInHadTowers"  ,100,0,2000);
    hCaloMaxEtInHadTowers->setAxisTitle("Et(Max) in Had Tower [GeV]",1);

    hCaloHadEtInHB          = dbe_->book1D("METTask_CaloHadEtInHB","METTask_CaloHadEtInHB",100,0,2000);
    hCaloHadEtInHB->setAxisTitle("Had Et [GeV]",1);
    hCaloHadEtInHO          = dbe_->book1D("METTask_CaloHadEtInHO","METTask_CaloHadEtInHO",25,0,500);
    hCaloHadEtInHO->setAxisTitle("Had Et [GeV]",1);
    hCaloHadEtInHE          = dbe_->book1D("METTask_CaloHadEtInHE","METTask_CaloHadEtInHE",100,0,2000);
    hCaloHadEtInHE->setAxisTitle("Had Et [GeV]",1);
    hCaloHadEtInHF          = dbe_->book1D("METTask_CaloHadEtInHF","METTask_CaloHadEtInHF",50,0,1000);
    hCaloHadEtInHF->setAxisTitle("Had Et [GeV]",1);
    hCaloEmEtInHF           = dbe_->book1D("METTask_CaloEmEtInHF" ,"METTask_CaloEmEtInHF" ,25,0,500);
    hCaloEmEtInHF->setAxisTitle("EM Et [GeV]",1);
    hCaloEmEtInEE           = dbe_->book1D("METTask_CaloEmEtInEE" ,"METTask_CaloEmEtInEE" ,50,0,1000);
    hCaloEmEtInEE->setAxisTitle("EM Et [GeV]",1);
    hCaloEmEtInEB           = dbe_->book1D("METTask_CaloEmEtInEB" ,"METTask_CaloEmEtInEB" ,100,0,2000);
    hCaloEmEtInEB->setAxisTitle("EM Et [GeV]",1);

    hCaloMETPhi020  = dbe_->book1D("METTask_CaloMETPhi020",  "METTask_CaloMETPhi020",   60, -3.2,  3.2);
    hCaloMETPhi020 ->setAxisTitle("METPhi [rad] (MET>20 GeV)", 1);

    //hCaloMaxEtInEmTowers    = dbe_->book1D("METTask_CaloMaxEtInEmTowers",   "METTask_CaloMaxEtInEmTowers"   ,100,0,2000);
    //hCaloMaxEtInEmTowers->setAxisTitle("Et(Max) in EM Tower [GeV]",1);
    //hCaloMaxEtInHadTowers   = dbe_->book1D("METTask_CaloMaxEtInHadTowers",  "METTask_CaloMaxEtInHadTowers"  ,100,0,2000);
    //hCaloMaxEtInHadTowers->setAxisTitle("Et(Max) in Had Tower [GeV]",1);
    hCaloEtFractionHadronic = dbe_->book1D("METTask_CaloEtFractionHadronic","METTask_CaloEtFractionHadronic",100,0,1);
    hCaloEtFractionHadronic->setAxisTitle("Hadronic Et Fraction",1);
    hCaloEmEtFraction       = dbe_->book1D("METTask_CaloEmEtFraction",      "METTask_CaloEmEtFraction"      ,100,0,1);
    hCaloEmEtFraction->setAxisTitle("EM Et Fraction",1);
    
    //hCaloEmEtFraction002    = dbe_->book1D("METTask_CaloEmEtFraction002",   "METTask_CaloEmEtFraction002"      ,100,0,1);
    //hCaloEmEtFraction002->setAxisTitle("EM Et Fraction (MET>2 GeV)",1);
    //hCaloEmEtFraction010    = dbe_->book1D("METTask_CaloEmEtFraction010",   "METTask_CaloEmEtFraction010"      ,100,0,1);
    //hCaloEmEtFraction010->setAxisTitle("EM Et Fraction (MET>10 GeV)",1);
    hCaloEmEtFraction020    = dbe_->book1D("METTask_CaloEmEtFraction020",   "METTask_CaloEmEtFraction020"      ,100,0,1);
    hCaloEmEtFraction020->setAxisTitle("EM Et Fraction (MET>20 GeV)",1);

    if (metCollectionLabel_.label() == "corMetGlobalMuons" ) {
      hCalomuPt    = dbe_->book1D("METTask_CalomuonPt", "METTask_CalomuonPt", 50, 0, 500);
      hCalomuEta   = dbe_->book1D("METTask_CalomuonEta", "METTask_CalomuonEta", 60, -3.0, 3.0);
      hCalomuNhits = dbe_->book1D("METTask_CalomuonNhits", "METTask_CalomuonNhits", 50, 0, 50);
      hCalomuChi2  = dbe_->book1D("METTask_CalomuonNormalizedChi2", "METTask_CalomuonNormalizedChi2", 20, 0, 20);
      hCalomuD0    = dbe_->book1D("METTask_CalomuonD0", "METTask_CalomuonD0", 50, -1, 1);
      hCaloMExCorrection       = dbe_->book1D("METTask_CaloMExCorrection", "METTask_CaloMExCorrection", 100, -500.0,500.0);
      hCaloMEyCorrection       = dbe_->book1D("METTask_CaloMEyCorrection", "METTask_CaloMEyCorrection", 100, -500.0,500.0);
      hCaloMuonCorrectionFlag  = dbe_->book1D("METTask_CaloCorrectionFlag","METTask_CaloCorrectionFlag", 5, -0.5, 4.5);
    }

  }

  if(isPFMet_){
    mePhotonEtFraction        = dbe_->book1D("METTask_PfPhotonEtFraction",        "pfmet.photonEtFraction()",         50, 0,    1);
    mePhotonEt                = dbe_->book1D("METTask_PfPhotonEt",                "pfmet.photonEt()",                100, 0, 1000);
    meNeutralHadronEtFraction = dbe_->book1D("METTask_PfNeutralHadronEtFraction", "pfmet.neutralHadronEtFraction()",  50, 0,    1);
    meNeutralHadronEt         = dbe_->book1D("METTask_PfNeutralHadronEt",         "pfmet.neutralHadronEt()",         100, 0, 1000);
    meElectronEtFraction      = dbe_->book1D("METTask_PfElectronEtFraction",      "pfmet.electronEtFraction()",       50, 0,    1);
    meElectronEt              = dbe_->book1D("METTask_PfElectronEt",              "pfmet.electronEt()",              100, 0, 1000);
    meChargedHadronEtFraction = dbe_->book1D("METTask_PfChargedHadronEtFraction", "pfmet.chargedHadronEtFraction()",  50, 0,    1);
    meChargedHadronEt         = dbe_->book1D("METTask_PfChargedHadronEt",         "pfmet.chargedHadronEt()",         100, 0, 1000);
    meMuonEtFraction          = dbe_->book1D("METTask_PfMuonEtFraction",          "pfmet.muonEtFraction()",           50, 0,    1);
    meMuonEt                  = dbe_->book1D("METTask_PfMuonEt",                  "pfmet.muonEt()",                  100, 0, 1000);
    meHFHadronEtFraction      = dbe_->book1D("METTask_PfHFHadronEtFraction",      "pfmet.HFHadronEtFraction()",       50, 0,    1);
    meHFHadronEt              = dbe_->book1D("METTask_PfHFHadronEt",              "pfmet.HFHadronEt()",              100, 0, 1000);
    meHFEMEtFraction          = dbe_->book1D("METTask_PfHFEMEtFraction",          "pfmet.HFEMEtFraction()",           50, 0,    1);
    meHFEMEt                  = dbe_->book1D("METTask_PfHFEMEt",                  "pfmet.HFEMEt()",                  100, 0, 1000);
    
    mePhotonEtFraction_profile        = dbe_->bookProfile("METTask_PfPhotonEtFraction_profile",        "pfmet.photonEtFraction()",        nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    mePhotonEt_profile                = dbe_->bookProfile("METTask_PfPhotonEt_profile",                "pfmet.photonEt()",                nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    meNeutralHadronEtFraction_profile = dbe_->bookProfile("METTask_PfNeutralHadronEtFraction_profile", "pfmet.neutralHadronEtFraction()", nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meNeutralHadronEt_profile         = dbe_->bookProfile("METTask_PfNeutralHadronEt_profile",         "pfmet.neutralHadronEt()",         nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    meElectronEtFraction_profile      = dbe_->bookProfile("METTask_PfElectronEtFraction_profile",      "pfmet.electronEtFraction()",      nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meElectronEt_profile              = dbe_->bookProfile("METTask_PfElectronEt_profile",              "pfmet.electronEt()",              nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    meChargedHadronEtFraction_profile = dbe_->bookProfile("METTask_PfChargedHadronEtFraction_profile", "pfmet.chargedHadronEtFraction()", nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meChargedHadronEt_profile         = dbe_->bookProfile("METTask_PfChargedHadronEt_profile",         "pfmet.chargedHadronEt()",         nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    meMuonEtFraction_profile          = dbe_->bookProfile("METTask_PfMuonEtFraction_profile",          "pfmet.muonEtFraction()",          nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meMuonEt_profile                  = dbe_->bookProfile("METTask_PfMuonEt_profile",                  "pfmet.muonEt()",                  nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    meHFHadronEtFraction_profile      = dbe_->bookProfile("METTask_PfHFHadronEtFraction_profile",      "pfmet.HFHadronEtFraction()",      nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meHFHadronEt_profile              = dbe_->bookProfile("METTask_PfHFHadronEt_profile",              "pfmet.HFHadronEt()",              nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    meHFEMEtFraction_profile          = dbe_->bookProfile("METTask_PfHFEMEtFraction_profile",          "pfmet.HFEMEtFraction()",          nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meHFEMEt_profile                  = dbe_->bookProfile("METTask_PfHFEMEt_profile",                  "pfmet.HFEMEt()",                  nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    
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



  if (isCaloMet_){
    if (bLumiSecPlot){
      hMExLS = dbe_->book2D("METTask_MEx_LS","METTask_MEx_LS",200,-200,200,50,0.,500.);
      hMExLS->setAxisTitle("MEx [GeV]",1);
      hMExLS->setAxisTitle("Lumi Section",2);
      hMEyLS = dbe_->book2D("METTask_MEy_LS","METTask_MEy_LS",200,-200,200,50,0.,500.);
      hMEyLS->setAxisTitle("MEy [GeV]",1);
      hMEyLS->setAxisTitle("Lumi Section",2);
    }
  }

  if (isTCMet_) {
    htrkPt    = dbe_->book1D("METTask_trackPt", "METTask_trackPt", 50, 0, 500);
    htrkEta   = dbe_->book1D("METTask_trackEta", "METTask_trackEta", 60, -3.0, 3.0);
    htrkNhits = dbe_->book1D("METTask_trackNhits", "METTask_trackNhits", 50, 0, 50);
    htrkChi2  = dbe_->book1D("METTask_trackNormalizedChi2", "METTask_trackNormalizedChi2", 20, 0, 20);
    htrkD0    = dbe_->book1D("METTask_trackD0", "METTask_trackd0", 50, -1, 1);
    helePt    = dbe_->book1D("METTask_electronPt", "METTask_electronPt", 50, 0, 500);
    heleEta   = dbe_->book1D("METTask_electronEta", "METTask_electronEta", 60, -3.0, 3.0);
    heleHoE   = dbe_->book1D("METTask_electronHoverE", "METTask_electronHoverE", 25, 0, 0.5);
    hmuPt     = dbe_->book1D("METTask_muonPt", "METTask_muonPt", 50, 0, 500);
    hmuEta    = dbe_->book1D("METTask_muonEta", "METTask_muonEta", 60, -3.0, 3.0);
    hmuNhits  = dbe_->book1D("METTask_muonNhits", "METTask_muonNhits", 50, 0, 50);
    hmuChi2   = dbe_->book1D("METTask_muonNormalizedChi2", "METTask_muonNormalizedChi2", 20, 0, 20);
    hmuD0     = dbe_->book1D("METTask_muonD0", "METTask_muonD0", 50, -1, 1);

    hMETIonFeedbck      = dbe_->book1D("METTask_METIonFeedbck", "METTask_METIonFeedbck" ,200,0,1000);
    hMETHPDNoise        = dbe_->book1D("METTask_METHPDNoise",   "METTask_METHPDNoise"   ,200,0,1000);
    hMETRBXNoise        = dbe_->book1D("METTask_METRBXNoise",   "METTask_METRBXNoise"   ,200,0,1000);
    hMExCorrection       = dbe_->book1D("METTask_MExCorrection", "METTask_MExCorrection", 100, -500.0,500.0);
    hMEyCorrection       = dbe_->book1D("METTask_MEyCorrection", "METTask_MEyCorrection", 100, -500.0,500.0);
    hMuonCorrectionFlag  = dbe_->book1D("METTask_CorrectionFlag","METTask_CorrectionFlag", 5, -0.5, 4.5);
  }



}

// ***********************************************************
void METAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  if ( highPtJetEventFlag_->on() ) highPtJetEventFlag_->initRun( iRun, iSetup );
  if ( lowPtJetEventFlag_ ->on() ) lowPtJetEventFlag_ ->initRun( iRun, iSetup );
  if ( minBiasEventFlag_  ->on() ) minBiasEventFlag_  ->initRun( iRun, iSetup );
  if ( highMETEventFlag_ ->on() ) highMETEventFlag_  ->initRun( iRun, iSetup );
  //  if ( _LowMETEventFlag   ->on() ) _LowMETEventFlag   ->initRun( iRun, iSetup );
  if ( eleEventFlag_      ->on() ) eleEventFlag_      ->initRun( iRun, iSetup );
  if ( muonEventFlag_     ->on() ) muonEventFlag_     ->initRun( iRun, iSetup );

  if (highPtJetEventFlag_->on() && highPtJetEventFlag_->expressionsFromDB(highPtJetEventFlag_->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    highPtJetExpr_ = highPtJetEventFlag_->expressionsFromDB(highPtJetEventFlag_->hltDBKey(), iSetup);
  if (lowPtJetEventFlag_->on() && lowPtJetEventFlag_->expressionsFromDB(lowPtJetEventFlag_->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    lowPtJetExpr_  = lowPtJetEventFlag_->expressionsFromDB(lowPtJetEventFlag_->hltDBKey(),   iSetup);
  if (highMETEventFlag_->on() && highMETEventFlag_->expressionsFromDB(highMETEventFlag_->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    highMETExpr_   = highMETEventFlag_->expressionsFromDB(highMETEventFlag_->hltDBKey(),     iSetup);
  //  if (_LowMETEventFlag->on() && _LowMETEventFlag->expressionsFromDB(_LowMETEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
  //    lowMETExpr_    = _LowMETEventFlag->expressionsFromDB(_LowMETEventFlag->hltDBKey(),       iSetup);
  if (muonEventFlag_->on() && muonEventFlag_->expressionsFromDB(muonEventFlag_->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    muonExpr_      = muonEventFlag_->expressionsFromDB(muonEventFlag_->hltDBKey(),           iSetup);
  if (eleEventFlag_->on() && eleEventFlag_->expressionsFromDB(eleEventFlag_->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    elecExpr_      = eleEventFlag_->expressionsFromDB(eleEventFlag_->hltDBKey(),             iSetup);
  if (minBiasEventFlag_->on() && minBiasEventFlag_->expressionsFromDB(minBiasEventFlag_->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    minbiasExpr_   = minBiasEventFlag_->expressionsFromDB(minBiasEventFlag_->hltDBKey(),     iSetup);

}

// ***********************************************************
void METAnalyzer::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup, DQMStore * dbe)
{

  //
  //--- Check the time length of the Run from the lumi section plots

  std::string dirName = FolderName_+metCollectionLabel_.label()+"/";
  dbe->setCurrentFolder(dirName);

  TH1F* tlumisec;

  MonitorElement *meLumiSec = dbe->get("aaa");
  meLumiSec = dbe->get("JetMET/lumisec");

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
  for (std::vector<std::string>::const_iterator ic = folderNames_.begin(); ic != folderNames_.end(); ic++)
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
  for (std::vector<std::string>::const_iterator ic = folderNames_.begin(); ic != folderNames_.end(); ic++)
    {

      std::string DirName;
      DirName = dirName+*ic;

      makeRatePlot(DirName,totltime);
      if ( highPtJetEventFlag_->on() )
	makeRatePlot(DirName+"/"+"triggerName_HighJetPt",totltime);
      if ( lowPtJetEventFlag_->on() )
	makeRatePlot(DirName+"/"+"triggerName_LowJetPt",totltime);
      if ( minBiasEventFlag_->on() )
	makeRatePlot(DirName+"/"+"triggerName_MinBias",totltime);
      if ( highMETEventFlag_->on() )
	makeRatePlot(DirName+"/"+"triggerName_HighMET",totltime);
      //      if ( _LowMETEventFlag->on() )
      //	makeRatePlot(DirName+"/"+"triggerName_LowMET",totltime);
      if ( eleEventFlag_->on() )
	makeRatePlot(DirName+"/"+"triggerName_Ele",totltime);
      if ( muonEventFlag_->on() )
	makeRatePlot(DirName+"/"+"triggerName_Muon",totltime);
    }
}


// ***********************************************************
void METAnalyzer::makeRatePlot(std::string DirName, double totltime)
{

  dbe_->setCurrentFolder(DirName);
  MonitorElement *meMET = dbe_->get(DirName+"/"+"METTask_MET");

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
      hMETRate      = dbe_->book1D("METTask_METRate",tMETRate);
    }
}

// ***********************************************************
void METAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  if (verbose_) std::cout << "METAnalyzer analyze" << std::endl;

  std::string DirName = FolderName_+metCollectionLabel_.label();


  hmetME->Fill(2);

  // ==========================================================
  // Trigger information
  //
  trigJetMB_=0;
  trigHighPtJet_=0;
  trigLowPtJet_=0;
  trigMinBias_=0;
  trigHighMET_=0;
  //  _trig_LowMET=0;
  trigEle_=0;
  trigMuon_=0;
  trigPhysDec_=0;

  // **** Get the TriggerResults container
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(triggerResultsToken_, triggerResults);

  if( triggerResults.isValid()) {

    /////////// Analyzing HLT Trigger Results (TriggerResults) //////////

    //
    //
    // Check how many HLT triggers are in triggerResults
    int ntrigs = (*triggerResults).size();
    if (verbose_) std::cout << "ntrigs=" << ntrigs << std::endl;

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
	  trigHighPtJet_=true;
        else if (triggerNames.triggerName(i).find(lowPtJetExpr_[0].substr(0,lowPtJetExpr_[0].rfind("_v")+2))!=std::string::npos && (*triggerResults).accept(i))
	  trigLowPtJet_=true;
        else if (triggerNames.triggerName(i).find(highMETExpr_[0].substr(0,highMETExpr_[0].rfind("_v")+2))!=std::string::npos && (*triggerResults).accept(i))
	  trigHighMET_=true;
	//        else if (triggerNames.triggerName(i).find(lowMETExpr_[0].substr(0,lowMETExpr_[0].rfind("_v")+2))!=std::string::npos && (*triggerResults).accept(i))
	//	  _trig_LowMET=true;
        else if (triggerNames.triggerName(i).find(muonExpr_[0].substr(0,muonExpr_[0].rfind("_v")+2))!=std::string::npos && (*triggerResults).accept(i))
	  trigMuon_=true;
        else if (triggerNames.triggerName(i).find(elecExpr_[0].substr(0,elecExpr_[0].rfind("_v")+2))!=std::string::npos && (*triggerResults).accept(i))
	  trigEle_=true;
        else if (triggerNames.triggerName(i).find(minbiasExpr_[0].substr(0,minbiasExpr_[0].rfind("_v")+2))!=std::string::npos && (*triggerResults).accept(i))
	  trigMinBias_=true;
      }

    // count number of requested Jet or MB HLT paths which have fired
    for (unsigned int i=0; i!=HLTPathsJetMBByName_.size(); i++) {
      unsigned int triggerIndex = triggerNames.triggerIndex(HLTPathsJetMBByName_[i]);
      if (triggerIndex<(*triggerResults).size()) {
	if ((*triggerResults).accept(triggerIndex)) {
	  trigJetMB_++;
	}
      }
    }
    // for empty input vectors (n==0), take all HLT triggers!
    if (HLTPathsJetMBByName_.size()==0) trigJetMB_=(*triggerResults).size()-1;

    /*that is what we had in TCMet
    //
    if (verbose_) std::cout << "triggerNames size" << " " << triggerNames.size() << std::endl;
    if (verbose_) std::cout << _hlt_HighPtJet << " " << triggerNames.triggerIndex(_hlt_HighPtJet) << std::endl;
    if (verbose_) std::cout << _hlt_LowPtJet  << " " << triggerNames.triggerIndex(_hlt_LowPtJet)  << std::endl;
    if (verbose_) std::cout << _hlt_HighMET   << " " << triggerNames.triggerIndex(_hlt_HighMET)   << std::endl;
    //    if (verbose_) std::cout << _hlt_LowMET    << " " << triggerNames.triggerIndex(_hlt_LowMET)    << std::endl;
    if (verbose_) std::cout << _hlt_Ele       << " " << triggerNames.triggerIndex(_hlt_Ele)       << std::endl;
    if (verbose_) std::cout << _hlt_Muon      << " " << triggerNames.triggerIndex(_hlt_Muon)      << std::endl;

    if (triggerNames.triggerIndex(_hlt_HighPtJet) != triggerNames.size() &&
	(*triggerResults).accept(triggerNames.triggerIndex(_hlt_HighPtJet))) trigHighPtJet_=1;

    if (triggerNames.triggerIndex(_hlt_LowPtJet)  != triggerNames.size() &&
	(*triggerResults).accept(triggerNames.triggerIndex(_hlt_LowPtJet)))  trigLowPtJet_=1;

    if (triggerNames.triggerIndex(_hlt_HighMET)   != triggerNames.size() &&
        (*triggerResults).accept(triggerNames.triggerIndex(_hlt_HighMET)))   trigHighMET_=1;

    //    if (triggerNames.triggerIndex(_hlt_LowMET)    != triggerNames.size() &&
    //        (*triggerResults).accept(triggerNames.triggerIndex(_hlt_LowMET)))    _trig_LowMET=1;

    if (triggerNames.triggerIndex(_hlt_Ele)       != triggerNames.size() &&
        (*triggerResults).accept(triggerNames.triggerIndex(_hlt_Ele)))       trigEle_=1;

    if (triggerNames.triggerIndex(_hlt_Muon)      != triggerNames.size() &&
        (*triggerResults).accept(triggerNames.triggerIndex(_hlt_Muon)))      trigMuon_=1;
    */

    /* commented out in METAnalyzer
      if ( highPtJetEventFlag_->on() && highPtJetEventFlag_->accept( iEvent, iSetup) )
      trigHighPtJet_=1;

      if ( lowPtJetEventFlag_->on() && lowPtJetEventFlag_->accept( iEvent, iSetup) )
      trigLowPtJet_=1;

      if ( minBiasEventFlag_->on() && minBiasEventFlag_->accept( iEvent, iSetup) )
      trigMinBias_=1;

      if ( highMETEventFlag->on() && highMETEventFlag->accept( iEvent, iSetup) )
      trigHighMET_=1;

      if ( _LowMETEventFlag->on() && _LowMETEventFlag->accept( iEvent, iSetup) )
      _trig_LowMET=1;

      if ( eleEventFlag->on() && eleEventFlag->accept( iEvent, iSetup) )
      trigEle_=1;

      if ( muonEventFlag->on() && muonEventFlag->accept( iEvent, iSetup) )
      trigMuon_=1;
    */
    if (triggerNames.triggerIndex(hltPhysDec_)   != triggerNames.size() &&
	(*triggerResults).accept(triggerNames.triggerIndex(hltPhysDec_)))   trigPhysDec_=1;
  } else {

    edm::LogInfo("MetAnalyzer") << "TriggerResults::HLT not found, "
      "automatically select events";

    // TriggerResults object not found. Look at all events.
    trigJetMB_=1;
  }

  // ==========================================================
  // MET information

  // **** Get the MET container
  edm::Handle<reco::METCollection> tcmetcoll;
  edm::Handle<reco::CaloMETCollection> calometcoll;
  edm::Handle<reco::PFMETCollection> pfmetcoll;

  if(isTCMet_){
    iEvent.getByToken(tcMetToken_, tcmetcoll);
    if(isTCMet_ && !tcmetcoll.isValid()) return;
  }
  if(isCaloMet_){
    iEvent.getByToken(caloMetToken_, calometcoll);
    if(isCaloMet_ && !calometcoll.isValid()) return;
  }
  if(isPFMet_){
    iEvent.getByToken(pfMetToken_, pfmetcoll);
    if(isPFMet_ && !pfmetcoll.isValid()) return;
  }

  const MET *met=NULL;
  const PFMET *pfmet=NULL;
  const CaloMET *calomet=NULL;
  if(isTCMet_){
    met=&(tcmetcoll->front());
  }
  if(isPFMet_){
    met=&(pfmetcoll->front());
    pfmet=&(pfmetcoll->front());
  }
  if(isCaloMet_){
    met=&(calometcoll->front());
    calomet=&(calometcoll->front());
  }

  LogTrace(metname)<<"[METAnalyzer] Call to the MET analyzer";

  // ==========================================================
  // TCMET

  

  if (isTCMet_ || (isCaloMet_ && metCollectionLabel_.label() == "corMetGlobalMuons")) {

    iEvent.getByToken(MuonToken_, muonHandle_);
    iEvent.getByToken(TrackToken_, trackHandle_);
    iEvent.getByToken(ElectronToken_, electronHandle_);
    iEvent.getByToken(BeamspotToken_, beamSpotHandle_);
    iEvent.getByToken(tcMETValueMapToken_,tcMetValueMapHandle_);

    if(!muonHandle_.isValid())     edm::LogInfo("OutputInfo") << "falied to retrieve muon data require by MET Task";
    if(!trackHandle_.isValid())    edm::LogInfo("OutputInfo") << "falied to retrieve track data require by MET Task";
    if(!electronHandle_.isValid()) edm::LogInfo("OutputInfo") << "falied to retrieve electron data require by MET Task";
    if(!beamSpotHandle_.isValid()) edm::LogInfo("OutputInfo") << "falied to retrieve beam spot data require by MET Task";

    beamSpot_ = ( beamSpotHandle_.isValid() ) ? beamSpotHandle_->position() : math::XYZPoint(0, 0, 0);

  }

  // ==========================================================
  //

  edm::Handle<HcalNoiseRBXCollection> HRBXCollection;
  iEvent.getByToken(HcalNoiseRBXToken_,HRBXCollection);
  if (!HRBXCollection.isValid()) {
    LogDebug("") << "METAnalyzer: Could not find HcalNoiseRBX Collection" << std::endl;
    if (verbose_) std::cout << "METAnalyzer: Could not find HcalNoiseRBX Collection" << std::endl;
  }


  edm::Handle<bool> HBHENoiseFilterResultHandle;
  iEvent.getByToken(hbheNoiseFilterResultToken_, HBHENoiseFilterResultHandle);
  bool HBHENoiseFilterResult = *HBHENoiseFilterResultHandle;
  if (!HBHENoiseFilterResultHandle.isValid()) {
    LogDebug("") << "METAnalyzer: Could not find HBHENoiseFilterResult" << std::endl;
    if (verbose_) std::cout << "METAnalyzer: Could not find HBHENoiseFilterResult" << std::endl;
  }

  bool bJetIDMinimal=true;
  bool bJetIDLoose=true;
  bool bJetIDTight=true;

  if(isCaloMet_){
    edm::Handle<reco::CaloJetCollection> caloJets;
    iEvent.getByToken(caloJetsToken_, caloJets);
    if (!caloJets.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
      if (verbose_) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
    }
    

    // JetID
    
    if (verbose_) std::cout << "caloJet JetID starts" << std::endl;
    
    //
    // --- Minimal cuts
    //

 
    for (reco::CaloJetCollection::const_iterator cal = caloJets->begin();
	 cal!=caloJets->end(); ++cal){
      jetID_->calculate(iEvent, *cal);
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
      jetID_->calculate(iEvent, *cal);
      if (verbose_) std::cout << jetID_->n90Hits() << " "
			      << jetID_->restrictedEMF() << " "
			      << cal->pt() << std::endl;
      if (cal->pt()>10.){
	//
	// for all regions
	if (jetID_->n90Hits()<2)  bJetIDLoose=false;
	if (jetID_->fHPD()>=0.98) bJetIDLoose=false;
	//if (jetID_->restrictedEMF()<0.01) bJetIDLoose=false;
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
      jetID_->calculate(iEvent, *cal);
      if (cal->pt()>25.){
	//
	// for all regions
	if (jetID_->fHPD()>=0.95) bJetIDTight=false;
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
    
    if (verbose_) std::cout << "caloJet JetID ends" << std::endl;
  }
  if(isTCMet_){
    edm::Handle<reco::JPTJetCollection> jptJets;
    iEvent.getByToken(jptJetsToken_, jptJets);
    if (!jptJets.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find JPT jet product" << std::endl;
      if (verbose_) std::cout << "METAnalyzer: Could not find JPT jet product" << std::endl;
    }

    for (reco::JPTJetCollection::const_iterator jpt = jptJets->begin();
	 jpt!=jptJets->end(); ++jpt){
      const edm::RefToBase<reco::Jet>&  rawJet = jpt->getCaloJetRef();
      const reco::CaloJet *rawCaloJet = dynamic_cast<const reco::CaloJet*>(&*rawJet);
      
      jetID_->calculate(iEvent, *rawCaloJet);
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
      jetID_->calculate(iEvent,*rawCaloJet);
      if (verbose_) std::cout << jetID_->n90Hits() << " "
			      << jetID_->restrictedEMF() << " "
			      << jpt->pt() << std::endl;
      //calculate ID for JPT jets above 10
      if (jpt->pt()>10.){
	//for ID itself calojet is the relevant quantity
	// for all regions
	if (jetID_->n90Hits()<2)  bJetIDLoose=false;
	if (jetID_->fHPD()>=0.98) bJetIDLoose=false;
	//if (jetID_->restrictedEMF()<0.01) bJetIDLoose=false;
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
      jetID_->calculate(iEvent, *rawCaloJet);
      if (jpt->pt()>25.){
	//
	// for all regions
	if (jetID_->fHPD()>=0.95) bJetIDTight=false;
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
    if(isPFMet_){
      edm::Handle<reco::PFJetCollection> pfJets;
      iEvent.getByToken(pfJetsToken_, pfJets);
      if (!pfJets.isValid()) {
	LogDebug("") << "METAnalyzer: Could not find PF jet product" << std::endl;
	if (verbose_) std::cout << "METAnalyzer: Could not find PF jet product" << std::endl;
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
    if (verbose_) std::cout << "TCMET JetID ends" << std::endl;
    }
  }

  // ==========================================================
  // HCAL Noise filter

  bool bHcalNoiseFilter = HBHENoiseFilterResult;

  // ==========================================================
  // Get BeamHaloSummary
  edm::Handle<BeamHaloSummary> TheBeamHaloSummary ;
  iEvent.getByToken(beamHaloSummaryToken_, TheBeamHaloSummary) ;

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
  Handle<VertexCollection> vertexHandle;
  iEvent.getByToken(vertexToken_, vertexHandle);

  if (!vertexHandle.isValid()) {
    LogDebug("") << "CaloMETAnalyzer: Could not find vertex collection" << std::endl;
    if (verbose_) std::cout << "CaloMETAnalyzer: Could not find vertex collection" << std::endl;
  }
  numPV_ = 0;
  if ( vertexHandle.isValid() ){
    VertexCollection vertexCollection = *(vertexHandle.product());
    numPV_  = vertexCollection.size();
  }
  bool bPrimaryVertex = (bypassAllPVChecks_ || (numPV_>0));
  // ==========================================================

  edm::Handle< L1GlobalTriggerReadoutRecord > gtReadoutRecord;
  iEvent.getByToken( gtToken_, gtReadoutRecord);

  if (!gtReadoutRecord.isValid()) {
    LogDebug("") << "CaloMETAnalyzer: Could not find GT readout record" << std::endl;
    if (verbose_) std::cout << "CaloMETAnalyzer: Could not find GT readout record product" << std::endl;
  }
  // ==========================================================
  // Reconstructed MET Information - fill MonitorElements

  bool bHcalNoise   = bHcalNoiseFilter;
  bool bBeamHaloID  = bBeamHaloIDLoosePass;
  bool bJetID       = true;

  if      (tightBHFiltering_)         bBeamHaloID = bBeamHaloIDTightPass;
  if      (tightJetIDFiltering_==1)   bJetID      = bJetIDMinimal;
  else if (tightJetIDFiltering_==2)   bJetID      = bJetIDLoose;
  else if (tightJetIDFiltering_==3)   bJetID      = bJetIDTight;
  else if (tightJetIDFiltering_==-1)  bJetID      = true;

  bool bBasicCleanup = bPrimaryVertex;
  bool bExtraCleanup = bBasicCleanup && bHcalNoise && bJetID && bBeamHaloID;


  for (std::vector<std::string>::const_iterator ic = folderNames_.begin();
       ic != folderNames_.end(); ic++){
    if (*ic=="All")                                             fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet);
    if (DCSFilter_->filter(iEvent, iSetup)) {
      if (cleanupSelection_){
	if (*ic=="BasicCleanup" && bBasicCleanup)                   fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet);
	if (*ic=="ExtraCleanup" && bExtraCleanup)                   fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet);
      }
//      if (allSelection_) {
//	if (*ic=="HcalNoiseFilter"      && bHcalNoiseFilter )       fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet);
//	if (*ic=="JetIDMinimal"         && bJetIDMinimal)           fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet);
//	if (*ic=="JetIDLoose"           && bJetIDLoose)             fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet);
//	if (*ic=="JetIDTight"           && bJetIDTight)             fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet);
//	if (*ic=="BeamHaloIDTightPass"  && bBeamHaloIDTightPass)    fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet);
//	if (*ic=="BeamHaloIDLoosePass"  && bBeamHaloIDLoosePass)    fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet);
//	if (*ic=="PV"                   && bPrimaryVertex)          fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet);
//      }
    } // DCS
  }
}


// ***********************************************************
void METAnalyzer::fillMESet(const edm::Event& iEvent, std::string DirName,
			    const reco::MET& met, const reco::PFMET& pfmet, const reco::CaloMET& calomet)
{

  dbe_->setCurrentFolder(DirName);

  bool bLumiSecPlot=false;
  if (DirName.find("All")) bLumiSecPlot=true;

  if (trigJetMB_)
    fillMonitorElement(iEvent,DirName,"",met,pfmet,calomet, bLumiSecPlot);
  if (trigHighPtJet_)
    fillMonitorElement(iEvent,DirName,"HighPtJet",met,pfmet,calomet,false);
  if (trigLowPtJet_)
    fillMonitorElement(iEvent,DirName,"LowPtJet",met,pfmet,calomet,false);
  if (trigMinBias_)
    fillMonitorElement(iEvent,DirName,"MinBias",met,pfmet,calomet,false);
  if (trigHighMET_)
    fillMonitorElement(iEvent,DirName,"HighMET",met,pfmet,calomet,false);
  //  if (_trig_LowMET)
  //    fillMonitorElement(iEvent,DirName,"LowMET",met,pfmet,calomet,false);
  if (trigEle_)
    fillMonitorElement(iEvent,DirName,"Ele",met,pfmet,calomet,false);
  if (trigMuon_)
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
    if (met.pt()<highMETThreshold_) return;
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
  myLuminosityBlock = iEvent.luminosityBlock();
  //

  if (TriggerTypeName!="") DirName = DirName +"/"+TriggerTypeName;

  if (verbose_) std::cout << "etThreshold_ = " << etThreshold_ << std::endl;

  if (SumET>etThreshold_){
    hMEx    = dbe_->get(DirName+"/"+"METTask_MEx");     if (hMEx           && hMEx->getRootObject()){     hMEx          ->Fill(MEx);}
    hMEy    = dbe_->get(DirName+"/"+"METTask_MEy");     if (hMEy           && hMEy->getRootObject())     hMEy          ->Fill(MEy);
    hMET    = dbe_->get(DirName+"/"+"METTask_MET");     if (hMET           && hMET->getRootObject())     hMET          ->Fill(MET);
    hMETPhi = dbe_->get(DirName+"/"+"METTask_METPhi");  if (hMETPhi        && hMETPhi->getRootObject())  hMETPhi       ->Fill(METPhi);
    hSumET  = dbe_->get(DirName+"/"+"METTask_SumET");   if (hSumET         && hSumET->getRootObject())   hSumET        ->Fill(SumET);
    hMETSig = dbe_->get(DirName+"/"+"METTask_METSig");  if (hMETSig        && hMETSig->getRootObject())  hMETSig       ->Fill(METSig);
    //hEz     = dbe_->get(DirName+"/"+"METTask_Ez");      if (hEz            && hEz->getRootObject())      hEz           ->Fill(Ez);

    hMET_logx   = dbe_->get(DirName+"/"+"METTask_MET_logx");    if (hMET_logx      && hMET_logx->getRootObject())    hMET_logx->Fill(log10(MET));
    hSumET_logx = dbe_->get(DirName+"/"+"METTask_SumET_logx");  if (hSumET_logx    && hSumET_logx->getRootObject())  hSumET_logx->Fill(log10(SumET));

    // Fill NPV profiles
      //--------------------------------------------------------------------------
    meMEx_profile   = dbe_->get(DirName + "/METTask_MEx_profile");
    meMEy_profile   = dbe_->get(DirName + "/METTask_MEy_profile");
    meMET_profile   = dbe_->get(DirName + "/METTask_MET_profile");
    meSumET_profile = dbe_->get(DirName + "/METTask_SumET_profile");
    
    if (meMEx_profile   && meMEx_profile  ->getRootObject()) meMEx_profile  ->Fill(numPV_, MEx);
    if (meMEy_profile   && meMEy_profile  ->getRootObject()) meMEy_profile  ->Fill(numPV_, MEy);
    if (meMET_profile   && meMET_profile  ->getRootObject()) meMET_profile  ->Fill(numPV_, MET);
    if (meSumET_profile && meSumET_profile->getRootObject()) meSumET_profile->Fill(numPV_, SumET);
 

    //hMETIonFeedbck = dbe_->get(DirName+"/"+"METTask_METIonFeedbck");  if (hMETIonFeedbck && hMETIonFeedbck->getRootObject())  hMETIonFeedbck->Fill(MET);
    //hMETHPDNoise   = dbe_->get(DirName+"/"+"METTask_METHPDNoise");    if (hMETHPDNoise   && hMETHPDNoise->getRootObject())    hMETHPDNoise->Fill(MET);
    //comment out like already done before for TcMET and PFMET
    if(isTCMet_ || metCollectionLabel_.label() == "corMetGlobalMuons"){
      hMETIonFeedbck = dbe_->get(DirName+"/"+"METTask_METIonFeedbck");  if (hMETIonFeedbck && hMETIonFeedbck->getRootObject()) hMETIonFeedbck->Fill(MET);
      hMETHPDNoise   = dbe_->get(DirName+"/"+"METTask_METHPDNoise");    if (hMETHPDNoise   && hMETHPDNoise->getRootObject())   hMETHPDNoise->Fill(MET);
      hMETRBXNoise   = dbe_->get(DirName+"/"+"METTask_METRBXNoise");    if (hMETRBXNoise   && hMETRBXNoise->getRootObject())   hMETRBXNoise->Fill(MET);
    }


    if(isCaloMet_){
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

      hCaloMaxEtInEmTowers  = dbe_->get(DirName+"/"+"METTask_CaloMaxEtInEmTowers");   if (hCaloMaxEtInEmTowers  && hCaloMaxEtInEmTowers->getRootObject())   hCaloMaxEtInEmTowers->Fill(caloMaxEtInEMTowers);
      hCaloMaxEtInHadTowers = dbe_->get(DirName+"/"+"METTask_CaloMaxEtInHadTowers");  if (hCaloMaxEtInHadTowers && hCaloMaxEtInHadTowers->getRootObject())  hCaloMaxEtInHadTowers->Fill(caloMaxEtInHadTowers);
      
      hCaloHadEtInHB = dbe_->get(DirName+"/"+"METTask_CaloHadEtInHB");  if (hCaloHadEtInHB  &&  hCaloHadEtInHB->getRootObject())  hCaloHadEtInHB->Fill(caloHadEtInHB);
      hCaloHadEtInHO = dbe_->get(DirName+"/"+"METTask_CaloHadEtInHO");  if (hCaloHadEtInHO  &&  hCaloHadEtInHO->getRootObject())  hCaloHadEtInHO->Fill(caloHadEtInHO);
      hCaloHadEtInHE = dbe_->get(DirName+"/"+"METTask_CaloHadEtInHE");  if (hCaloHadEtInHE  &&  hCaloHadEtInHE->getRootObject())  hCaloHadEtInHE->Fill(caloHadEtInHE);
      hCaloHadEtInHF = dbe_->get(DirName+"/"+"METTask_CaloHadEtInHF");  if (hCaloHadEtInHF  &&  hCaloHadEtInHF->getRootObject())  hCaloHadEtInHF->Fill(caloHadEtInHF);
      hCaloEmEtInEB  = dbe_->get(DirName+"/"+"METTask_CaloEmEtInEB");   if (hCaloEmEtInEB   &&  hCaloEmEtInEB->getRootObject())   hCaloEmEtInEB->Fill(caloEmEtInEB);
      hCaloEmEtInEE  = dbe_->get(DirName+"/"+"METTask_CaloEmEtInEE");   if (hCaloEmEtInEE   &&  hCaloEmEtInEE->getRootObject())   hCaloEmEtInEE->Fill(caloEmEtInEE);
      hCaloEmEtInHF  = dbe_->get(DirName+"/"+"METTask_CaloEmEtInHF");   if (hCaloEmEtInHF   &&  hCaloEmEtInHF->getRootObject())   hCaloEmEtInHF->Fill(caloEmEtInHF);

      hCaloMETPhi020 = dbe_->get(DirName+"/"+"METTask_CaloMETPhi020");    if (MET> 20. && hCaloMETPhi020  &&  hCaloMETPhi020->getRootObject()) { hCaloMETPhi020->Fill(METPhi);}


      hCaloEtFractionHadronic = dbe_->get(DirName+"/"+"METTask_CaloEtFractionHadronic"); if (hCaloEtFractionHadronic && hCaloEtFractionHadronic->getRootObject())  hCaloEtFractionHadronic->Fill(caloEtFractionHadronic);
      hCaloEmEtFraction       = dbe_->get(DirName+"/"+"METTask_CaloEmEtFraction");       if (hCaloEmEtFraction       && hCaloEmEtFraction->getRootObject())        hCaloEmEtFraction->Fill(caloEmEtFraction);
      hCaloEmEtFraction020 = dbe_->get(DirName+"/"+"METTask_CaloEmEtFraction020");       if (MET> 20.  &&  hCaloEmEtFraction020    && hCaloEmEtFraction020->getRootObject()) hCaloEmEtFraction020->Fill(caloEmEtFraction);
      if (metCollectionLabel_.label() == "corMetGlobalMuons" ) {
	
	for( reco::MuonCollection::const_iterator muonit = muonHandle_->begin(); muonit != muonHandle_->end(); muonit++ ) {
	  const reco::TrackRef siTrack = muonit->innerTrack();
	  hCalomuPt    = dbe_->get(DirName+"/"+"METTask_CalomuonPt");  
	  if (hCalomuPt    && hCalomuPt->getRootObject())   hCalomuPt->Fill( muonit->p4().pt() );
	  hCalomuEta   = dbe_->get(DirName+"/"+"METTask_CalomuonEta");    if (hCalomuEta   && hCalomuEta->getRootObject())    hCalomuEta->Fill( muonit->p4().eta() );
	  hCalomuNhits = dbe_->get(DirName+"/"+"METTask_CalomuonNhits");  if (hCalomuNhits && hCalomuNhits->getRootObject())  hCalomuNhits->Fill( siTrack.isNonnull() ? siTrack->numberOfValidHits() : -999 );
	  hCalomuChi2  = dbe_->get(DirName+"/"+"METTask_CalomuonNormalizedChi2");   if (hCalomuChi2  && hCalomuChi2->getRootObject())   hCalomuChi2->Fill( siTrack.isNonnull() ? siTrack->chi2()/siTrack->ndof() : -999 );
	  double d0 = siTrack.isNonnull() ? -1 * siTrack->dxy( beamSpot_) : -999;
	  hCalomuD0    = dbe_->get(DirName+"/"+"METTask_CalomuonD0");     if (hCalomuD0    && hCalomuD0->getRootObject())  hCalomuD0->Fill( d0 );
	}
	
	const unsigned int nMuons = muonHandle_->size();
	for( unsigned int mus = 0; mus < nMuons; mus++ ) {
	  reco::MuonRef muref( muonHandle_, mus);
	  reco::MuonMETCorrectionData muCorrData = (*tcMetValueMapHandle_)[muref];
	  hCaloMExCorrection      = dbe_->get(DirName+"/"+"METTask_CaloMExCorrection");       if (hCaloMExCorrection      && hCaloMExCorrection->getRootObject())       hCaloMExCorrection-> Fill(muCorrData.corrY());
	  hCaloMEyCorrection      = dbe_->get(DirName+"/"+"METTask_CaloMEyCorrection");       if (hCaloMEyCorrection      && hCaloMEyCorrection->getRootObject())       hCaloMEyCorrection-> Fill(muCorrData.corrX());
	  hCaloMuonCorrectionFlag = dbe_->get(DirName+"/"+"METTask_CaloMuonCorrectionFlag");  if (hCaloMuonCorrectionFlag && hCaloMuonCorrectionFlag->getRootObject())  hCaloMuonCorrectionFlag-> Fill(muCorrData.type());
	}
      } 
    }

    if(isPFMet_){
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
      
      mePhotonEtFraction        = dbe_->get(DirName + "/METTask_PfPhotonEtFraction");
      mePhotonEt                = dbe_->get(DirName + "/METTask_PfPhotonEt");
      meNeutralHadronEtFraction = dbe_->get(DirName + "/METTask_PfNeutralHadronEtFraction");
      meNeutralHadronEt         = dbe_->get(DirName + "/METTask_PfNeutralHadronEt");
      meElectronEtFraction      = dbe_->get(DirName + "/METTask_PfElectronEtFraction");
      meElectronEt              = dbe_->get(DirName + "/METTask_PfElectronEt");
      meChargedHadronEtFraction = dbe_->get(DirName + "/METTask_PfChargedHadronEtFraction");
      meChargedHadronEt         = dbe_->get(DirName + "/METTask_PfChargedHadronEt");
      meMuonEtFraction          = dbe_->get(DirName + "/METTask_PfMuonEtFraction");
      meMuonEt                  = dbe_->get(DirName + "/METTask_PfMuonEt");
      meHFHadronEtFraction      = dbe_->get(DirName + "/METTask_PfHFHadronEtFraction");
      meHFHadronEt              = dbe_->get(DirName + "/METTask_PfHFHadronEt");
      meHFEMEtFraction          = dbe_->get(DirName + "/METTask_PfHFEMEtFraction");
      meHFEMEt                  = dbe_->get(DirName + "/METTask_PfHFEMEt");
      
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
      
      mePhotonEtFraction_profile        = dbe_->get(DirName + "/METTask_PfPhotonEtFraction_profile");
      mePhotonEt_profile                = dbe_->get(DirName + "/METTask_PfPhotonEt_profile");
      meNeutralHadronEtFraction_profile = dbe_->get(DirName + "/METTask_PfNeutralHadronEtFraction_profile");
      meNeutralHadronEt_profile         = dbe_->get(DirName + "/METTask_PfNeutralHadronEt_profile");
      meElectronEtFraction_profile      = dbe_->get(DirName + "/METTask_PfElectronEtFraction_profile");
      meElectronEt_profile              = dbe_->get(DirName + "/METTask_PfElectronEt_profile");
      meChargedHadronEtFraction_profile = dbe_->get(DirName + "/METTask_PfChargedHadronEtFraction_profile");
      meChargedHadronEt_profile         = dbe_->get(DirName + "/METTask_PfChargedHadronEt_profile");
      meMuonEtFraction_profile          = dbe_->get(DirName + "/METTask_PfMuonEtFraction_profile");
      meMuonEt_profile                  = dbe_->get(DirName + "/METTask_PfMuonEt_profile");
      meHFHadronEtFraction_profile      = dbe_->get(DirName + "/METTask_PfHFHadronEtFraction_profile");
      meHFHadronEt_profile              = dbe_->get(DirName + "/METTask_PfHFHadronEt_profile");
      meHFEMEtFraction_profile          = dbe_->get(DirName + "/METTask_PfHFEMEtFraction_profile");
      meHFEMEt_profile                  = dbe_->get(DirName + "/METTask_PfHFEMEt_profile");
      
      if (mePhotonEtFraction_profile        && mePhotonEtFraction_profile       ->getRootObject()) mePhotonEtFraction_profile       ->Fill(numPV_, pfPhotonEtFraction);
      if (mePhotonEt_profile                && mePhotonEt_profile               ->getRootObject()) mePhotonEt_profile               ->Fill(numPV_, pfPhotonEt);
      if (meNeutralHadronEtFraction_profile && meNeutralHadronEtFraction_profile->getRootObject()) meNeutralHadronEtFraction_profile->Fill(numPV_, pfNeutralHadronEtFraction);
      if (meNeutralHadronEt_profile         && meNeutralHadronEt_profile        ->getRootObject()) meNeutralHadronEt_profile        ->Fill(numPV_, pfNeutralHadronEt);
      if (meElectronEtFraction_profile      && meElectronEtFraction_profile     ->getRootObject()) meElectronEtFraction_profile     ->Fill(numPV_, pfElectronEtFraction);
      if (meElectronEt_profile              && meElectronEt_profile             ->getRootObject()) meElectronEt_profile             ->Fill(numPV_, pfElectronEt);
      if (meChargedHadronEtFraction_profile && meChargedHadronEtFraction_profile->getRootObject()) meChargedHadronEtFraction_profile->Fill(numPV_, pfChargedHadronEtFraction);
      if (meChargedHadronEt_profile         && meChargedHadronEt_profile        ->getRootObject()) meChargedHadronEt_profile        ->Fill(numPV_, pfChargedHadronEt);
      if (meMuonEtFraction_profile          && meMuonEtFraction_profile         ->getRootObject()) meMuonEtFraction_profile         ->Fill(numPV_, pfMuonEtFraction);
      if (meMuonEt_profile                  && meMuonEt_profile                 ->getRootObject()) meMuonEt_profile                 ->Fill(numPV_, pfMuonEt);
      if (meHFHadronEtFraction_profile      && meHFHadronEtFraction_profile     ->getRootObject()) meHFHadronEtFraction_profile     ->Fill(numPV_, pfHFHadronEtFraction);
      if (meHFHadronEt_profile              && meHFHadronEt_profile             ->getRootObject()) meHFHadronEt_profile             ->Fill(numPV_, pfHFHadronEt);
      if (meHFEMEtFraction_profile          && meHFEMEtFraction_profile         ->getRootObject()) meHFEMEtFraction_profile         ->Fill(numPV_, pfHFEMEtFraction);
      if (meHFEMEt_profile                  && meHFEMEt_profile                 ->getRootObject()) meHFEMEt_profile                 ->Fill(numPV_, pfHFEMEt);
    }

    if (isCaloMet_){
      if (bLumiSecPlot){
	hMExLS = dbe_->get(DirName+"/"+"METTask_MExLS"); if (hMExLS  &&  hMExLS->getRootObject())   hMExLS->Fill(MEx,myLuminosityBlock);
	hMEyLS = dbe_->get(DirName+"/"+"METTask_MEyLS"); if (hMEyLS  &&  hMEyLS->getRootObject())   hMEyLS->Fill(MEy,myLuminosityBlock);
      }
    } 

    ////////////////////////////////////
    if (isTCMet_) {

      if(trackHandle_.isValid()) {
	for( edm::View<reco::Track>::const_iterator trkit = trackHandle_->begin(); trkit != trackHandle_->end(); trkit++ ) {
	  htrkPt    = dbe_->get(DirName+"/"+"METTask_trackPt");     if (htrkPt    && htrkPt->getRootObject())     htrkPt->Fill( trkit->pt() );
	  htrkEta   = dbe_->get(DirName+"/"+"METTask_trackEta");    if (htrkEta   && htrkEta->getRootObject())    htrkEta->Fill( trkit->eta() );
	  htrkNhits = dbe_->get(DirName+"/"+"METTask_trackNhits");  if (htrkNhits && htrkNhits->getRootObject())  htrkNhits->Fill( trkit->numberOfValidHits() );
	  htrkChi2  = dbe_->get(DirName+"/"+"METTask_trackNormalizedChi2");  
	  if (htrkChi2  && htrkChi2->getRootObject())   htrkChi2->Fill( trkit->chi2() / trkit->ndof() );
	  double d0 = -1 * trkit->dxy( beamSpot_ );
	  htrkD0    = dbe_->get(DirName+"/"+"METTask_trackD0");     if (htrkD0 && htrkD0->getRootObject())        htrkD0->Fill( d0 );
	}
      }else{std::cout<<"tracks not valid"<<std::endl;}

      if(electronHandle_.isValid()) {
	for( edm::View<reco::GsfElectron>::const_iterator eleit = electronHandle_->begin(); eleit != electronHandle_->end(); eleit++ ) {
	  helePt  = dbe_->get(DirName+"/"+"METTask_electronPt");   if (helePt  && helePt->getRootObject())   helePt->Fill( eleit->p4().pt() );
	  heleEta = dbe_->get(DirName+"/"+"METTask_electronEta");  if (heleEta && heleEta->getRootObject())  heleEta->Fill( eleit->p4().eta() );
	  heleHoE = dbe_->get(DirName+"/"+"METTask_electronHoverE");  if (heleHoE && heleHoE->getRootObject())  heleHoE->Fill( eleit->hadronicOverEm() );
	}
      }else{
	std::cout<<"electrons not valid"<<std::endl;
      }

      if(muonHandle_.isValid()) {
	for( reco::MuonCollection::const_iterator muonit = muonHandle_->begin(); muonit != muonHandle_->end(); muonit++ ) {
	  const reco::TrackRef siTrack = muonit->innerTrack();
	  hmuPt    = dbe_->get(DirName+"/"+"METTask_muonPt");     if (hmuPt    && hmuPt->getRootObject())  hmuPt   ->Fill( muonit->p4().pt() );
	  hmuEta   = dbe_->get(DirName+"/"+"METTask_muonEta");    if (hmuEta   && hmuEta->getRootObject())  hmuEta  ->Fill( muonit->p4().eta() );
	  hmuNhits = dbe_->get(DirName+"/"+"METTask_muonNhits");  if (hmuNhits && hmuNhits->getRootObject())  hmuNhits->Fill( siTrack.isNonnull() ? siTrack->numberOfValidHits() : -999 );
	  hmuChi2  = dbe_->get(DirName+"/"+"METTask_muonNormalizedChi2");   if (hmuChi2  && hmuChi2->getRootObject())  hmuChi2 ->Fill( siTrack.isNonnull() ? siTrack->chi2()/siTrack->ndof() : -999 );
	  double d0 = siTrack.isNonnull() ? -1 * siTrack->dxy( beamSpot_) : -999;
	  hmuD0    = dbe_->get(DirName+"/"+"METTask_muonD0");     if (hmuD0    && hmuD0->getRootObject())  hmuD0->Fill( d0 );
	}
	const unsigned int nMuons = muonHandle_->size();
	for( unsigned int mus = 0; mus < nMuons; mus++ ) {
	  reco::MuonRef muref( muonHandle_, mus);
	  reco::MuonMETCorrectionData muCorrData = (*tcMetValueMapHandle_)[muref];
	  hMExCorrection      = dbe_->get(DirName+"/"+"METTask_MExCorrection");       if (hMExCorrection      && hMExCorrection->getRootObject())       hMExCorrection-> Fill(muCorrData.corrY());
	  hMEyCorrection      = dbe_->get(DirName+"/"+"METTask_MEyCorrection");       if (hMEyCorrection      && hMEyCorrection->getRootObject())       hMEyCorrection-> Fill(muCorrData.corrX());
	  hMuonCorrectionFlag = dbe_->get(DirName+"/"+"METTask_CorrectionFlag");  if (hMuonCorrectionFlag && hMuonCorrectionFlag->getRootObject())  hMuonCorrectionFlag-> Fill(muCorrData.type());
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

  if(isCaloMet_){
    edm::Handle<reco::CaloJetCollection> caloJets;
    iEvent.getByToken(caloJetsToken_, caloJets);
    if (!caloJets.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
      if (verbose_) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
    }
    
    for (reco::CaloJetCollection::const_iterator cal = caloJets->begin();
	 cal!=caloJets->end(); ++cal){
      if (cal->pt()>highPtJetThreshold_){
	return_value=true;
      }
    }
  }
  if(isTCMet_){
    edm::Handle<reco::JPTJetCollection> jptJets;
    iEvent.getByToken(jptJetsToken_, jptJets);
    if (!jptJets.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
      if (verbose_) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
    }
    
    for (reco::JPTJetCollection::const_iterator cal = jptJets->begin();
	 cal!=jptJets->end(); ++cal){
      if (cal->pt()>highPtJetThreshold_){
	return_value=true;
      }
    }
  }
  if(isPFMet_){
    edm::Handle<reco::PFJetCollection> PFJets;
    iEvent.getByToken(pfJetsToken_, PFJets);
    if (!PFJets.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
      if (verbose_) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
    }
    for (reco::PFJetCollection::const_iterator cal = PFJets->begin();
	 cal!=PFJets->end(); ++cal){
      if (cal->pt()>highPtJetThreshold_){
	return_value=true;
      }
    }
  }


  return return_value;
}

// // ***********************************************************
bool METAnalyzer::selectLowPtJetEvent(const edm::Event& iEvent){

  bool return_value=false;
  if(isCaloMet_){
    edm::Handle<reco::CaloJetCollection> caloJets;
    iEvent.getByToken(caloJetsToken_, caloJets);
    if (!caloJets.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
      if (verbose_) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
    }
    
    for (reco::CaloJetCollection::const_iterator cal = caloJets->begin();
	 cal!=caloJets->end(); ++cal){
      if (cal->pt()>lowPtJetThreshold_){
	return_value=true;
      }
    }
  }
  if(isTCMet_){
    edm::Handle<reco::JPTJetCollection> jptJets;
    iEvent.getByToken(jptJetsToken_, jptJets);
    if (!jptJets.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
      if (verbose_) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
    }
    
    for (reco::JPTJetCollection::const_iterator cal = jptJets->begin();
	 cal!=jptJets->end(); ++cal){
      if (cal->pt()>lowPtJetThreshold_){
	return_value=true;
      }
    }
  }
  if(isPFMet_){
    edm::Handle<reco::PFJetCollection> PFJets;
    iEvent.getByToken(pfJetsToken_, PFJets);
    if (!PFJets.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
      if (verbose_) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
    }
    for (reco::PFJetCollection::const_iterator cal = PFJets->begin();
	 cal!=PFJets->end(); ++cal){
      if (cal->pt()>lowPtJetThreshold_){
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


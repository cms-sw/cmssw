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

  mOutputFile_   = parameters.getParameter<std::string>("OutputFile");
  MetType_ = parameters.getUntrackedParameter<std::string>("METType");

  triggerResultsLabel_        = parameters.getParameter<edm::InputTag>("TriggerResultsLabel");
  triggerResultsToken_= consumes<edm::TriggerResults>(edm::InputTag(triggerResultsLabel_));
  jetCorrectionService_ = pSet.getParameter<std::string> ("JetCorrections");

  isCaloMet_ = (std::string("calo")==MetType_);
  isTCMet_ = (std::string("tc") ==MetType_);
  isPFMet_ = (std::string("pf") ==MetType_);

  // MET information
  metCollectionLabel_       = parameters.getParameter<edm::InputTag>("METCollectionLabel");

  if(isTCMet_ || isCaloMet_){
    inputJetIDValueMap      = pSet.getParameter<edm::InputTag>("InputJetIDValueMap");
    jetID_ValueMapToken_= consumes< edm::ValueMap<reco::JetID> >(inputJetIDValueMap);
    jetIDFunctorLoose=JetIDSelectionFunctor(JetIDSelectionFunctor::PURE09, JetIDSelectionFunctor::LOOSE);
  }
  if(isPFMet_){
    pfjetIDFunctorLoose=PFJetIDSelectionFunctor(PFJetIDSelectionFunctor::FIRSTDATA, PFJetIDSelectionFunctor::LOOSE);
  }
  ptThreshold_ = parameters.getParameter<double>("ptThreshold");


  if(isPFMet_){
    pfMetToken_= consumes<reco::PFMETCollection>(edm::InputTag(metCollectionLabel_));
  }
 if(isCaloMet_){
    caloMetToken_= consumes<reco::CaloMETCollection>(edm::InputTag(metCollectionLabel_));
  }
 if(isTCMet_){
    tcMetToken_= consumes<reco::METCollection>(edm::InputTag(metCollectionLabel_));
  }
  hTriggerLabelsIsSet_ = false;
  //jet cleanup parameters
  cleaningParameters_ = pSet.getParameter<ParameterSet>("CleaningParameters");


  //Vertex requirements
  bypassAllPVChecks_    = cleaningParameters_.getParameter<bool>("bypassAllPVChecks");
  bypassAllDCSChecks_    = cleaningParameters_.getParameter<bool>("bypassAllDCSChecks");
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
  if (isTCMet_)   jptJetsToken_ = consumes<reco::JPTJetCollection>(jetCollectionLabel_);
  if (isPFMet_)   pfJetsToken_ = consumes<reco::PFJetCollection>(jetCollectionLabel_);

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

  triggerSelectedSubFolders_ = parameters.getParameter<edm::VParameterSet>("triggerSelectedSubFolders");
  for (edm::VParameterSet::const_iterator it = triggerSelectedSubFolders_.begin(); it!= triggerSelectedSubFolders_.end(); it++) {
    triggerFolderEventFlag_.push_back(new GenericTriggerEventFlag( *it, consumesCollector() ));
    triggerFolderExpr_.push_back(it->getParameter<std::vector<std::string> >("hltPaths"));
    triggerFolderLabels_.push_back(it->getParameter<std::string>("label"));
  }

//  edm::ParameterSet highptjetparms = parameters.getParameter<edm::ParameterSet>("highPtJetTrigger");
//  edm::ParameterSet lowptjetparms  = parameters.getParameter<edm::ParameterSet>("lowPtJetTrigger" );
//  edm::ParameterSet minbiasparms   = parameters.getParameter<edm::ParameterSet>("minBiasTrigger"  );
//  edm::ParameterSet highmetparms   = parameters.getParameter<edm::ParameterSet>("highMETTrigger"  );
//  //  edm::ParameterSet lowmetparms    = parameters.getParameter<edm::ParameterSet>("lowMETTrigger"   );
//  edm::ParameterSet eleparms       = parameters.getParameter<edm::ParameterSet>("eleTrigger"      );
//  edm::ParameterSet muonparms      = parameters.getParameter<edm::ParameterSet>("muonTrigger"     );
//
//  highPtJetEventFlag_ = new GenericTriggerEventFlag( highptjetparms, consumesCollector() );
//  highPtJetExpr_ = highptjetparms.getParameter<std::vector<std::string> >("hltPaths");
//
//  lowPtJetEventFlag_  = new GenericTriggerEventFlag( lowptjetparms, consumesCollector() );
//  lowPtJetExpr_  = lowptjetparms .getParameter<std::vector<std::string> >("hltPaths");
//
//  minBiasEventFlag_   = new GenericTriggerEventFlag( minbiasparms , consumesCollector() );
//  minbiasExpr_   = minbiasparms  .getParameter<std::vector<std::string> >("hltPaths");
//
//  highMETEventFlag_   = new GenericTriggerEventFlag( highmetparms , consumesCollector() );
//  highMETExpr_   = highmetparms  .getParameter<std::vector<std::string> >("hltPaths");
//
//  eleEventFlag_       = new GenericTriggerEventFlag( eleparms     , consumesCollector() );
//  elecExpr_      = eleparms      .getParameter<std::vector<std::string> >("hltPaths");
//
//  muonEventFlag_      = new GenericTriggerEventFlag( muonparms    , consumesCollector() );
//  muonExpr_      = muonparms     .getParameter<std::vector<std::string> >("hltPaths");

}

// ***********************************************************
METAnalyzer::~METAnalyzer() {

  for (std::vector<GenericTriggerEventFlag *>::const_iterator it = triggerFolderEventFlag_.begin(); it!= triggerFolderEventFlag_.end(); it++) {
    delete *it;
  }

//  delete highPtJetEventFlag_;
//  delete lowPtJetEventFlag_;
//  delete minBiasEventFlag_;
//  delete highMETEventFlag_;
//  delete eleEventFlag_;
//  delete muonEventFlag_;
}

void METAnalyzer::beginJob(){

  // trigger information
//  HLTPathsJetMBByName_ = parameters.getParameter<std::vector<std::string > >("HLTPathsJetMB");

  cleaningParameters_ = parameters.getParameter<ParameterSet>("CleaningParameters"),

  // ==========================================================
  //DCS information
  // ==========================================================
  DCSFilter_ = new JetMETDQMDCSFilter(parameters.getParameter<ParameterSet>("DCSFilter"));

  // misc
  verbose_      = parameters.getParameter<int>("verbose");
//  etThreshold_  = parameters.getParameter<double>("etThreshold"); // MET threshold

  FolderName_              = parameters.getUntrackedParameter<std::string>("FolderName");

//  highPtJetThreshold_ = parameters.getParameter<double>("HighPtJetThreshold"); // High Pt Jet threshold
//  lowPtJetThreshold_  = parameters.getParameter<double>("LowPtJetThreshold");  // Low Pt Jet threshold
//  highMETThreshold_   = parameters.getParameter<double>("HighMETThreshold");   // High MET threshold

  // DQStore stuff
  dbe_ = edm::Service<DQMStore>().operator->();
  LogTrace(metname)<<"[METAnalyzer] Parameters initialization";
  std::string DirName = "JetMET/MET/"+metCollectionLabel_.label();
  dbe_->setCurrentFolder(DirName);

  folderNames_.push_back("Uncleaned");
  folderNames_.push_back("Cleaned");
  folderNames_.push_back("DiJet");

  for (std::vector<std::string>::const_iterator ic = folderNames_.begin();
       ic != folderNames_.end(); ic++){
    bookMESet(DirName+"/"+*ic);
  }
}

// ***********************************************************
void METAnalyzer::endJob() {
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
  if (DirName.find("Uncleaned")!=std::string::npos) bLumiSecPlot=true;
  bookMonitorElement(DirName,bLumiSecPlot);

  if (DirName.find("Cleaned")!=std::string::npos) {
    for (unsigned i = 0; i<triggerFolderEventFlag_.size(); i++) {
      if (triggerFolderEventFlag_[i]->on()) {
        bookMonitorElement(DirName+"/"+triggerFolderLabels_[i],false);
  //      triggerFolderME_.push_back(dbe_->bookString("triggerFolder_"+triggerFolderLabels_[i], triggerFolderExpr_[i][0]));
      }
    }
  }
//  if ( highPtJetEventFlag_->on() ) {
//    bookMonitorElement(DirName+"/"+"HighPtJet",false);
//    hTriggerName_HighPtJet = dbe_->bookString("triggerName_HighPtJet", highPtJetExpr_[0]);
//  }
//
//  if ( lowPtJetEventFlag_->on() ) {
//    bookMonitorElement(DirName+"/"+"LowPtJet",false);
//    hTriggerName_LowPtJet = dbe_->bookString("triggerName_LowPtJet", lowPtJetExpr_[0]);
//  }
//
//  if ( minBiasEventFlag_->on() ) {
//    bookMonitorElement(DirName+"/"+"MinBias",false);
//    hTriggerName_MinBias = dbe_->bookString("triggerName_MinBias", minbiasExpr_[0]);
//    if (verbose_) std::cout << "minBiasEventFlag_ is on, folder created\n";
//  }
//
//  if ( highMETEventFlag_->on() ) {
//    bookMonitorElement(DirName+"/"+"HighMET",false);
//    hTriggerName_HighMET = dbe_->bookString("triggerName_HighMET", highMETExpr_[0]);
//  }
//
//  if ( eleEventFlag_->on() ) {
//    bookMonitorElement(DirName+"/"+"Ele",false);
//    hTriggerName_Ele = dbe_->bookString("triggerName_Ele", elecExpr_[0]);
//    if (verbose_) std::cout << "eleEventFlag is on, folder created\n";
//  }
//
//  if ( muonEventFlag_->on() ) {
//    bookMonitorElement(DirName+"/"+"Muon",false);
//    hTriggerName_Muon = dbe_->bookString("triggerName_Muon", muonExpr_[0]);
//    if (verbose_) std::cout << "muonEventFlag is on, folder created\n";
//  }
}

// ***********************************************************
void METAnalyzer::bookMonitorElement(std::string DirName, bool bLumiSecPlot=false)
{
  if (verbose_) std::cout << "bookMonitorElement " << DirName << std::endl;

  dbe_->setCurrentFolder(DirName);

  hTrigger    = dbe_->book1D("triggerResults", "triggerResults", 500, 0, 500); 
//  hTrigger    = dbe_->book1D("triggerResults", "triggerResults", allTriggerNames_.size(), 0, allTriggerNames_.size()); 
//  for (unsigned i = 0; i< allTriggerNames_.size(); i++) {
//    hTrigger->setBinLabel(i, allTriggerNames_[i]);
//    std::cout<<"Setting label "<<i<<" "<<allTriggerNames_[i]<<std::endl;
//  }
  hMEx        = dbe_->book1D("MEx",        "MEx",        200, -500,  500);
  hMEy        = dbe_->book1D("MEy",        "MEy",        200, -500,  500);
  hMET        = dbe_->book1D("MET",        "MET",        200,    0, 1000);
  hSumET      = dbe_->book1D("SumET",      "SumET",      400,    0, 4000);
  hMETSig     = dbe_->book1D("METSig",     "METSig",      51,    0,   51);
  hMETPhi     = dbe_->book1D("METPhi",     "METPhi",      60, -3.2,  3.2);
  hMET_logx   = dbe_->book1D("MET_logx",   "MET_logx",    40,   -1,    7);
  hSumET_logx = dbe_->book1D("SumET_logx", "SumET_logx",  40,   -1,    7);

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
  meMEx_profile   = dbe_->bookProfile("MEx_profile",   "met.px()",    nbinsPV_, nPVMin_, nPVMax_, 200, -500,  500);
  meMEy_profile   = dbe_->bookProfile("MEy_profile",   "met.py()",    nbinsPV_, nPVMin_, nPVMax_, 200, -500,  500);
  meMET_profile   = dbe_->bookProfile("MET_profile",   "met.pt()",    nbinsPV_, nPVMin_, nPVMax_, 200,    0, 1000);
  meSumET_profile = dbe_->bookProfile("SumET_profile", "met.sumEt()", nbinsPV_, nPVMin_, nPVMax_, 400,    0, 4000);
  // Set NPV profiles x-axis title
  //----------------------------------------------------------------------------
  meMEx_profile  ->setAxisTitle("nvtx", 1);
  meMEy_profile  ->setAxisTitle("nvtx", 1);
  meMET_profile  ->setAxisTitle("nvtx", 1);
  meSumET_profile->setAxisTitle("nvtx", 1);
    
  if(isCaloMet_){
    hCaloMaxEtInEmTowers    = dbe_->book1D("CaloMaxEtInEmTowers",   "CaloMaxEtInEmTowers"   ,100,0,2000);
    hCaloMaxEtInEmTowers->setAxisTitle("Et(Max) in EM Tower [GeV]",1);
    hCaloMaxEtInHadTowers   = dbe_->book1D("CaloMaxEtInHadTowers",  "CaloMaxEtInHadTowers"  ,100,0,2000);
    hCaloMaxEtInHadTowers->setAxisTitle("Et(Max) in Had Tower [GeV]",1);

    hCaloHadEtInHB          = dbe_->book1D("CaloHadEtInHB","CaloHadEtInHB",100,0,2000);
    hCaloHadEtInHB->setAxisTitle("Had Et [GeV]",1);
    hCaloHadEtInHO          = dbe_->book1D("CaloHadEtInHO","CaloHadEtInHO",25,0,500);
    hCaloHadEtInHO->setAxisTitle("Had Et [GeV]",1);
    hCaloHadEtInHE          = dbe_->book1D("CaloHadEtInHE","CaloHadEtInHE",100,0,2000);
    hCaloHadEtInHE->setAxisTitle("Had Et [GeV]",1);
    hCaloHadEtInHF          = dbe_->book1D("CaloHadEtInHF","CaloHadEtInHF",50,0,1000);
    hCaloHadEtInHF->setAxisTitle("Had Et [GeV]",1);
    hCaloEmEtInHF           = dbe_->book1D("CaloEmEtInHF" ,"CaloEmEtInHF" ,25,0,500);
    hCaloEmEtInHF->setAxisTitle("EM Et [GeV]",1);
    hCaloEmEtInEE           = dbe_->book1D("CaloEmEtInEE" ,"CaloEmEtInEE" ,50,0,1000);
    hCaloEmEtInEE->setAxisTitle("EM Et [GeV]",1);
    hCaloEmEtInEB           = dbe_->book1D("CaloEmEtInEB" ,"CaloEmEtInEB" ,100,0,2000);
    hCaloEmEtInEB->setAxisTitle("EM Et [GeV]",1);

    hCaloMETPhi020  = dbe_->book1D("CaloMETPhi020",  "CaloMETPhi020",   60, -3.2,  3.2);
    hCaloMETPhi020 ->setAxisTitle("METPhi [rad] (MET>20 GeV)", 1);

    //hCaloMaxEtInEmTowers    = dbe_->book1D("CaloMaxEtInEmTowers",   "CaloMaxEtInEmTowers"   ,100,0,2000);
    //hCaloMaxEtInEmTowers->setAxisTitle("Et(Max) in EM Tower [GeV]",1);
    //hCaloMaxEtInHadTowers   = dbe_->book1D("CaloMaxEtInHadTowers",  "CaloMaxEtInHadTowers"  ,100,0,2000);
    //hCaloMaxEtInHadTowers->setAxisTitle("Et(Max) in Had Tower [GeV]",1);
    hCaloEtFractionHadronic = dbe_->book1D("CaloEtFractionHadronic","CaloEtFractionHadronic",100,0,1);
    hCaloEtFractionHadronic->setAxisTitle("Hadronic Et Fraction",1);
    hCaloEmEtFraction       = dbe_->book1D("CaloEmEtFraction",      "CaloEmEtFraction"      ,100,0,1);
    hCaloEmEtFraction->setAxisTitle("EM Et Fraction",1);
    
    //hCaloEmEtFraction002    = dbe_->book1D("CaloEmEtFraction002",   "CaloEmEtFraction002"      ,100,0,1);
    //hCaloEmEtFraction002->setAxisTitle("EM Et Fraction (MET>2 GeV)",1);
    //hCaloEmEtFraction010    = dbe_->book1D("CaloEmEtFraction010",   "CaloEmEtFraction010"      ,100,0,1);
    //hCaloEmEtFraction010->setAxisTitle("EM Et Fraction (MET>10 GeV)",1);
    hCaloEmEtFraction020    = dbe_->book1D("CaloEmEtFraction020",   "CaloEmEtFraction020"      ,100,0,1);
    hCaloEmEtFraction020->setAxisTitle("EM Et Fraction (MET>20 GeV)",1);

    if (metCollectionLabel_.label() == "corMetGlobalMuons" ) {
      hCalomuPt    = dbe_->book1D("CalomuonPt", "CalomuonPt", 50, 0, 500);
      hCalomuEta   = dbe_->book1D("CalomuonEta", "CalomuonEta", 60, -3.0, 3.0);
      hCalomuNhits = dbe_->book1D("CalomuonNhits", "CalomuonNhits", 50, 0, 50);
      hCalomuChi2  = dbe_->book1D("CalomuonNormalizedChi2", "CalomuonNormalizedChi2", 20, 0, 20);
      hCalomuD0    = dbe_->book1D("CalomuonD0", "CalomuonD0", 50, -1, 1);
      hCaloMExCorrection       = dbe_->book1D("CaloMExCorrection", "CaloMExCorrection", 100, -500.0,500.0);
      hCaloMEyCorrection       = dbe_->book1D("CaloMEyCorrection", "CaloMEyCorrection", 100, -500.0,500.0);
      hCaloMuonCorrectionFlag  = dbe_->book1D("CaloCorrectionFlag","CaloCorrectionFlag", 5, -0.5, 4.5);
    }

  }

  if(isPFMet_){
    mePhotonEtFraction        = dbe_->book1D("PfPhotonEtFraction",        "pfmet.photonEtFraction()",         50, 0,    1);
    mePhotonEt                = dbe_->book1D("PfPhotonEt",                "pfmet.photonEt()",                100, 0, 1000);
    meNeutralHadronEtFraction = dbe_->book1D("PfNeutralHadronEtFraction", "pfmet.neutralHadronEtFraction()",  50, 0,    1);
    meNeutralHadronEt         = dbe_->book1D("PfNeutralHadronEt",         "pfmet.neutralHadronEt()",         100, 0, 1000);
    meElectronEtFraction      = dbe_->book1D("PfElectronEtFraction",      "pfmet.electronEtFraction()",       50, 0,    1);
    meElectronEt              = dbe_->book1D("PfElectronEt",              "pfmet.electronEt()",              100, 0, 1000);
    meChargedHadronEtFraction = dbe_->book1D("PfChargedHadronEtFraction", "pfmet.chargedHadronEtFraction()",  50, 0,    1);
    meChargedHadronEt         = dbe_->book1D("PfChargedHadronEt",         "pfmet.chargedHadronEt()",         100, 0, 1000);
    meMuonEtFraction          = dbe_->book1D("PfMuonEtFraction",          "pfmet.muonEtFraction()",           50, 0,    1);
    meMuonEt                  = dbe_->book1D("PfMuonEt",                  "pfmet.muonEt()",                  100, 0, 1000);
    meHFHadronEtFraction      = dbe_->book1D("PfHFHadronEtFraction",      "pfmet.HFHadronEtFraction()",       50, 0,    1);
    meHFHadronEt              = dbe_->book1D("PfHFHadronEt",              "pfmet.HFHadronEt()",              100, 0, 1000);
    meHFEMEtFraction          = dbe_->book1D("PfHFEMEtFraction",          "pfmet.HFEMEtFraction()",           50, 0,    1);
    meHFEMEt                  = dbe_->book1D("PfHFEMEt",                  "pfmet.HFEMEt()",                  100, 0, 1000);
    
    mePhotonEtFraction_profile        = dbe_->bookProfile("PfPhotonEtFraction_profile",        "pfmet.photonEtFraction()",        nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    mePhotonEt_profile                = dbe_->bookProfile("PfPhotonEt_profile",                "pfmet.photonEt()",                nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    meNeutralHadronEtFraction_profile = dbe_->bookProfile("PfNeutralHadronEtFraction_profile", "pfmet.neutralHadronEtFraction()", nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meNeutralHadronEt_profile         = dbe_->bookProfile("PfNeutralHadronEt_profile",         "pfmet.neutralHadronEt()",         nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    meElectronEtFraction_profile      = dbe_->bookProfile("PfElectronEtFraction_profile",      "pfmet.electronEtFraction()",      nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meElectronEt_profile              = dbe_->bookProfile("PfElectronEt_profile",              "pfmet.electronEt()",              nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    meChargedHadronEtFraction_profile = dbe_->bookProfile("PfChargedHadronEtFraction_profile", "pfmet.chargedHadronEtFraction()", nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meChargedHadronEt_profile         = dbe_->bookProfile("PfChargedHadronEt_profile",         "pfmet.chargedHadronEt()",         nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    meMuonEtFraction_profile          = dbe_->bookProfile("PfMuonEtFraction_profile",          "pfmet.muonEtFraction()",          nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meMuonEt_profile                  = dbe_->bookProfile("PfMuonEt_profile",                  "pfmet.muonEt()",                  nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    meHFHadronEtFraction_profile      = dbe_->bookProfile("PfHFHadronEtFraction_profile",      "pfmet.HFHadronEtFraction()",      nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meHFHadronEt_profile              = dbe_->bookProfile("PfHFHadronEt_profile",              "pfmet.HFHadronEt()",              nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    meHFEMEtFraction_profile          = dbe_->bookProfile("PfHFEMEtFraction_profile",          "pfmet.HFEMEtFraction()",          nbinsPV_, nPVMin_, nPVMax_,  50, 0,    1);
    meHFEMEt_profile                  = dbe_->bookProfile("PfHFEMEt_profile",                  "pfmet.HFEMEt()",                  nbinsPV_, nPVMin_, nPVMax_, 100, 0, 1000);
    
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
      hMExLS = dbe_->book2D("MEx_LS","MEx_LS",200,-200,200,50,0.,500.);
      hMExLS->setAxisTitle("MEx [GeV]",1);
      hMExLS->setAxisTitle("Lumi Section",2);
      hMEyLS = dbe_->book2D("MEy_LS","MEy_LS",200,-200,200,50,0.,500.);
      hMEyLS->setAxisTitle("MEy [GeV]",1);
      hMEyLS->setAxisTitle("Lumi Section",2);
    }
  }

  if (isTCMet_) {
    htrkPt    = dbe_->book1D("trackPt", "trackPt", 50, 0, 500);
    htrkEta   = dbe_->book1D("trackEta", "trackEta", 60, -3.0, 3.0);
    htrkNhits = dbe_->book1D("trackNhits", "trackNhits", 50, 0, 50);
    htrkChi2  = dbe_->book1D("trackNormalizedChi2", "trackNormalizedChi2", 20, 0, 20);
    htrkD0    = dbe_->book1D("trackD0", "trackd0", 50, -1, 1);
    helePt    = dbe_->book1D("electronPt", "electronPt", 50, 0, 500);
    heleEta   = dbe_->book1D("electronEta", "electronEta", 60, -3.0, 3.0);
    heleHoE   = dbe_->book1D("electronHoverE", "electronHoverE", 25, 0, 0.5);
    hmuPt     = dbe_->book1D("muonPt", "muonPt", 50, 0, 500);
    hmuEta    = dbe_->book1D("muonEta", "muonEta", 60, -3.0, 3.0);
    hmuNhits  = dbe_->book1D("muonNhits", "muonNhits", 50, 0, 50);
    hmuChi2   = dbe_->book1D("muonNormalizedChi2", "muonNormalizedChi2", 20, 0, 20);
    hmuD0     = dbe_->book1D("muonD0", "muonD0", 50, -1, 1);

    hMETIonFeedbck      = dbe_->book1D("METIonFeedbck", "METIonFeedbck" ,200,0,1000);
    hMETHPDNoise        = dbe_->book1D("METHPDNoise",   "METHPDNoise"   ,200,0,1000);
    hMETRBXNoise        = dbe_->book1D("METRBXNoise",   "METRBXNoise"   ,200,0,1000);
    hMExCorrection       = dbe_->book1D("MExCorrection", "MExCorrection", 100, -500.0,500.0);
    hMEyCorrection       = dbe_->book1D("MEyCorrection", "MEyCorrection", 100, -500.0,500.0);
    hMuonCorrectionFlag  = dbe_->book1D("CorrectionFlag","CorrectionFlag", 5, -0.5, 4.5);
  }
}

// ***********************************************************
void METAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
//  std::cout  << "Run " << iRun.run() << " hltconfig.init " 
//             << hltConfig_.init(iRun,iSetup,triggerResultsLabel_.process(),changed_) << " length: "<<hltConfig_.triggerNames().size()<<" changed "<<changed_<<std::endl; 
  bool changed(true);
  if (hltConfig_.init(iRun,iSetup,triggerResultsLabel_.process(),changed)) {
    if (changed) {
      hltConfig_.dump("ProcessName");
      hltConfig_.dump("GlobalTag");
      hltConfig_.dump("TableName");
//      hltConfig_.dump("Streams");
//      hltConfig_.dump("Datasets");
//      hltConfig_.dump("PrescaleTable");
//      hltConfig_.dump("ProcessPSet");
    }
  } else {
    std::cout << "HLTEventAnalyzerAOD::analyze:"
              << " config extraction failure with process name "
              << triggerResultsLabel_.process() << std::endl;
  }

  allTriggerNames_.clear();
  for (unsigned i = 0; i<hltConfig_.size();i++) {
    allTriggerNames_.push_back(hltConfig_.triggerName(i));
  }
//  std::cout<<"Length: "<<allTriggerNames_.size()<<std::endl;

  triggerSelectedSubFolders_ = parameters.getParameter<edm::VParameterSet>("triggerSelectedSubFolders");
  for ( std::vector<GenericTriggerEventFlag *>::const_iterator it = triggerFolderEventFlag_.begin(); it!= triggerFolderEventFlag_.end(); it++) {
    int pos = it - triggerFolderEventFlag_.begin();
    if ((*it)->on()) {
      (*it)->initRun( iRun, iSetup );
      if (triggerSelectedSubFolders_[pos].exists(std::string("hltDBKey"))) {
//        std::cout<<"Looking for hltDBKey for"<<triggerFolderLabels_[pos]<<std::endl;
        if ((*it)->expressionsFromDB((*it)->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
          triggerFolderExpr_[pos] = (*it)->expressionsFromDB((*it)->hltDBKey(), iSetup);
      }
//      for (unsigned j = 0; j<triggerFolderExpr_[pos].size(); j++) std::cout<<"pos "<<pos<<" "<<triggerFolderLabels_[pos]<<" triggerFolderExpr_"<<triggerFolderExpr_[pos][j]<<std::endl;
    }
  }
//  if ( highPtJetEventFlag_->on() ) highPtJetEventFlag_->initRun( iRun, iSetup );
//  if ( lowPtJetEventFlag_ ->on() ) lowPtJetEventFlag_ ->initRun( iRun, iSetup );
//  if ( minBiasEventFlag_  ->on() ) minBiasEventFlag_  ->initRun( iRun, iSetup );
//  if ( highMETEventFlag_ ->on() ) highMETEventFlag_  ->initRun( iRun, iSetup );
//  //  if ( _LowMETEventFlag   ->on() ) _LowMETEventFlag   ->initRun( iRun, iSetup );
//  if ( eleEventFlag_      ->on() ) eleEventFlag_      ->initRun( iRun, iSetup );
//  if ( muonEventFlag_     ->on() ) muonEventFlag_     ->initRun( iRun, iSetup );
//
//  if (highPtJetEventFlag_->on() && highPtJetEventFlag_->expressionsFromDB(highPtJetEventFlag_->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
//    highPtJetExpr_ = highPtJetEventFlag_->expressionsFromDB(highPtJetEventFlag_->hltDBKey(), iSetup);
//  if (lowPtJetEventFlag_->on() && lowPtJetEventFlag_->expressionsFromDB(lowPtJetEventFlag_->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
//    lowPtJetExpr_  = lowPtJetEventFlag_->expressionsFromDB(lowPtJetEventFlag_->hltDBKey(),   iSetup);
//  if (highMETEventFlag_->on() && highMETEventFlag_->expressionsFromDB(highMETEventFlag_->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
//    highMETExpr_   = highMETEventFlag_->expressionsFromDB(highMETEventFlag_->hltDBKey(),     iSetup);
//  //  if (_LowMETEventFlag->on() && _LowMETEventFlag->expressionsFromDB(_LowMETEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
//  //    lowMETExpr_    = _LowMETEventFlag->expressionsFromDB(_LowMETEventFlag->hltDBKey(),       iSetup);
//  if (muonEventFlag_->on() && muonEventFlag_->expressionsFromDB(muonEventFlag_->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
//    muonExpr_      = muonEventFlag_->expressionsFromDB(muonEventFlag_->hltDBKey(),           iSetup);
//  if (eleEventFlag_->on() && eleEventFlag_->expressionsFromDB(eleEventFlag_->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
//    elecExpr_      = eleEventFlag_->expressionsFromDB(eleEventFlag_->hltDBKey(),             iSetup);
//  if (minBiasEventFlag_->on() && minBiasEventFlag_->expressionsFromDB(minBiasEventFlag_->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
//    minbiasExpr_   = minBiasEventFlag_->expressionsFromDB(minBiasEventFlag_->hltDBKey(),     iSetup);

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

  //below is the original METAnalyzer formulation
  for (std::vector<std::string>::const_iterator ic = folderNames_.begin(); ic != folderNames_.end(); ic++) {
    std::string DirName;
    DirName = dirName+*ic;

    makeRatePlot(DirName,totltime);
    for ( std::vector<GenericTriggerEventFlag *>::const_iterator it = triggerFolderEventFlag_.begin(); it!= triggerFolderEventFlag_.end(); it++) {
      int pos = it - triggerFolderEventFlag_.begin();
      if ((*it)->on()) {
        makeRatePlot(DirName+"/"+triggerFolderLabels_[pos],totltime);
      }
    }
//      if ( highPtJetEventFlag_->on() )
//	makeRatePlot(DirName+"/"+"triggerName_HighJetPt",totltime);
//      if ( lowPtJetEventFlag_->on() )
//	makeRatePlot(DirName+"/"+"triggerName_LowJetPt",totltime);
//      if ( minBiasEventFlag_->on() )
//	makeRatePlot(DirName+"/"+"triggerName_MinBias",totltime);
//      if ( highMETEventFlag_->on() )
//	makeRatePlot(DirName+"/"+"triggerName_HighMET",totltime);
//      //      if ( _LowMETEventFlag->on() )
//      //	makeRatePlot(DirName+"/"+"triggerName_LowMET",totltime);
//      if ( eleEventFlag_->on() )
//	makeRatePlot(DirName+"/"+"triggerName_Ele",totltime);
//      if ( muonEventFlag_->on() )
//	makeRatePlot(DirName+"/"+"triggerName_Muon",totltime);
  }
}


// ***********************************************************
void METAnalyzer::makeRatePlot(std::string DirName, double totltime)
{

  dbe_->setCurrentFolder(DirName);
  MonitorElement *meMET = dbe_->get(DirName+"/"+"MET");

  TH1F* tMET;
  TH1F* tMETRate;

  if ( meMET )
    if ( meMET->getRootObject() ) {
      tMET     = meMET->getTH1F();

      // Integral plot & convert number of events to rate (hz)
      tMETRate = (TH1F*) tMET->Clone("METRate");
      for (int i = tMETRate->GetNbinsX()-1; i>=0; i--){
	tMETRate->SetBinContent(i+1,tMETRate->GetBinContent(i+2)+tMET->GetBinContent(i+1));
      }
      for (int i = 0; i<tMETRate->GetNbinsX(); i++){
	tMETRate->SetBinContent(i+1,tMETRate->GetBinContent(i+1)/double(totltime));
      }

      tMETRate->SetName("METRate");
      tMETRate->SetTitle("METRate");
      hMETRate      = dbe_->book1D("METRate",tMETRate);
    }
}

// ***********************************************************
void METAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  if (verbose_) std::cout << "METAnalyzer analyze" << std::endl;

  std::string DirName = FolderName_+metCollectionLabel_.label();


  // ==========================================================
  // Trigger information
  //
//  trigJetMB_=0;
//  trigHighPtJet_=0;
//  trigLowPtJet_=0;
//  trigMinBias_=0;
//  trigHighMET_=0;
//  //  _trig_LowMET=0;
//  trigEle_=0;
//  trigMuon_=0;
//  trigPhysDec_=0;
  std::vector<int> triggerFolderDecisions;
  triggerFolderDecisions_ = std::vector<int> (triggerFolderEventFlag_.size(), 0);
  // **** Get the TriggerResults container
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(triggerResultsToken_, triggerResults);

  if( triggerResults.isValid()) {
    /////////// Analyzing HLT Trigger Results (TriggerResults) //////////
    // Check how many HLT triggers are in triggerResults
    int ntrigs = (*triggerResults).size();
    if (verbose_) std::cout << "ntrigs=" << ntrigs << std::endl;
    // If index=ntrigs, this HLT trigger doesn't exist in the HLT table for this data.
    for (std::vector<GenericTriggerEventFlag *>::const_iterator it =  triggerFolderEventFlag_.begin(); it!=triggerFolderEventFlag_.end();it++) {
      unsigned pos = it - triggerFolderEventFlag_.begin();
      bool fd = (*it)->accept(iEvent, iSetup);
      triggerFolderDecisions_[pos] = fd;
    }
    allTriggerDecisions_.clear();
    for (unsigned i=0;i<allTriggerNames_.size();++i)  {
      allTriggerDecisions_.push_back((*triggerResults).accept(i)); 
//      std::cout<<"TR "<<(*triggerResults).size()<<" "<<(*triggerResults).accept(i)<<" "<<allTriggerNames_[i]<<std::endl;
    }
//    const unsigned int nTrig(triggerNames.size());
//    for (unsigned i=0;i<nTrig;++i)  {
//      for ( unsigned j = 0; j<triggerFolderExpr_.size(); j++) {
//        for ( unsigned k = 0; k<triggerFolderExpr_[j].size(); k++) {
////          std::cout<<"At trigger: "<<i<<" "<<triggerNames.triggerName(i)<<" testing for " <<triggerFolderExpr_[j][k]<<" fired? "<<(*triggerResults).accept(i)<<" decision now "<<triggerFolderDecisions_[j]<<std::endl;
//          if (std::strstr(triggerNames.triggerName(i).c_str(), (triggerFolderExpr_[j][k]+"_v").c_str())) {
//            if ((*triggerResults).accept(i)) triggerFolderDecisions_[j] = true; 
////            std::cout<<"At trigger: "<<i<<" "<<triggerNames.triggerName(i)<<" testing for " <<triggerFolderExpr_[j][k]<<" fired? "<<(*triggerResults).accept(i)<<" decision now "<<triggerFolderDecisions_[j]<<std::endl;
//          }
//        }
//      }
      //FIXME store decision here
//        if (triggerNames.triggerName(i).find(highPtJetExpr_[0].substr(0,highPtJetExpr_[0].rfind("_v")+2))!=std::string::npos && (*triggerResults).accept(i))#FIXME
//	  trigHighPtJet_=true;
//        else if (triggerNames.triggerName(i).find(lowPtJetExpr_[0].substr(0,lowPtJetExpr_[0].rfind("_v")+2))!=std::string::npos && (*triggerResults).accept(i))
//	  trigLowPtJet_=true;
//        else if (triggerNames.triggerName(i).find(highMETExpr_[0].substr(0,highMETExpr_[0].rfind("_v")+2))!=std::string::npos && (*triggerResults).accept(i))
//	  trigHighMET_=true;
//	//        else if (triggerNames.triggerName(i).find(lowMETExpr_[0].substr(0,lowMETExpr_[0].rfind("_v")+2))!=std::string::npos && (*triggerResults).accept(i))
//	//	  _trig_LowMET=true;
//        else if (triggerNames.triggerName(i).find(muonExpr_[0].substr(0,muonExpr_[0].rfind("_v")+2))!=std::string::npos && (*triggerResults).accept(i))
//	  trigMuon_=true;
//        else if (triggerNames.triggerName(i).find(elecExpr_[0].substr(0,elecExpr_[0].rfind("_v")+2))!=std::string::npos && (*triggerResults).accept(i))
//	  trigEle_=true;
//        else if (triggerNames.triggerName(i).find(minbiasExpr_[0].substr(0,minbiasExpr_[0].rfind("_v")+2))!=std::string::npos && (*triggerResults).accept(i))
//	  trigMinBias_=true;
    
//    for (std::vector<GenericTriggerEventFlag *>::const_iterator it =  triggerFolderEventFlag_.begin(); it!=triggerFolderEventFlag_.end();it++) {
//      unsigned pos = it - triggerFolderEventFlag_.begin();
//      bool fd = (*it)->accept(iEvent, iSetup);
//      bool md = triggerFolderDecisions_[pos];
//      std::cout <<triggerFolderLabels_[pos]<<" FlagDecision "<<(*it)->accept(iEvent, iSetup)<<" myDecision: "<<triggerFolderDecisions_[pos]<<std::endl;
//      if (fd!=md) std::cout<<"Warning! Inconsistent!"<<std::endl;
//    }
//    // count number of requested Jet or MB HLT paths which have fired
//    for (unsigned int i=0; i!=HLTPathsJetMBByName_.size(); i++) {
//      unsigned int triggerIndex = triggerNames.triggerIndex(HLTPathsJetMBByName_[i]);
//      if (triggerIndex<(*triggerResults).size()) {
//        if ((*triggerResults).accept(triggerIndex)) {
//          trigJetMB_++;
//        }
//      }
//    }
//    // for empty input vectors (n==0), take all HLT triggers!
//    if (HLTPathsJetMBByName_.size()==0) trigJetMB_=(*triggerResults).size()-1;
//
//    if (triggerNames.triggerIndex(hltPhysDec_)   != triggerNames.size() &&
//      (*triggerResults).accept(triggerNames.triggerIndex(hltPhysDec_)))   trigPhysDec_=1;
//      } else {
//
//    edm::LogInfo("MetAnalyzer") << "TriggerResults::HLT not found, "
//      "automatically select events";
//
//    // TriggerResults object not found. Look at all events.
//    trigJetMB_=1;
  }

  // ==========================================================
  // MET information

  // **** Get the MET container
  edm::Handle<reco::METCollection> tcmetcoll;
  edm::Handle<reco::CaloMETCollection> calometcoll;
  edm::Handle<reco::PFMETCollection> pfmetcoll;

  if(isTCMet_){
    iEvent.getByToken(tcMetToken_, tcmetcoll);
    if(!tcmetcoll.isValid()) return;
  }
  if(isCaloMet_){
    iEvent.getByToken(caloMetToken_, calometcoll);
    if(!calometcoll.isValid()) return;
  }
  if(isPFMet_){
    iEvent.getByToken(pfMetToken_, pfmetcoll);
    if(!pfmetcoll.isValid()) return;
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

  // ==========================================================
  bool bJetID = false;
  bool bDiJetID = false;
  // Jet ID -------------------------------------------------------
  //

  edm::Handle<CaloJetCollection> caloJets;
  edm::Handle<JPTJetCollection> jptJets;
  edm::Handle<PFJetCollection> pfJets;

  int collsize=-1;

  if (isCaloMet_){
    iEvent.getByToken(caloJetsToken_, caloJets);
    if (!caloJets.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find calojet product" << std::endl;
      if (verbose_) std::cout << "METAnalyzer: Could not find calojet product" << std::endl;
    }
    collsize=caloJets->size();
  }
  if (isTCMet_){
    iEvent.getByToken(jptJetsToken_, jptJets);
    if (!caloJets.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find jptjet product" << std::endl;
      if (verbose_) std::cout << "METAnalyzer: Could not find jptjet product" << std::endl;
    }
    collsize=jptJets->size();
  }

  edm::Handle< edm::ValueMap<reco::JetID> >jetID_ValueMap_Handle;
  if(isTCMet_ || isCaloMet_){
    iEvent.getByToken(jetID_ValueMapToken_,jetID_ValueMap_Handle);
  }

  if (isPFMet_){ iEvent.getByToken(pfJetsToken_, pfJets);
    if (!pfJets.isValid()) {
      LogDebug("") << "METAnalyzer: Could not find pfjet product" << std::endl;
      if (verbose_) std::cout << "METAnalyzer: Could not find pfjet product" << std::endl;
    }
    collsize=pfJets->size();
  }

  unsigned int ind1=-1;
  double pt1=-1;
  bool pass_jetID1=false;
  unsigned int ind2=-1;
  double pt2=-1;
  bool pass_jetID2=false;

  //do loose jet ID-> check threshold on corrected jets
  for (int ijet=0; ijet<collsize; ijet++) {
    double pt_jet=-10;
    double scale=1.;
    bool iscleaned=false;
    if (!jetCorrectionService_.empty()) {
      const JetCorrector* corrector = JetCorrector::getJetCorrector(jetCorrectionService_, iSetup);	 
      if(isCaloMet_){
	scale = corrector->correction((*caloJets)[ijet], iEvent, iSetup);
      }
      if(isTCMet_){
	scale = corrector->correction((*jptJets)[ijet], iEvent, iSetup);
      }
      if(isPFMet_){
	scale = corrector->correction((*pfJets)[ijet], iEvent, iSetup);
      }
    }
    if(isCaloMet_){
	pt_jet=scale*(*caloJets)[ijet].pt();
	if(pt_jet> ptThreshold_){
	  reco::CaloJetRef calojetref(caloJets, ijet);
	  reco::JetID jetID = (*jetID_ValueMap_Handle)[calojetref];
	  iscleaned = jetIDFunctorLoose((*caloJets)[ijet], jetID);
	}
    }
    if(isTCMet_){
      pt_jet=scale*(*jptJets)[ijet].pt();
      if(pt_jet> ptThreshold_){
	const edm::RefToBase<reco::Jet>&  rawJet = (*jptJets)[ijet].getCaloJetRef();
	const reco::CaloJet *rawCaloJet = dynamic_cast<const reco::CaloJet*>(&*rawJet);
	reco::CaloJetRef const theCaloJetRef = (rawJet).castTo<reco::CaloJetRef>();
	reco::JetID jetID = (*jetID_ValueMap_Handle)[theCaloJetRef];
	iscleaned = jetIDFunctorLoose(*rawCaloJet, jetID);
      }
    }
    if(isPFMet_){
      pt_jet=scale*(*pfJets)[ijet].pt();
      if(pt_jet> ptThreshold_){
	iscleaned = pfjetIDFunctorLoose((*pfJets)[ijet]);
      }
    }
    if(iscleaned){
      bJetID=true;
    }
    if(pt_jet>pt1){
      pt2=pt1;
      ind2=ind1;
      pass_jetID2=pass_jetID1;
      pt1=pt_jet;
      ind1=ijet;
      pass_jetID1=iscleaned;
    }else if (pt_jet>pt2){
      pt2=pt_jet;
      ind2=ijet;
      pass_jetID2=iscleaned;
    }
  }
  if(pass_jetID1 && pass_jetID2){
    double dphi=-1.0;
    if(isCaloMet_){
      dphi=fabs((*caloJets)[ind1].phi()-(*caloJets)[ind2].phi());
    }
    if(isTCMet_){
      dphi=fabs((*jptJets)[ind1].phi()-(*jptJets)[ind2].phi());
    }
    if(isPFMet_){
      dphi=fabs((*pfJets)[ind1].phi()-(*pfJets)[ind2].phi());
    }
    if(dphi>acos(-1.)){
      dphi=2*acos(-1.)-dphi;
    }
    if(dphi>2.7){
      bDiJetID=true;
    }
  }

  // ==========================================================
  // HCAL Noise filter

  bool bHBHENoiseFilter = HBHENoiseFilterResult;

  // ==========================================================
  // Get BeamHaloSummary
  edm::Handle<BeamHaloSummary> TheBeamHaloSummary ;
  iEvent.getByToken(beamHaloSummaryToken_, TheBeamHaloSummary) ;

  if (!TheBeamHaloSummary.isValid()) {
    std::cout << "BeamHaloSummary doesn't exist" << std::endl;
  }

  bool bBeamHaloID = true;

  if(!TheBeamHaloSummary.isValid()) {

  const BeamHaloSummary TheSummary = (*TheBeamHaloSummary.product() );

  if( !TheSummary.EcalTightHaloId()  && !TheSummary.HcalTightHaloId() &&
      !TheSummary.CSCTightHaloId()   && !TheSummary.GlobalTightHaloId() )
    bBeamHaloID = false;
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
  // DCS Filter
  bool bDCSFilter = (bypassAllDCSChecks_ || DCSFilter_->filter(iEvent, iSetup));
  // ==========================================================
  // Reconstructed MET Information - fill MonitorElements

  for (std::vector<std::string>::const_iterator ic = folderNames_.begin();
       ic != folderNames_.end(); ic++){
    if ((*ic=="Uncleaned")  &&(isCaloMet_ || bPrimaryVertex))     fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet);
    if ((*ic=="Cleaned")    &&bDCSFilter&&bHBHENoiseFilter&&bPrimaryVertex&&bBeamHaloID&&bJetID) fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet);
    if ((*ic=="DiJet" )     &&bDCSFilter&&bHBHENoiseFilter&&bPrimaryVertex&&bBeamHaloID&&bDiJetID) fillMESet(iEvent, DirName+"/"+*ic, *met,*pfmet,*calomet);
  }
}


// ***********************************************************
void METAnalyzer::fillMESet(const edm::Event& iEvent, std::string DirName,
			    const reco::MET& met, const reco::PFMET& pfmet, const reco::CaloMET& calomet)
{

  dbe_->setCurrentFolder(DirName);

  bool bLumiSecPlot=false;
  if (DirName.find("Uncleaned")) bLumiSecPlot=true;
  fillMonitorElement(iEvent, DirName, std::string(""), met, pfmet, calomet, bLumiSecPlot);

  if (DirName.find("Cleaned")) {
    for (unsigned i = 0; i<triggerFolderLabels_.size(); i++) {
      if (triggerFolderDecisions_[i])  fillMonitorElement(iEvent, DirName, triggerFolderLabels_[i], met, pfmet, calomet, false);
    }
  }

if (DirName.find("DiJet")) {
    for (unsigned i = 0; i<triggerFolderLabels_.size(); i++) {
      if (triggerFolderDecisions_[i])  fillMonitorElement(iEvent, DirName, triggerFolderLabels_[i], met, pfmet, calomet, false);
    }
  }

//  if (trigJetMB_)
//    fillMonitorElement(iEvent,DirName,"",met,pfmet,calomet, bLumiSecPlot);
//  if (trigHighPtJet_)
//    fillMonitorElement(iEvent,DirName,"HighPtJet",met,pfmet,calomet,false);
//  if (trigLowPtJet_)
//    fillMonitorElement(iEvent,DirName,"LowPtJet",met,pfmet,calomet,false);
//  if (trigMinBias_)
//    fillMonitorElement(iEvent,DirName,"MinBias",met,pfmet,calomet,false);
//  if (trigHighMET_)
//    fillMonitorElement(iEvent,DirName,"HighMET",met,pfmet,calomet,false);
//  //  if (_trig_LowMET)
//  //    fillMonitorElement(iEvent,DirName,"LowMET",met,pfmet,calomet,false);
//  if (trigEle_)
//    fillMonitorElement(iEvent,DirName,"Ele",met,pfmet,calomet,false);
//  if (trigMuon_)
//    fillMonitorElement(iEvent,DirName,"Muon",met,pfmet,calomet,false);
}

// ***********************************************************
void METAnalyzer::fillMonitorElement(const edm::Event& iEvent, std::string DirName,
					 std::string subFolderName,
				     const reco::MET& met, const reco::PFMET & pfmet, const reco::CaloMET &calomet, bool bLumiSecPlot)
{

//  if (subFolderName=="HighPtJet") {
//    if (!selectHighPtJetEvent(iEvent)) return;
//  }
//  else if (subFolderName=="LowPtJet") {
//    if (!selectLowPtJetEvent(iEvent)) return;
//  }
//  else if (subFolderName=="HighMET") {
//    if (met.pt()<highMETThreshold_) return;
//  }
//  //  else if (subFolderName=="LowMET") {
//  //    if (met.pt()<_lowMETThreshold) return;
//  //  }
//  else if (subFolderName=="Ele") {
//    if (!selectWElectronEvent(iEvent)) return;
//  }
//  else if (subFolderName=="Muon") {
//    if (!selectWMuonEvent(iEvent)) return;
//  }

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

  if (subFolderName!="") DirName = DirName +"/"+subFolderName;

//  if (verbose_) std::cout << "etThreshold_ = " << etThreshold_ << std::endl;
//  unsigned c(0);
  if (true){
//  if (SumET>etThreshold_){
    hTrigger = dbe_->get(DirName+"/triggerResults");
//    std::cout<<"Hello"<<c++<<":"<<hTrigger <<std::endl;//":"<< hTrigger->getRootObject()<<std::endl;
    if (hTrigger       && hTrigger->getRootObject()) {
//      std::cout<<"Hello"<<c++<<std::endl;
      for (unsigned i = 0; i<allTriggerDecisions_.size();i++){ 
//        std::cout<<"Hello"<<c++<<":"<<i<<":"<< allTriggerDecisions_[i]<<":"<<allTriggerDecisions_[i]<<std::endl;
        hTrigger->Fill(i + .5, allTriggerDecisions_[i]);
        if (!hTriggerLabelsIsSet_) {
          hTrigger->setBinLabel(i+1, allTriggerNames_[i]);//Can't be done in beginJob (no trigger list). Can't be done in beginRun (would have to anticipate folder structure).FIXME doesn't work
        }
//        std::cout<<"Filling decision "<<allTriggerNames_[i]<<" "<<allTriggerDecisions_[i]<<std::endl;
      }
      if (!hTriggerLabelsIsSet_) for (unsigned i = allTriggerDecisions_.size(); i<500;i++){ 
        hTrigger->setBinLabel(i+1, "");//Can't be done in beginJob (no trigger list). Can't be done in beginRun (would have to anticipate folder structure).
      }
    }
    hTriggerLabelsIsSet_ = true;
    hMEx    = dbe_->get(DirName+"/"+"MEx");     if (hMEx           && hMEx->getRootObject()){    hMEx          ->Fill(MEx);}
    hMEy    = dbe_->get(DirName+"/"+"MEy");     if (hMEy           && hMEy->getRootObject())     hMEy          ->Fill(MEy);
    hMET    = dbe_->get(DirName+"/"+"MET");     if (hMET           && hMET->getRootObject())     hMET          ->Fill(MET);
    hMETPhi = dbe_->get(DirName+"/"+"METPhi");  if (hMETPhi        && hMETPhi->getRootObject())  hMETPhi       ->Fill(METPhi);
    hSumET  = dbe_->get(DirName+"/"+"SumET");   if (hSumET         && hSumET->getRootObject())   hSumET        ->Fill(SumET);
    hMETSig = dbe_->get(DirName+"/"+"METSig");  if (hMETSig        && hMETSig->getRootObject())  hMETSig       ->Fill(METSig);
    //hEz     = dbe_->get(DirName+"/"+"Ez");      if (hEz            && hEz->getRootObject())      hEz           ->Fill(Ez);

    hMET_logx   = dbe_->get(DirName+"/"+"MET_logx");    if (hMET_logx      && hMET_logx->getRootObject())    hMET_logx->Fill(log10(MET));
    hSumET_logx = dbe_->get(DirName+"/"+"SumET_logx");  if (hSumET_logx    && hSumET_logx->getRootObject())  hSumET_logx->Fill(log10(SumET));

    // Fill NPV profiles
      //--------------------------------------------------------------------------
    meMEx_profile   = dbe_->get(DirName + "/MEx_profile");
    meMEy_profile   = dbe_->get(DirName + "/MEy_profile");
    meMET_profile   = dbe_->get(DirName + "/MET_profile");
    meSumET_profile = dbe_->get(DirName + "/SumET_profile");
    
    if (meMEx_profile   && meMEx_profile  ->getRootObject()) meMEx_profile  ->Fill(numPV_, MEx);
    if (meMEy_profile   && meMEy_profile  ->getRootObject()) meMEy_profile  ->Fill(numPV_, MEy);
    if (meMET_profile   && meMET_profile  ->getRootObject()) meMET_profile  ->Fill(numPV_, MET);
    if (meSumET_profile && meSumET_profile->getRootObject()) meSumET_profile->Fill(numPV_, SumET);
 
    //hMETIonFeedbck = dbe_->get(DirName+"/"+"METIonFeedbck");  if (hMETIonFeedbck && hMETIonFeedbck->getRootObject())  hMETIonFeedbck->Fill(MET);
    //hMETHPDNoise   = dbe_->get(DirName+"/"+"METHPDNoise");    if (hMETHPDNoise   && hMETHPDNoise->getRootObject())    hMETHPDNoise->Fill(MET);
    //comment out like already done before for TcMET and PFMET
    if(isTCMet_ || metCollectionLabel_.label() == "corMetGlobalMuons"){
      hMETIonFeedbck = dbe_->get(DirName+"/"+"METIonFeedbck");  if (hMETIonFeedbck && hMETIonFeedbck->getRootObject()) hMETIonFeedbck->Fill(MET);
      hMETHPDNoise   = dbe_->get(DirName+"/"+"METHPDNoise");    if (hMETHPDNoise   && hMETHPDNoise->getRootObject())   hMETHPDNoise->Fill(MET);
      hMETRBXNoise   = dbe_->get(DirName+"/"+"METRBXNoise");    if (hMETRBXNoise   && hMETRBXNoise->getRootObject())   hMETRBXNoise->Fill(MET);
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

      hCaloMaxEtInEmTowers  = dbe_->get(DirName+"/"+"CaloMaxEtInEmTowers");   if (hCaloMaxEtInEmTowers  && hCaloMaxEtInEmTowers->getRootObject())   hCaloMaxEtInEmTowers->Fill(caloMaxEtInEMTowers);
      hCaloMaxEtInHadTowers = dbe_->get(DirName+"/"+"CaloMaxEtInHadTowers");  if (hCaloMaxEtInHadTowers && hCaloMaxEtInHadTowers->getRootObject())  hCaloMaxEtInHadTowers->Fill(caloMaxEtInHadTowers);
      
      hCaloHadEtInHB = dbe_->get(DirName+"/"+"CaloHadEtInHB");  if (hCaloHadEtInHB  &&  hCaloHadEtInHB->getRootObject())  hCaloHadEtInHB->Fill(caloHadEtInHB);
      hCaloHadEtInHO = dbe_->get(DirName+"/"+"CaloHadEtInHO");  if (hCaloHadEtInHO  &&  hCaloHadEtInHO->getRootObject())  hCaloHadEtInHO->Fill(caloHadEtInHO);
      hCaloHadEtInHE = dbe_->get(DirName+"/"+"CaloHadEtInHE");  if (hCaloHadEtInHE  &&  hCaloHadEtInHE->getRootObject())  hCaloHadEtInHE->Fill(caloHadEtInHE);
      hCaloHadEtInHF = dbe_->get(DirName+"/"+"CaloHadEtInHF");  if (hCaloHadEtInHF  &&  hCaloHadEtInHF->getRootObject())  hCaloHadEtInHF->Fill(caloHadEtInHF);
      hCaloEmEtInEB  = dbe_->get(DirName+"/"+"CaloEmEtInEB");   if (hCaloEmEtInEB   &&  hCaloEmEtInEB->getRootObject())   hCaloEmEtInEB->Fill(caloEmEtInEB);
      hCaloEmEtInEE  = dbe_->get(DirName+"/"+"CaloEmEtInEE");   if (hCaloEmEtInEE   &&  hCaloEmEtInEE->getRootObject())   hCaloEmEtInEE->Fill(caloEmEtInEE);
      hCaloEmEtInHF  = dbe_->get(DirName+"/"+"CaloEmEtInHF");   if (hCaloEmEtInHF   &&  hCaloEmEtInHF->getRootObject())   hCaloEmEtInHF->Fill(caloEmEtInHF);

      hCaloMETPhi020 = dbe_->get(DirName+"/"+"CaloMETPhi020");    if (MET> 20. && hCaloMETPhi020  &&  hCaloMETPhi020->getRootObject()) { hCaloMETPhi020->Fill(METPhi);}


      hCaloEtFractionHadronic = dbe_->get(DirName+"/"+"CaloEtFractionHadronic"); if (hCaloEtFractionHadronic && hCaloEtFractionHadronic->getRootObject())  hCaloEtFractionHadronic->Fill(caloEtFractionHadronic);
      hCaloEmEtFraction       = dbe_->get(DirName+"/"+"CaloEmEtFraction");       if (hCaloEmEtFraction       && hCaloEmEtFraction->getRootObject())        hCaloEmEtFraction->Fill(caloEmEtFraction);
      hCaloEmEtFraction020 = dbe_->get(DirName+"/"+"CaloEmEtFraction020");       if (MET> 20.  &&  hCaloEmEtFraction020    && hCaloEmEtFraction020->getRootObject()) hCaloEmEtFraction020->Fill(caloEmEtFraction);
      if (metCollectionLabel_.label() == "corMetGlobalMuons" ) {
	
	for( reco::MuonCollection::const_iterator muonit = muonHandle_->begin(); muonit != muonHandle_->end(); muonit++ ) {
	  const reco::TrackRef siTrack = muonit->innerTrack();
	  hCalomuPt    = dbe_->get(DirName+"/"+"CalomuonPt");  
	  if (hCalomuPt    && hCalomuPt->getRootObject())   hCalomuPt->Fill( muonit->p4().pt() );
	  hCalomuEta   = dbe_->get(DirName+"/"+"CalomuonEta");    if (hCalomuEta   && hCalomuEta->getRootObject())    hCalomuEta->Fill( muonit->p4().eta() );
	  hCalomuNhits = dbe_->get(DirName+"/"+"CalomuonNhits");  if (hCalomuNhits && hCalomuNhits->getRootObject())  hCalomuNhits->Fill( siTrack.isNonnull() ? siTrack->numberOfValidHits() : -999 );
	  hCalomuChi2  = dbe_->get(DirName+"/"+"CalomuonNormalizedChi2");   if (hCalomuChi2  && hCalomuChi2->getRootObject())   hCalomuChi2->Fill( siTrack.isNonnull() ? siTrack->chi2()/siTrack->ndof() : -999 );
	  double d0 = siTrack.isNonnull() ? -1 * siTrack->dxy( beamSpot_) : -999;
	  hCalomuD0    = dbe_->get(DirName+"/"+"CalomuonD0");     if (hCalomuD0    && hCalomuD0->getRootObject())  hCalomuD0->Fill( d0 );
	}
	
	const unsigned int nMuons = muonHandle_->size();
	for( unsigned int mus = 0; mus < nMuons; mus++ ) {
	  reco::MuonRef muref( muonHandle_, mus);
	  reco::MuonMETCorrectionData muCorrData = (*tcMetValueMapHandle_)[muref];
	  hCaloMExCorrection      = dbe_->get(DirName+"/"+"CaloMExCorrection");       if (hCaloMExCorrection      && hCaloMExCorrection->getRootObject())       hCaloMExCorrection-> Fill(muCorrData.corrY());
	  hCaloMEyCorrection      = dbe_->get(DirName+"/"+"CaloMEyCorrection");       if (hCaloMEyCorrection      && hCaloMEyCorrection->getRootObject())       hCaloMEyCorrection-> Fill(muCorrData.corrX());
	  hCaloMuonCorrectionFlag = dbe_->get(DirName+"/"+"CaloMuonCorrectionFlag");  if (hCaloMuonCorrectionFlag && hCaloMuonCorrectionFlag->getRootObject())  hCaloMuonCorrectionFlag-> Fill(muCorrData.type());
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
      
      mePhotonEtFraction        = dbe_->get(DirName + "/PfPhotonEtFraction");
      mePhotonEt                = dbe_->get(DirName + "/PfPhotonEt");
      meNeutralHadronEtFraction = dbe_->get(DirName + "/PfNeutralHadronEtFraction");
      meNeutralHadronEt         = dbe_->get(DirName + "/PfNeutralHadronEt");
      meElectronEtFraction      = dbe_->get(DirName + "/PfElectronEtFraction");
      meElectronEt              = dbe_->get(DirName + "/PfElectronEt");
      meChargedHadronEtFraction = dbe_->get(DirName + "/PfChargedHadronEtFraction");
      meChargedHadronEt         = dbe_->get(DirName + "/PfChargedHadronEt");
      meMuonEtFraction          = dbe_->get(DirName + "/PfMuonEtFraction");
      meMuonEt                  = dbe_->get(DirName + "/PfMuonEt");
      meHFHadronEtFraction      = dbe_->get(DirName + "/PfHFHadronEtFraction");
      meHFHadronEt              = dbe_->get(DirName + "/PfHFHadronEt");
      meHFEMEtFraction          = dbe_->get(DirName + "/PfHFEMEtFraction");
      meHFEMEt                  = dbe_->get(DirName + "/PfHFEMEt");
      
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
      
      mePhotonEtFraction_profile        = dbe_->get(DirName + "/PfPhotonEtFraction_profile");
      mePhotonEt_profile                = dbe_->get(DirName + "/PfPhotonEt_profile");
      meNeutralHadronEtFraction_profile = dbe_->get(DirName + "/PfNeutralHadronEtFraction_profile");
      meNeutralHadronEt_profile         = dbe_->get(DirName + "/PfNeutralHadronEt_profile");
      meElectronEtFraction_profile      = dbe_->get(DirName + "/PfElectronEtFraction_profile");
      meElectronEt_profile              = dbe_->get(DirName + "/PfElectronEt_profile");
      meChargedHadronEtFraction_profile = dbe_->get(DirName + "/PfChargedHadronEtFraction_profile");
      meChargedHadronEt_profile         = dbe_->get(DirName + "/PfChargedHadronEt_profile");
      meMuonEtFraction_profile          = dbe_->get(DirName + "/PfMuonEtFraction_profile");
      meMuonEt_profile                  = dbe_->get(DirName + "/PfMuonEt_profile");
      meHFHadronEtFraction_profile      = dbe_->get(DirName + "/PfHFHadronEtFraction_profile");
      meHFHadronEt_profile              = dbe_->get(DirName + "/PfHFHadronEt_profile");
      meHFEMEtFraction_profile          = dbe_->get(DirName + "/PfHFEMEtFraction_profile");
      meHFEMEt_profile                  = dbe_->get(DirName + "/PfHFEMEt_profile");
      
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
	hMExLS = dbe_->get(DirName+"/"+"MExLS"); if (hMExLS  &&  hMExLS->getRootObject())   hMExLS->Fill(MEx,myLuminosityBlock);
	hMEyLS = dbe_->get(DirName+"/"+"MEyLS"); if (hMEyLS  &&  hMEyLS->getRootObject())   hMEyLS->Fill(MEy,myLuminosityBlock);
      }
    } 

    ////////////////////////////////////
    if (isTCMet_) {

      if(trackHandle_.isValid()) {
	for( edm::View<reco::Track>::const_iterator trkit = trackHandle_->begin(); trkit != trackHandle_->end(); trkit++ ) {
	  htrkPt    = dbe_->get(DirName+"/"+"trackPt");     if (htrkPt    && htrkPt->getRootObject())     htrkPt->Fill( trkit->pt() );
	  htrkEta   = dbe_->get(DirName+"/"+"trackEta");    if (htrkEta   && htrkEta->getRootObject())    htrkEta->Fill( trkit->eta() );
	  htrkNhits = dbe_->get(DirName+"/"+"trackNhits");  if (htrkNhits && htrkNhits->getRootObject())  htrkNhits->Fill( trkit->numberOfValidHits() );
	  htrkChi2  = dbe_->get(DirName+"/"+"trackNormalizedChi2");  
	  if (htrkChi2  && htrkChi2->getRootObject())   htrkChi2->Fill( trkit->chi2() / trkit->ndof() );
	  double d0 = -1 * trkit->dxy( beamSpot_ );
	  htrkD0    = dbe_->get(DirName+"/"+"trackD0");     if (htrkD0 && htrkD0->getRootObject())        htrkD0->Fill( d0 );
	}
      }else{std::cout<<"tracks not valid"<<std::endl;}

      if(electronHandle_.isValid()) {
	for( edm::View<reco::GsfElectron>::const_iterator eleit = electronHandle_->begin(); eleit != electronHandle_->end(); eleit++ ) {
	  helePt  = dbe_->get(DirName+"/"+"electronPt");   if (helePt  && helePt->getRootObject())   helePt->Fill( eleit->p4().pt() );
	  heleEta = dbe_->get(DirName+"/"+"electronEta");  if (heleEta && heleEta->getRootObject())  heleEta->Fill( eleit->p4().eta() );
	  heleHoE = dbe_->get(DirName+"/"+"electronHoverE");  if (heleHoE && heleHoE->getRootObject())  heleHoE->Fill( eleit->hadronicOverEm() );
	}
      }else{
	std::cout<<"electrons not valid"<<std::endl;
      }

      if(muonHandle_.isValid()) {
	for( reco::MuonCollection::const_iterator muonit = muonHandle_->begin(); muonit != muonHandle_->end(); muonit++ ) {
	  const reco::TrackRef siTrack = muonit->innerTrack();
	  hmuPt    = dbe_->get(DirName+"/"+"muonPt");     if (hmuPt    && hmuPt->getRootObject())  hmuPt   ->Fill( muonit->p4().pt() );
	  hmuEta   = dbe_->get(DirName+"/"+"muonEta");    if (hmuEta   && hmuEta->getRootObject())  hmuEta  ->Fill( muonit->p4().eta() );
	  hmuNhits = dbe_->get(DirName+"/"+"muonNhits");  if (hmuNhits && hmuNhits->getRootObject())  hmuNhits->Fill( siTrack.isNonnull() ? siTrack->numberOfValidHits() : -999 );
	  hmuChi2  = dbe_->get(DirName+"/"+"muonNormalizedChi2");   if (hmuChi2  && hmuChi2->getRootObject())  hmuChi2 ->Fill( siTrack.isNonnull() ? siTrack->chi2()/siTrack->ndof() : -999 );
	  double d0 = siTrack.isNonnull() ? -1 * siTrack->dxy( beamSpot_) : -999;
	  hmuD0    = dbe_->get(DirName+"/"+"muonD0");     if (hmuD0    && hmuD0->getRootObject())  hmuD0->Fill( d0 );
	}
	const unsigned int nMuons = muonHandle_->size();
	for( unsigned int mus = 0; mus < nMuons; mus++ ) {
	  reco::MuonRef muref( muonHandle_, mus);
	  reco::MuonMETCorrectionData muCorrData = (*tcMetValueMapHandle_)[muref];
	  hMExCorrection      = dbe_->get(DirName+"/"+"MExCorrection");       if (hMExCorrection      && hMExCorrection->getRootObject())       hMExCorrection-> Fill(muCorrData.corrY());
	  hMEyCorrection      = dbe_->get(DirName+"/"+"MEyCorrection");       if (hMEyCorrection      && hMEyCorrection->getRootObject())       hMEyCorrection-> Fill(muCorrData.corrX());
	  hMuonCorrectionFlag = dbe_->get(DirName+"/"+"CorrectionFlag");  if (hMuonCorrectionFlag && hMuonCorrectionFlag->getRootObject())  hMuonCorrectionFlag-> Fill(muCorrData.type());
	}
      }else{
	std::cout<<"muons not valid"<<std::endl;
      }
    }
  } // et threshold cut
}

//// ***********************************************************
//bool METAnalyzer::selectHighPtJetEvent(const edm::Event& iEvent){
//
//  bool return_value=false;
//
//  if(isCaloMet_){
//    edm::Handle<reco::CaloJetCollection> caloJets;
//    iEvent.getByToken(caloJetsToken_, caloJets);
//    if (!caloJets.isValid()) {
//      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
//      if (verbose_) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
//    }
//    
//    for (reco::CaloJetCollection::const_iterator cal = caloJets->begin();
//	 cal!=caloJets->end(); ++cal){
//      if (cal->pt()>highPtJetThreshold_){
//	return_value=true;
//      }
//    }
//  }
//  if(isTCMet_){
//    edm::Handle<reco::JPTJetCollection> jptJets;
//    iEvent.getByToken(jptJetsToken_, jptJets);
//    if (!jptJets.isValid()) {
//      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
//      if (verbose_) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
//    }
//    
//    for (reco::JPTJetCollection::const_iterator cal = jptJets->begin();
//	 cal!=jptJets->end(); ++cal){
//      if (cal->pt()>highPtJetThreshold_){
//	return_value=true;
//      }
//    }
//  }
//  if(isPFMet_){
//    edm::Handle<reco::PFJetCollection> PFJets;
//    iEvent.getByToken(pfJetsToken_, PFJets);
//    if (!PFJets.isValid()) {
//      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
//      if (verbose_) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
//    }
//    for (reco::PFJetCollection::const_iterator cal = PFJets->begin();
//	 cal!=PFJets->end(); ++cal){
//      if (cal->pt()>highPtJetThreshold_){
//	return_value=true;
//      }
//    }
//  }
//
//
//  return return_value;
//}
//
//// // ***********************************************************
//bool METAnalyzer::selectLowPtJetEvent(const edm::Event& iEvent){
//
//  bool return_value=false;
//  if(isCaloMet_){
//    edm::Handle<reco::CaloJetCollection> caloJets;
//    iEvent.getByToken(caloJetsToken_, caloJets);
//    if (!caloJets.isValid()) {
//      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
//      if (verbose_) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
//    }
//    
//    for (reco::CaloJetCollection::const_iterator cal = caloJets->begin();
//	 cal!=caloJets->end(); ++cal){
//      if (cal->pt()>lowPtJetThreshold_){
//	return_value=true;
//      }
//    }
//  }
//  if(isTCMet_){
//    edm::Handle<reco::JPTJetCollection> jptJets;
//    iEvent.getByToken(jptJetsToken_, jptJets);
//    if (!jptJets.isValid()) {
//      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
//      if (verbose_) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
//    }
//    
//    for (reco::JPTJetCollection::const_iterator cal = jptJets->begin();
//	 cal!=jptJets->end(); ++cal){
//      if (cal->pt()>lowPtJetThreshold_){
//	return_value=true;
//      }
//    }
//  }
//  if(isPFMet_){
//    edm::Handle<reco::PFJetCollection> PFJets;
//    iEvent.getByToken(pfJetsToken_, PFJets);
//    if (!PFJets.isValid()) {
//      LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
//      if (verbose_) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
//    }
//    for (reco::PFJetCollection::const_iterator cal = PFJets->begin();
//	 cal!=PFJets->end(); ++cal){
//      if (cal->pt()>lowPtJetThreshold_){
//	return_value=true;
//      }
//    }
//  }
//  return return_value;
//
//}
//
//
//// ***********************************************************
//bool METAnalyzer::selectWElectronEvent(const edm::Event& iEvent){
//
//  bool return_value=true;
//
//  /*
//    W-electron event selection comes here
//   */
//
//  return return_value;
//
//}
//
//// ***********************************************************
//bool METAnalyzer::selectWMuonEvent(const edm::Event& iEvent){
//
//  bool return_value=true;
//
//  /*
//    W-muon event selection comes here
//   */
//
//  return return_value;
//
//}


/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/03/06 11:39:22 $
 *  $Revision: 1.41 $
 *  \author K. Hatakeyama - Rockefeller University
 *          A.Apresyan - Caltech
 */

#include "DQMOffline/JetMET/interface/PFMETAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
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
PFMETAnalyzer::PFMETAnalyzer(const edm::ParameterSet& pSet) {

  parameters = pSet;

  edm::ParameterSet highptjetparms = parameters.getParameter<edm::ParameterSet>("highPtJetTrigger");
  edm::ParameterSet lowptjetparms  = parameters.getParameter<edm::ParameterSet>("lowPtJetTrigger" );
  edm::ParameterSet minbiasparms   = parameters.getParameter<edm::ParameterSet>("minBiasTrigger"  );
  edm::ParameterSet highmetparms   = parameters.getParameter<edm::ParameterSet>("highMETTrigger"  );
  edm::ParameterSet lowmetparms    = parameters.getParameter<edm::ParameterSet>("lowMETTrigger"   );
  edm::ParameterSet eleparms       = parameters.getParameter<edm::ParameterSet>("eleTrigger"      );
  edm::ParameterSet muonparms      = parameters.getParameter<edm::ParameterSet>("muonTrigger"     );

  //genericTriggerEventFlag_( new GenericTriggerEventFlag( conf_ ) );
  _HighPtJetEventFlag = new GenericTriggerEventFlag( highptjetparms );
  _LowPtJetEventFlag  = new GenericTriggerEventFlag( lowptjetparms  );
  _MinBiasEventFlag   = new GenericTriggerEventFlag( minbiasparms   );
  _HighMETEventFlag   = new GenericTriggerEventFlag( highmetparms   );
  _LowMETEventFlag    = new GenericTriggerEventFlag( lowmetparms    );
  _EleEventFlag       = new GenericTriggerEventFlag( eleparms       );
  _MuonEventFlag      = new GenericTriggerEventFlag( muonparms      );

  highPtJetExpr_ = highptjetparms.getParameter<std::vector<std::string> >("hltPaths");
  lowPtJetExpr_  = lowptjetparms .getParameter<std::vector<std::string> >("hltPaths");
  highMETExpr_   = highmetparms  .getParameter<std::vector<std::string> >("hltPaths");
  lowMETExpr_    = lowmetparms   .getParameter<std::vector<std::string> >("hltPaths");
  muonExpr_      = muonparms     .getParameter<std::vector<std::string> >("hltPaths");
  elecExpr_      = eleparms      .getParameter<std::vector<std::string> >("hltPaths");
  minbiasExpr_   = minbiasparms  .getParameter<std::vector<std::string> >("hltPaths");

}

// ***********************************************************
PFMETAnalyzer::~PFMETAnalyzer() { 

  delete _HighPtJetEventFlag;
  delete _LowPtJetEventFlag;
  delete _MinBiasEventFlag;
  delete _HighMETEventFlag;
  delete _LowMETEventFlag;
  delete _EleEventFlag;
  delete _MuonEventFlag;

}

void PFMETAnalyzer::beginJob(DQMStore * dbe) {

  evtCounter = 0;
  metname = "pfMETAnalyzer";

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
  _tightHcalFiltering   = theCleaningParameters.getParameter<bool>("tightHcalFiltering");

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


  // PFMET information
  thePfMETCollectionLabel       = parameters.getParameter<edm::InputTag>("METCollectionLabel");
  thePfJetCollectionLabel       = parameters.getParameter<edm::InputTag>("PfJetCollectionLabel");
  _source                       = parameters.getParameter<std::string>("Source");

  // Other data collections
  theJetCollectionLabel       = parameters.getParameter<edm::InputTag>("JetCollectionLabel");
  PFCandidatesTag             = parameters.getParameter<edm::InputTag>("PFCandidates");
  HcalNoiseRBXCollectionTag   = parameters.getParameter<edm::InputTag>("HcalNoiseRBXCollection");
  HcalNoiseSummaryTag         = parameters.getParameter<edm::InputTag>("HcalNoiseSummary");
  BeamHaloSummaryTag          = parameters.getParameter<edm::InputTag>("BeamHaloSummaryLabel");

  // misc
  _verbose     = parameters.getParameter<int>("verbose");
  _etThreshold = parameters.getParameter<double>("etThreshold"); // MET threshold
  _allhist     = parameters.getParameter<bool>("allHist");       // Full set of monitoring histograms
  _allSelection= parameters.getParameter<bool>("allSelection");  // Plot with all sets of event selection
  _cleanupSelection= parameters.getParameter<bool>("cleanupSelection");  // Plot with all sets of event selection

  _highPtPFJetThreshold = parameters.getParameter<double>("HighPtJetThreshold"); // High Pt Jet threshold
  _lowPtPFJetThreshold  = parameters.getParameter<double>("LowPtJetThreshold");   // Low Pt Jet threshold
  _highPFMETThreshold   = parameters.getParameter<double>("HighMETThreshold");     // High MET threshold
  _lowPFMETThreshold    = parameters.getParameter<double>("LowMETThreshold");       // Low MET threshold

  //
  jetID = new reco::helper::JetIDHelper(parameters.getParameter<ParameterSet>("JetIDParams"));

  // DQStore stuff
  LogTrace(metname)<<"[PFMETAnalyzer] Parameters initialization";
  std::string DirName = "JetMET/MET/"+_source;
  dbe->setCurrentFolder(DirName);

  metME = dbe->book1D("metReco", "metReco", 4, 1, 5);
  metME->setBinLabel(3,"PFMET",1);

  _dbe = dbe;

  _FolderNames.push_back("All");
  _FolderNames.push_back("BasicCleanup");
  _FolderNames.push_back("ExtraCleanup");
  _FolderNames.push_back("HcalNoiseFilter");
  _FolderNames.push_back("HcalNoiseFilterTight");
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
      if (*ic=="HcalNoiseFilterTight") bookMESet(DirName+"/"+*ic);
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
void PFMETAnalyzer::endJob() {

  delete jetID;
  delete DCSFilter;

}

// ***********************************************************
void PFMETAnalyzer::bookMESet(std::string DirName)
{

  bool bLumiSecPlot=false;
  if (DirName.find("All")!=std::string::npos) bLumiSecPlot=true;

  bookMonitorElement(DirName,bLumiSecPlot);

  if ( _HighPtJetEventFlag->on() ) {
    bookMonitorElement(DirName+"/"+"HighPtJet",false);
    meTriggerName_HighPtJet = _dbe->bookString("triggerName_HighPtJet", highPtJetExpr_[0]);
  }  

  if ( _LowPtJetEventFlag->on() ) {
    bookMonitorElement(DirName+"/"+"LowPtJet",false);
    meTriggerName_LowPtJet = _dbe->bookString("triggerName_LowPtJet", lowPtJetExpr_[0]);
  }

  if ( _MinBiasEventFlag->on() ) {
    bookMonitorElement(DirName+"/"+"MinBias",false);
    meTriggerName_MinBias = _dbe->bookString("triggerName_MinBias", minbiasExpr_[0]);
    if (_verbose) std::cout << "_MinBiasEventFlag is on, folder created\n";
  }

  if ( _HighMETEventFlag->on() ) {
    bookMonitorElement(DirName+"/"+"HighMET",false);
    meTriggerName_HighMET = _dbe->bookString("triggerName_HighMET", highMETExpr_[0]);
  }

  if ( _LowMETEventFlag->on() ) {
    bookMonitorElement(DirName+"/"+"LowMET",false);
    meTriggerName_LowMET = _dbe->bookString("triggerName_LowMET", lowMETExpr_[0]);
  }

  if ( _EleEventFlag->on() ) {
    bookMonitorElement(DirName+"/"+"Ele",false);
    meTriggerName_Ele = _dbe->bookString("triggerName_Ele", elecExpr_[0]);
    if (_verbose) std::cout << "_EleEventFlag is on, folder created\n";
  }

  if ( _MuonEventFlag->on() ) {
    bookMonitorElement(DirName+"/"+"Muon",false);
    meTriggerName_Muon = _dbe->bookString("triggerName_Muon", muonExpr_[0]);
    if (_verbose) std::cout << "_MuonEventFlag is on, folder created\n";
  }
}

// ***********************************************************
void PFMETAnalyzer::bookMonitorElement(std::string DirName, bool bLumiSecPlot=false)
{

  if (_verbose) std::cout << "booMonitorElement " << DirName << std::endl;
  _dbe->setCurrentFolder(DirName);
 
  //meNevents              = _dbe->book1D("METTask_Nevents", "METTask_Nevents"   ,1,0,1);
  mePfMEx                = _dbe->book1D("METTask_PfMEx",   "METTask_PfMEx"   ,200,-500,500); 
  mePfMEx->setAxisTitle("MEx [GeV]",1);
  mePfMEy                = _dbe->book1D("METTask_PfMEy",   "METTask_PfMEy"   ,200,-500,500); 
  mePfMEy->setAxisTitle("MEy [GeV]",1);
  //mePfEz                 = _dbe->book1D("METTask_PfEz",    "METTask_PfEz"    ,500,-500,500);
  //mePfEz->setAxisTitle("MEz [GeV]",1);
  mePfMETSig             = _dbe->book1D("METTask_PfMETSig","METTask_PfMETSig",51,0,51);
  mePfMETSig->setAxisTitle("METSig",1);
  mePfMET                = _dbe->book1D("METTask_PfMET",   "METTask_PfMET"   ,200,0,1000); 
  mePfMET->setAxisTitle("MET [GeV]",1);
  mePfMETPhi             = _dbe->book1D("METTask_PfMETPhi","METTask_PfMETPhi",60,-TMath::Pi(),TMath::Pi());
  mePfMETPhi->setAxisTitle("METPhi [rad]",1);
  mePfSumET              = _dbe->book1D("METTask_PfSumET", "METTask_PfSumET" ,400,0,2000); 
  mePfSumET->setAxisTitle("SumET [GeV]",1);

  mePfMET_logx           = _dbe->book1D("METTask_PfMET_logx",   "METTask_PfMET_logx"   ,40,-1.,7.);
  mePfMET_logx->setAxisTitle("log(MET) [GeV]",1);
  mePfSumET_logx         = _dbe->book1D("METTask_PfSumET_logx", "METTask_PfSumET_logx" ,40,-1.,7.);
  mePfSumET_logx->setAxisTitle("log(SumET) [GeV]",1);

  mePfNeutralEMFraction  = _dbe->book1D("METTask_PfNeutralEMFraction", "METTask_PfNeutralEMFraction" ,50,0.,1.);
  mePfNeutralEMFraction->setAxisTitle("Pf Neutral EM Fraction",1);
  mePfNeutralHadFraction = _dbe->book1D("METTask_PfNeutralHadFraction","METTask_PfNeutralHadFraction",50,0.,1.);
  mePfNeutralHadFraction->setAxisTitle("Pf Neutral Had Fraction",1);
  mePfChargedEMFraction  = _dbe->book1D("METTask_PfChargedEMFraction", "METTask_PfChargedEMFraction" ,50,0.,1.);
  mePfChargedEMFraction->setAxisTitle("Pf Charged EM Fraction",1);
  mePfChargedHadFraction = _dbe->book1D("METTask_PfChargedHadFraction","METTask_PfChargedHadFraction",50,0.,1.);
  mePfChargedHadFraction->setAxisTitle("Pf Charged Had Fraction",1);
  mePfMuonFraction       = _dbe->book1D("METTask_PfMuonFraction",      "METTask_PfMuonFraction"      ,50,0.,1.);
  mePfMuonFraction->setAxisTitle("Pf Muon Fraction",1);

  //mePfMETIonFeedbck      = _dbe->book1D("METTask_PfMETIonFeedbck", "METTask_PfMETIonFeedbck" ,500,0,1000);
  //mePfMETIonFeedbck->setAxisTitle("MET [GeV]",1);
  //mePfMETHPDNoise        = _dbe->book1D("METTask_PfMETHPDNoise",   "METTask_PfMETHPDNoise"   ,500,0,1000);
  //mePfMETHPDNoise->setAxisTitle("MET [GeV]",1);
  //mePfMETRBXNoise        = _dbe->book1D("METTask_PfMETRBXNoise",   "METTask_PfMETRBXNoise"   ,500,0,1000);
  //mePfMETRBXNoise->setAxisTitle("MET [GeV]",1);

  if (_allhist){
    if (bLumiSecPlot){
      mePfMExLS              = _dbe->book2D("METTask_PfMEx_LS","METTask_PfMEx_LS",200,-200,200,50,0.,500.);
      mePfMExLS->setAxisTitle("MEx [GeV]",1);
      mePfMExLS->setAxisTitle("Lumi Section",2);
      mePfMEyLS              = _dbe->book2D("METTask_PfMEy_LS","METTask_PfMEy_LS",200,-200,200,50,0.,500.);
      mePfMEyLS->setAxisTitle("MEy [GeV]",1);
      mePfMEyLS->setAxisTitle("Lumi Section",2);
    }
  }


  // NPV binned
  //----------------------------------------------------------------------------
  mePfMEx_profile = _dbe->bookProfile("METTask_PfMEx_profile", "METTask_PfMEx_profile", 50, 0, 50, 200, -500, 500);

  for (int bin=0; bin<_npvRanges; ++bin) {

    mePfMEx_npv[bin]                = _dbe->book1D(Form("METTask_PfMEx_npvBin%d", bin),   "METTask_PfMEx"+_npvs[bin]   ,200,-500,500); 
    mePfMEy_npv[bin]                = _dbe->book1D(Form("METTask_PfMEy_npvBin%d", bin),   "METTask_PfMEy"+_npvs[bin]   ,200,-500,500); 
    //mePfEz_npv[bin]                 = _dbe->book1D(Form("METTask_PfEz_npvBin%d", bin),    "METTask_PfEz"+_npvs[bin]    ,500,-500,500);
    //mePfMETSig_npv[bin]             = _dbe->book1D(Form("METTask_PfMETSig_npvBin%d", bin),"METTask_PfMETSig"+_npvs[bin],51,0,51);
    mePfMET_npv[bin]                = _dbe->book1D(Form("METTask_PfMET_npvBin%d", bin),   "METTask_PfMET"+_npvs[bin]   ,200,0,1000); 
    //mePfMETPhi_npv[bin]             = _dbe->book1D(Form("METTask_PfMETPhi_npvBin%d", bin),"METTask_PfMETPhi"+_npvs[bin],60,-TMath::Pi(),TMath::Pi());
    mePfSumET_npv[bin]              = _dbe->book1D(Form("METTask_PfSumET_npvBin%d", bin), "METTask_PfSumET"+_npvs[bin] ,400,0,2000); 

    mePfMET_logx_npv[bin]           = _dbe->book1D(Form("METTask_PfMET_logx_npvBin%d", bin),   "METTask_PfMET_logx"+_npvs[bin]   ,40,-1.,7.);
    mePfSumET_logx_npv[bin]         = _dbe->book1D(Form("METTask_PfSumET_logx_npvBin%d", bin), "METTask_PfSumET_logx"+_npvs[bin] ,40,-1.,7.);
    
    mePfNeutralEMFraction_npv[bin]  = _dbe->book1D(Form("METTask_PfNeutralEMFraction_npvBin%d", bin), "METTask_PfNeutralEMFraction"+_npvs[bin] ,50,0.,1.);
    mePfNeutralHadFraction_npv[bin] = _dbe->book1D(Form("METTask_PfNeutralHadFraction_npvBin%d", bin),"METTask_PfNeutralHadFraction"+_npvs[bin],50,0.,1.);
    mePfChargedEMFraction_npv[bin]  = _dbe->book1D(Form("METTask_PfChargedEMFraction_npvBin%d", bin), "METTask_PfChargedEMFraction"+_npvs[bin] ,50,0.,1.);
    mePfChargedHadFraction_npv[bin] = _dbe->book1D(Form("METTask_PfChargedHadFraction_npvBin%d", bin),"METTask_PfChargedHadFraction"+_npvs[bin],50,0.,1.);
    mePfMuonFraction_npv[bin]       = _dbe->book1D(Form("METTask_PfMuonFraction_npvBin%d", bin),      "METTask_PfMuonFraction"+_npvs[bin]      ,50,0.,1.);

    ///////////////////////////////////    mePfMEx_profile->getTH1()->Sumw2();  // crashes at run time

    mePfMEx_npv               [bin]->setAxisTitle("MEx [GeV]",1);
    mePfMEy_npv               [bin]->setAxisTitle("MEy [GeV]",1);
    //    mePfEz_npv                [bin]->setAxisTitle("MEz [GeV]",1);
    //    mePfMETSig_npv            [bin]->setAxisTitle("METSig",1);
    mePfMET_npv               [bin]->setAxisTitle("MET [GeV]",1);
    //    mePfMETPhi_npv            [bin]->setAxisTitle("METPhi [rad]",1);
    mePfSumET_npv             [bin]->setAxisTitle("SumET [GeV]",1);
    mePfMET_logx_npv          [bin]->setAxisTitle("log(MET) [GeV]",1);
    mePfSumET_logx_npv        [bin]->setAxisTitle("log(SumET) [GeV]",1);
    mePfNeutralEMFraction_npv [bin]->setAxisTitle("Pf Neutral EM Fraction",1);
    mePfNeutralHadFraction_npv[bin]->setAxisTitle("Pf Neutral Had Fraction",1);
    mePfChargedEMFraction_npv [bin]->setAxisTitle("Pf Charged EM Fraction",1);
    mePfChargedHadFraction_npv[bin]->setAxisTitle("Pf Charged Had Fraction",1);
    mePfMuonFraction_npv      [bin]->setAxisTitle("Pf Muon Fraction",1);
  }
}


// ***********************************************************
void PFMETAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  if ( _HighPtJetEventFlag->on() ) _HighPtJetEventFlag->initRun( iRun, iSetup );
  if ( _LowPtJetEventFlag ->on() ) _LowPtJetEventFlag ->initRun( iRun, iSetup );
  if ( _MinBiasEventFlag  ->on() ) _MinBiasEventFlag  ->initRun( iRun, iSetup );
  if ( _HighMETEventFlag  ->on() ) _HighMETEventFlag  ->initRun( iRun, iSetup );
  if ( _LowMETEventFlag   ->on() ) _LowMETEventFlag   ->initRun( iRun, iSetup );
  if ( _EleEventFlag      ->on() ) _EleEventFlag      ->initRun( iRun, iSetup );
  if ( _MuonEventFlag     ->on() ) _MuonEventFlag     ->initRun( iRun, iSetup );

  if (_HighPtJetEventFlag->on() && _HighPtJetEventFlag->expressionsFromDB(_HighPtJetEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    highPtJetExpr_ = _HighPtJetEventFlag->expressionsFromDB(_HighPtJetEventFlag->hltDBKey(), iSetup);
  if (_LowPtJetEventFlag->on() && _LowPtJetEventFlag->expressionsFromDB(_LowPtJetEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    lowPtJetExpr_  = _LowPtJetEventFlag->expressionsFromDB(_LowPtJetEventFlag->hltDBKey(),   iSetup);
  if (_HighMETEventFlag->on() && _HighMETEventFlag->expressionsFromDB(_HighMETEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    highMETExpr_   = _HighMETEventFlag->expressionsFromDB(_HighMETEventFlag->hltDBKey(),     iSetup);
  if (_LowMETEventFlag->on() && _LowMETEventFlag->expressionsFromDB(_LowMETEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    lowMETExpr_    = _LowMETEventFlag->expressionsFromDB(_LowMETEventFlag->hltDBKey(),       iSetup);
  if (_MuonEventFlag->on() && _MuonEventFlag->expressionsFromDB(_MuonEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    muonExpr_      = _MuonEventFlag->expressionsFromDB(_MuonEventFlag->hltDBKey(),           iSetup);
  if (_EleEventFlag->on() && _EleEventFlag->expressionsFromDB(_EleEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    elecExpr_      = _EleEventFlag->expressionsFromDB(_EleEventFlag->hltDBKey(),             iSetup);
  if (_MinBiasEventFlag->on() && _MinBiasEventFlag->expressionsFromDB(_MinBiasEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    minbiasExpr_   = _MinBiasEventFlag->expressionsFromDB(_MinBiasEventFlag->hltDBKey(),     iSetup);

}

// ***********************************************************
void PFMETAnalyzer::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup, DQMStore * dbe)
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
      if ( _LowMETEventFlag->on() ) 
	makeRatePlot(DirName+"/"+"triggerName_LowMET",totltime);
      if ( _EleEventFlag->on() ) 
	makeRatePlot(DirName+"/"+"triggerName_Ele",totltime);
      if ( _MuonEventFlag->on() ) 
	makeRatePlot(DirName+"/"+"triggerName_Muon",totltime);
    }
}

// ***********************************************************
void PFMETAnalyzer::makeRatePlot(std::string DirName, double totltime)
{

  _dbe->setCurrentFolder(DirName);
  MonitorElement *mePfMET = _dbe->get(DirName+"/"+"METTask_PfMET");

  TH1F* tPfMET;
  TH1F* tPfMETRate;

  if ( mePfMET )
    if ( mePfMET->getRootObject() ) {
      tPfMET     = mePfMET->getTH1F();
      
      // Integral plot & convert number of events to rate (hz)
      tPfMETRate = (TH1F*) tPfMET->Clone("METTask_PfMETRate");
      for (int i = tPfMETRate->GetNbinsX()-1; i>=0; i--){
	tPfMETRate->SetBinContent(i+1,tPfMETRate->GetBinContent(i+2)+tPfMET->GetBinContent(i+1));
      }
      for (int i = 0; i<tPfMETRate->GetNbinsX(); i++){
	tPfMETRate->SetBinContent(i+1,tPfMETRate->GetBinContent(i+1)/double(totltime));
      }

      tPfMETRate->SetName("METTask_PfMETRate");
      tPfMETRate->SetTitle("METTask_PfMETRate");
      mePfMETRate      = _dbe->book1D("METTask_PfMETRate",tPfMETRate);
      
    }
}

// ***********************************************************
void PFMETAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
			    const edm::TriggerResults& triggerResults) {

  if (_verbose) std::cout << "PfMETAnalyzer analyze" << std::endl;

  LogTrace(metname)<<"[PFMETAnalyzer] Analyze PFMET";

  metME->Fill(3);

  // ==========================================================  
  // Trigger information 
  //
  _trig_JetMB=0;
  _trig_HighPtJet=0;
  _trig_LowPtJet=0;
  _trig_MinBias=0;
  _trig_HighMET=0;
  _trig_LowMET=0;
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
        else if (triggerNames.triggerName(i).find(lowMETExpr_[0].substr(0,lowMETExpr_[0].rfind("_v")+2))!=std::string::npos && triggerResults.accept(i))
	  _trig_LowMET=true;
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

    //
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

    edm::LogInfo("PFMetAnalyzer") << "TriggerResults::HLT not found, "
      "automatically select events"; 

    // TriggerResults object not found. Look at all events.    
    _trig_JetMB=1;
  }

  // ==========================================================
  // PfMET information
  
  // **** Get the MET container  
  edm::Handle<reco::PFMETCollection> pfmetcoll;
  iEvent.getByLabel(thePfMETCollectionLabel, pfmetcoll);
  
  if(!pfmetcoll.isValid()) return;

  const PFMETCollection *pfmetcol = pfmetcoll.product();
  const PFMET *pfmet;
  pfmet = &(pfmetcol->front());
    
  LogTrace(metname)<<"[PfMETAnalyzer] Call to the PfMET analyzer";

  // ==========================================================
  //
  edm::Handle<HcalNoiseRBXCollection> HRBXCollection;
  iEvent.getByLabel(HcalNoiseRBXCollectionTag,HRBXCollection);
  if (!HRBXCollection.isValid()) {
    LogDebug("") << "PfMETAnalyzer: Could not find HcalNoiseRBX Collection" << std::endl;
    if (_verbose) std::cout << "PfMETAnalyzer: Could not find HcalNoiseRBX Collection" << std::endl;
  }
  
  edm::Handle<HcalNoiseSummary> HNoiseSummary;
  iEvent.getByLabel(HcalNoiseSummaryTag,HNoiseSummary);
  if (!HNoiseSummary.isValid()) {
    LogDebug("") << "PfMETAnalyzer: Could not find Hcal NoiseSummary product" << std::endl;
    if (_verbose) std::cout << "PfMETAnalyzer: Could not find Hcal NoiseSummary product" << std::endl;
  }

  edm::Handle<reco::CaloJetCollection> caloJets;
  iEvent.getByLabel(theJetCollectionLabel, caloJets);
  if (!caloJets.isValid()) {
    LogDebug("") << "PFMETAnalyzer: Could not find jet product" << std::endl;
    if (_verbose) std::cout << "PFMETAnalyzer: Could not find jet product" << std::endl;
  }

  edm::Handle<edm::View<PFCandidate> > pfCandidates;
  iEvent.getByLabel(PFCandidatesTag, pfCandidates);
  if (!pfCandidates.isValid()) {
    LogDebug("") << "PfMETAnalyzer: Could not find pfcandidates product" << std::endl;
    if (_verbose) std::cout << "PfMETAnalyzer: Could not find pfcandidates product" << std::endl;
  }

  edm::Handle<reco::PFJetCollection> pfJets;
  iEvent.getByLabel(thePfJetCollectionLabel, pfJets);
  if (!pfJets.isValid()) {
    LogDebug("") << "PFMETAnalyzer: Could not find pfjet product" << std::endl;
    if (_verbose) std::cout << "PFMETAnalyzer: Could not find pfjet product" << std::endl;
  }
  // ==========================================================
  // PfMET sanity check

  if (_source=="PfMET") validateMET(*pfmet, pfCandidates);
  
  // ==========================================================
  // JetID 

  if (_verbose) std::cout << "JetID starts" << std::endl;
  
  //
  // --- Minimal cuts
  //
  bool bJetIDMinimal=true;
  for (reco::CaloJetCollection::const_iterator cal = caloJets->begin(); 
       cal!=caloJets->end(); ++cal){
    jetID->calculate(iEvent, *cal);
    if (cal->pt()>10.){
      if (fabs(cal->eta())<=2.6 && 
	  cal->emEnergyFraction()<=0.01) bJetIDMinimal=false;
    }
  }

  //
  // --- Loose cuts, not PF specific for now!
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
  
  bool bHcalNoiseFilter      = HNoiseSummary->passLooseNoiseFilter();
  bool bHcalNoiseFilterTight = HNoiseSummary->passTightNoiseFilter();

  // ==========================================================
  // Get BeamHaloSummary
  edm::Handle<BeamHaloSummary> TheBeamHaloSummary ;
  iEvent.getByLabel(BeamHaloSummaryTag, TheBeamHaloSummary) ;

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

    iEvent.getByLabel(vertexTag, vertexHandle);

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
  
  bool bHcalNoise   = bHcalNoiseFilter;
  bool bBeamHaloID  = bBeamHaloIDLoosePass;
  bool bJetID       = bJetIDMinimal;

  bool bPhysicsDeclared = true;
  if(_doHLTPhysicsOn) bPhysicsDeclared =_trig_PhysDec;

  if      (_tightHcalFiltering)     bHcalNoise  = bHcalNoiseFilterTight;
  if      (_tightBHFiltering)       bBeamHaloID = bBeamHaloIDTightPass;

  if      (_tightJetIDFiltering==1)  bJetID      = bJetIDMinimal;
  else if (_tightJetIDFiltering==2)  bJetID      = bJetIDLoose;
  else if (_tightJetIDFiltering==3)  bJetID      = bJetIDTight;
  else if (_tightJetIDFiltering==-1) bJetID      = true;

  bool bBasicCleanup = bTechTriggers && bPrimaryVertex && bPhysicsDeclared;
  bool bExtraCleanup = bBasicCleanup && bHcalNoise && bJetID && bBeamHaloID;
  
  std::string DirName = "JetMET/MET/"+_source;
  
  for (std::vector<std::string>::const_iterator ic = _FolderNames.begin(); 
       ic != _FolderNames.end(); ic++){
    if (*ic=="All")                                             fillMESet(iEvent, DirName+"/"+*ic, *pfmet);
    if (DCSFilter->filter(iEvent, iSetup)) {
    if (_cleanupSelection){
    if (*ic=="BasicCleanup" && bBasicCleanup)                   fillMESet(iEvent, DirName+"/"+*ic, *pfmet);
    if (*ic=="ExtraCleanup" && bExtraCleanup)                   fillMESet(iEvent, DirName+"/"+*ic, *pfmet);
    }
    if (_allSelection) {
      if (*ic=="HcalNoiseFilter"      && bHcalNoiseFilter )       fillMESet(iEvent, DirName+"/"+*ic, *pfmet);
      if (*ic=="HcalNoiseFilterTight" && bHcalNoiseFilterTight )  fillMESet(iEvent, DirName+"/"+*ic, *pfmet);
      if (*ic=="JetIDMinimal"         && bJetIDMinimal)           fillMESet(iEvent, DirName+"/"+*ic, *pfmet);
      if (*ic=="JetIDLoose"           && bJetIDLoose)             fillMESet(iEvent, DirName+"/"+*ic, *pfmet);
      if (*ic=="JetIDTight"           && bJetIDTight)             fillMESet(iEvent, DirName+"/"+*ic, *pfmet);
      if (*ic=="BeamHaloIDTightPass"  && bBeamHaloIDTightPass)    fillMESet(iEvent, DirName+"/"+*ic, *pfmet);
      if (*ic=="BeamHaloIDLoosePass"  && bBeamHaloIDLoosePass)    fillMESet(iEvent, DirName+"/"+*ic, *pfmet);
      if (*ic=="Triggers"             && bTechTriggers)           fillMESet(iEvent, DirName+"/"+*ic, *pfmet);
      if (*ic=="PV"                   && bPrimaryVertex)          fillMESet(iEvent, DirName+"/"+*ic, *pfmet);
    }
    } // DCS
  }
}

  
// ***********************************************************
void PFMETAnalyzer::validateMET(const reco::PFMET& pfmet, 
				edm::Handle<edm::View<PFCandidate> > pfCandidates)
{          
  double sumEx = 0;
  double sumEy = 0;
  double sumEt = 0;
      
  for( unsigned i=0; i<pfCandidates->size(); i++ ) {
	
    const reco::PFCandidate& cand = (*pfCandidates)[i];
	
    double E = cand.energy();
	
    /// HF calibration factor (in 31X applied by PFProducer)
    // 	if( cand.particleId()==PFCandidate::h_HF || 
    // 	    cand.particleId()==PFCandidate::egamma_HF )
    // 	  E *= hfCalibFactor_;
	
    double phi = cand.phi();
    double cosphi = cos(phi);
    double sinphi = sin(phi);
	
    double theta = cand.theta();
    double sintheta = sin(theta);
	
    double et = E*sintheta;
    double ex = et*cosphi;
    double ey = et*sinphi;
	
    sumEx += ex;
    sumEy += ey;
    sumEt += et;
  }
      
  double Et = sqrt( sumEx*sumEx + sumEy*sumEy);
  XYZTLorentzVector missingEt( -sumEx, -sumEy, 0, Et);
      
  if(_verbose) 
    if (sumEt!=pfmet.sumEt() || sumEx!=pfmet.px() || sumEy!=pfmet.py() || missingEt.T()!=pfmet.pt() )	
      {
	std::cout<<"PFSumEt: " << sumEt         <<", "<<"PFMETBlock: "<<pfmet.pt()<<std::endl;
	std::cout<<"PFMET: "   << missingEt.T() <<", "<<"PFMETBlock: "<<pfmet.pt()<<std::endl;
	std::cout<<"PFMETx: "  << missingEt.X() <<", "<<"PFMETBlockx: "<<pfmet.pt()<<std::endl;
	std::cout<<"PFMETy: "  << missingEt.Y() <<", "<<"PFMETBlocky: "<<pfmet.pt()<<std::endl;
      }
}

// ***********************************************************
void PFMETAnalyzer::fillMESet(const edm::Event& iEvent, std::string DirName, 
			      const reco::PFMET& pfmet)
{

  _dbe->setCurrentFolder(DirName);

  bool bLumiSecPlot=false;
  if (DirName.find("All")) bLumiSecPlot=true;

  if (_trig_JetMB)
    fillMonitorElement(iEvent,DirName,"",pfmet, bLumiSecPlot);
  if (_trig_HighPtJet)
    fillMonitorElement(iEvent,DirName,"HighPtJet",pfmet,false);
  if (_trig_LowPtJet)
    fillMonitorElement(iEvent,DirName,"LowPtJet",pfmet,false);
  if (_trig_MinBias)
    fillMonitorElement(iEvent,DirName,"MinBias",pfmet,false);
  if (_trig_HighMET)
    fillMonitorElement(iEvent,DirName,"HighMET",pfmet,false);
  if (_trig_LowMET)
    fillMonitorElement(iEvent,DirName,"LowMET",pfmet,false);
  if (_trig_Ele)
    fillMonitorElement(iEvent,DirName,"Ele",pfmet,false);
  if (_trig_Muon)
    fillMonitorElement(iEvent,DirName,"Muon",pfmet,false);
}

// ***********************************************************
void PFMETAnalyzer::fillMonitorElement(const edm::Event& iEvent, std::string DirName, 
				       std::string TriggerTypeName, 
				       const reco::PFMET& pfmet, bool bLumiSecPlot)
{

  if (TriggerTypeName=="HighPtJet") {
    if (!selectHighPtJetEvent(iEvent)) return;
  }
  else if (TriggerTypeName=="LowPtJet") {
    if (!selectLowPtJetEvent(iEvent)) return;
  }
  else if (TriggerTypeName=="HighMET") {
    if (pfmet.pt()<_highPFMETThreshold) return;
  }
  else if (TriggerTypeName=="LowMET") {
    if (pfmet.pt()<_lowPFMETThreshold) return;
  }
  else if (TriggerTypeName=="Ele") {
    if (!selectWElectronEvent(iEvent)) return;
  }
  else if (TriggerTypeName=="Muon") {
    if (!selectWMuonEvent(iEvent)) return;
  }
  
  // Reconstructed MET Information
  double pfSumET  = pfmet.sumEt();
  double pfMETSig = pfmet.mEtSig();
  //double pfEz     = pfmet.e_longitudinal();
  double pfMET    = pfmet.pt();
  double pfMEx    = pfmet.px();
  double pfMEy    = pfmet.py();
  double pfMETPhi = pfmet.phi();

  double pfNeutralEMFraction  = pfmet.NeutralEMFraction();
  double pfNeutralHadFraction = pfmet.NeutralHadFraction();
  double pfChargedEMFraction  = pfmet.ChargedEMFraction();
  double pfChargedHadFraction = pfmet.ChargedHadFraction();
  double pfMuonFraction       = pfmet.MuonFraction();

  
  //
  int myLuminosityBlock;
  //  myLuminosityBlock = (evtCounter++)/1000;
  myLuminosityBlock = iEvent.luminosityBlock();
  //

  if (TriggerTypeName!="") DirName = DirName +"/"+TriggerTypeName;

  if (_verbose) std::cout << "_etThreshold = " << _etThreshold << std::endl;
  if (pfSumET>_etThreshold){
    
    mePfMEx    = _dbe->get(DirName+"/"+"METTask_PfMEx");    if (mePfMEx    && mePfMEx->getRootObject())    mePfMEx->Fill(pfMEx);
    mePfMEy    = _dbe->get(DirName+"/"+"METTask_PfMEy");    if (mePfMEy    && mePfMEy->getRootObject())    mePfMEy->Fill(pfMEy);
    mePfMET    = _dbe->get(DirName+"/"+"METTask_PfMET");    if (mePfMET    && mePfMET->getRootObject())    mePfMET->Fill(pfMET);
    mePfMETPhi = _dbe->get(DirName+"/"+"METTask_PfMETPhi"); if (mePfMETPhi && mePfMETPhi->getRootObject()) mePfMETPhi->Fill(pfMETPhi);
    mePfSumET  = _dbe->get(DirName+"/"+"METTask_PfSumET");  if (mePfSumET  && mePfSumET->getRootObject())  mePfSumET->Fill(pfSumET);
    mePfMETSig = _dbe->get(DirName+"/"+"METTask_PfMETSig"); if (mePfMETSig && mePfMETSig->getRootObject()) mePfMETSig->Fill(pfMETSig);
    //mePfEz     = _dbe->get(DirName+"/"+"METTask_PfEz");     if (mePfEz     && mePfEz->getRootObject())     mePfEz->Fill(pfEz);

    mePfMET_logx    = _dbe->get(DirName+"/"+"METTask_PfMET_logx");    if (mePfMET_logx    && mePfMET_logx->getRootObject())    mePfMET_logx->Fill(log10(pfMET));
    mePfSumET_logx  = _dbe->get(DirName+"/"+"METTask_PfSumET_logx");  if (mePfSumET_logx  && mePfSumET_logx->getRootObject())  mePfSumET_logx->Fill(log10(pfSumET));

    //mePfMETIonFeedbck = _dbe->get(DirName+"/"+"METTask_PfMETIonFeedbck");  if (mePfMETIonFeedbck && mePfMETIonFeedbck->getRootObject()) mePfMETIonFeedbck->Fill(pfMET);
    //mePfMETHPDNoise   = _dbe->get(DirName+"/"+"METTask_PfMETHPDNoise");    if (mePfMETHPDNoise   && mePfMETHPDNoise->getRootObject())   mePfMETHPDNoise->Fill(pfMET);
    //mePfMETRBXNoise   = _dbe->get(DirName+"/"+"METTask_PfMETRBXNoise");    if (mePfMETRBXNoise   && mePfMETRBXNoise->getRootObject())   mePfMETRBXNoise->Fill(pfMET);
    
    mePfNeutralEMFraction = _dbe->get(DirName+"/"+"METTask_PfNeutralEMFraction"); 
    if (mePfNeutralEMFraction   && mePfNeutralEMFraction->getRootObject()) mePfNeutralEMFraction->Fill(pfNeutralEMFraction);
    mePfNeutralHadFraction = _dbe->get(DirName+"/"+"METTask_PfNeutralHadFraction"); 
    if (mePfNeutralHadFraction   && mePfNeutralHadFraction->getRootObject()) mePfNeutralHadFraction->Fill(pfNeutralHadFraction);
    mePfChargedEMFraction = _dbe->get(DirName+"/"+"METTask_PfChargedEMFraction"); 
    if (mePfChargedEMFraction   && mePfChargedEMFraction->getRootObject()) mePfChargedEMFraction->Fill(pfChargedEMFraction);
    mePfChargedHadFraction = _dbe->get(DirName+"/"+"METTask_PfChargedHadFraction"); 
    if (mePfChargedHadFraction   && mePfChargedHadFraction->getRootObject()) mePfChargedHadFraction->Fill(pfChargedHadFraction);
    mePfMuonFraction = _dbe->get(DirName+"/"+"METTask_PfMuonFraction"); 
    if (mePfMuonFraction   && mePfMuonFraction->getRootObject()) mePfMuonFraction->Fill(pfMuonFraction);
    
    if (_allhist){
      if (bLumiSecPlot){
	mePfMExLS = _dbe->get(DirName+"/"+"METTask_PfMExLS"); if (mePfMExLS && mePfMExLS->getRootObject()) mePfMExLS->Fill(pfMEx,myLuminosityBlock);
	mePfMEyLS = _dbe->get(DirName+"/"+"METTask_PfMEyLS"); if (mePfMEyLS && mePfMEyLS->getRootObject()) mePfMEyLS->Fill(pfMEy,myLuminosityBlock);
      }
    } // _allhist


    // NPV binned
    //--------------------------------------------------------------------------
    int npvbin = -1;
    if      (_numPV <  5) npvbin = 0;
    else if (_numPV < 10) npvbin = 1;
    else if (_numPV < 15) npvbin = 2;
    else if (_numPV < 25) npvbin = 3;
    else                  npvbin = 4;


    mePfMEx_profile = _dbe->get(Form("%s/METTask_PfMEx_profile", DirName.c_str())); if (mePfMEx_profile && mePfMEx_profile->getRootObject()) mePfMEx_profile->Fill(_numPV, pfMEx);
    

    mePfMEx_npv[npvbin]    = _dbe->get(Form("%s/METTask_PfMEx_npvBin%d", DirName.c_str(), npvbin));    if (mePfMEx_npv[npvbin]    && mePfMEx_npv[npvbin]->getRootObject())    mePfMEx_npv[npvbin]->Fill(pfMEx);
    mePfMEy_npv[npvbin]    = _dbe->get(Form("%s/METTask_PfMEy_npvBin%d", DirName.c_str(), npvbin));    if (mePfMEy_npv[npvbin]    && mePfMEy_npv[npvbin]->getRootObject())    mePfMEy_npv[npvbin]->Fill(pfMEy);
    mePfMET_npv[npvbin]    = _dbe->get(Form("%s/METTask_PfMET_npvBin%d", DirName.c_str(), npvbin));    if (mePfMET_npv[npvbin]    && mePfMET_npv[npvbin]->getRootObject())    mePfMET_npv[npvbin]->Fill(pfMET);
    //mePfMETPhi_npv[npvbin] = _dbe->get(Form("%s/METTask_PfMETPhi_npvBin%d", DirName.c_str(), npvbin)); if (mePfMETPhi_npv[npvbin] && mePfMETPhi_npv[npvbin]->getRootObject()) mePfMETPhi_npv[npvbin]->Fill(pfMETPhi);
    mePfSumET_npv[npvbin]  = _dbe->get(Form("%s/METTask_PfSumET_npvBin%d", DirName.c_str(), npvbin));  if (mePfSumET_npv[npvbin]  && mePfSumET_npv[npvbin]->getRootObject())  mePfSumET_npv[npvbin]->Fill(pfSumET);
    //mePfMETSig_npv[npvbin] = _dbe->get(Form("%s/METTask_PfMETSig_npvBin%d", DirName.c_str(), npvbin)); if (mePfMETSig_npv[npvbin] && mePfMETSig_npv[npvbin]->getRootObject()) mePfMETSig_npv[npvbin]->Fill(pfMETSig);
    //mePfEz_npv[npvbin]     = _dbe->get(Form("%s/METTask_PfEz_npvBin%d", DirName.c_str(), npvbin));     if (mePfEz_npv[npvbin]     && mePfEz_npv[npvbin]->getRootObject())     mePfEz_npv[npvbin]->Fill(pfEz);

    mePfMET_logx_npv[npvbin]    = _dbe->get(Form("%s/METTask_PfMET_logx_npvBin%d", DirName.c_str(), npvbin));    if (mePfMET_logx    && mePfMET_logx_npv[npvbin]->getRootObject())    mePfMET_logx_npv[npvbin]->Fill(log10(pfMET));
    mePfSumET_logx_npv[npvbin]  = _dbe->get(Form("%s/METTask_PfSumET_logx_npvBin%d", DirName.c_str(), npvbin));  if (mePfSumET_logx  && mePfSumET_logx_npv[npvbin]->getRootObject())  mePfSumET_logx_npv[npvbin]->Fill(log10(pfSumET));

    mePfNeutralEMFraction_npv[npvbin] = _dbe->get(Form("%s/METTask_PfNeutralEMFraction_npvBin%d", DirName.c_str(), npvbin)); 
    if (mePfNeutralEMFraction_npv[npvbin]   && mePfNeutralEMFraction_npv[npvbin]->getRootObject()) mePfNeutralEMFraction_npv[npvbin]->Fill(pfNeutralEMFraction);
    mePfNeutralHadFraction_npv[npvbin] = _dbe->get(Form("%s/METTask_PfNeutralHadFraction_npvBin%d", DirName.c_str(), npvbin)); 
    if (mePfNeutralHadFraction_npv[npvbin]   && mePfNeutralHadFraction_npv[npvbin]->getRootObject()) mePfNeutralHadFraction_npv[npvbin]->Fill(pfNeutralHadFraction);
    mePfChargedEMFraction_npv[npvbin] = _dbe->get(Form("%s/METTask_PfChargedEMFraction_npvBin%d", DirName.c_str(), npvbin)); 
    if (mePfChargedEMFraction_npv[npvbin]   && mePfChargedEMFraction_npv[npvbin]->getRootObject()) mePfChargedEMFraction_npv[npvbin]->Fill(pfChargedEMFraction);
    mePfChargedHadFraction_npv[npvbin] = _dbe->get(Form("%s/METTask_PfChargedHadFraction_npvBin%d", DirName.c_str(), npvbin)); 
    if (mePfChargedHadFraction_npv[npvbin]   && mePfChargedHadFraction_npv[npvbin]->getRootObject()) mePfChargedHadFraction_npv[npvbin]->Fill(pfChargedHadFraction);
    mePfMuonFraction_npv[npvbin] = _dbe->get(Form("%s/METTask_PfMuonFraction_npvBin%d", DirName.c_str(), npvbin)); 
    if (mePfMuonFraction_npv[npvbin]   && mePfMuonFraction_npv[npvbin]->getRootObject()) mePfMuonFraction_npv[npvbin]->Fill(pfMuonFraction);
    












  } // et threshold cut
}

// ***********************************************************
bool PFMETAnalyzer::selectHighPtJetEvent(const edm::Event& iEvent){

  bool return_value=false;

  edm::Handle<reco::PFJetCollection> pfJets;
  iEvent.getByLabel(thePfJetCollectionLabel, pfJets);
  if (!pfJets.isValid()) {
    LogDebug("") << "PFMETAnalyzer: Could not find pfjet product" << std::endl;
    if (_verbose) std::cout << "PFMETAnalyzer: Could not find pfjet product" << std::endl;
  }

  for (reco::PFJetCollection::const_iterator pf = pfJets->begin(); 
       pf!=pfJets->end(); ++pf){
    if (pf->pt()>_highPtPFJetThreshold){
      return_value=true;
    }
  }
  
  return return_value;
}

// // ***********************************************************
bool PFMETAnalyzer::selectLowPtJetEvent(const edm::Event& iEvent){

  bool return_value=false;

  edm::Handle<reco::PFJetCollection> pfJets;
  iEvent.getByLabel(thePfJetCollectionLabel, pfJets);
  if (!pfJets.isValid()) {
    LogDebug("") << "PFMETAnalyzer: Could not find jet product" << std::endl;
    if (_verbose) std::cout << "PFMETAnalyzer: Could not find jet product" << std::endl;
  }

  for (reco::PFJetCollection::const_iterator cal = pfJets->begin(); 
       cal!=pfJets->end(); ++cal){
    if (cal->pt()>_lowPtPFJetThreshold){
      return_value=true;
    }
  }

  return return_value;

}

// ***********************************************************
bool PFMETAnalyzer::selectWElectronEvent(const edm::Event& iEvent){

  bool return_value=true;

  /*
    W-electron event selection comes here
  */

  return return_value;

}

// ***********************************************************
bool PFMETAnalyzer::selectWMuonEvent(const edm::Event& iEvent){

  bool return_value=true;

  /*
    W-muon event selection comes here
  */

  return return_value;

}


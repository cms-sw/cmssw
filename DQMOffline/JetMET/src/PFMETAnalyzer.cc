/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/05/20 17:57:01 $
 *  $Revision: 1.51 $
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
PFMETAnalyzer::~PFMETAnalyzer() { 

  delete _HighPtJetEventFlag;
  delete _LowPtJetEventFlag;
  delete _MinBiasEventFlag;
  delete _HighMETEventFlag;
  //  delete _LowMETEventFlag;
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

  _tightBHFiltering    = theCleaningParameters.getParameter<bool>("tightBHFiltering");
  _tightJetIDFiltering = theCleaningParameters.getParameter<int>("tightJetIDFiltering");


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
  BeamHaloSummaryTag          = parameters.getParameter<edm::InputTag>("BeamHaloSummaryLabel");
  HBHENoiseFilterResultTag    = parameters.getParameter<edm::InputTag>("HBHENoiseFilterResultLabel");

  // misc
  _verbose     = parameters.getParameter<int>("verbose");
  _etThreshold = parameters.getParameter<double>("etThreshold"); // MET threshold
  _allhist     = parameters.getParameter<bool>("allHist");       // Full set of monitoring histograms
  _allSelection= parameters.getParameter<bool>("allSelection");  // Plot with all sets of event selection
  _cleanupSelection= parameters.getParameter<bool>("cleanupSelection");  // Plot with all sets of event selection

  _highPtPFJetThreshold = parameters.getParameter<double>("HighPtJetThreshold");
  _lowPtPFJetThreshold  = parameters.getParameter<double>("LowPtJetThreshold");
  _highPFMETThreshold   = parameters.getParameter<double>("HighMETThreshold");

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


//------------------------------------------------------------------------------
// bookMonitorElement
//------------------------------------------------------------------------------
void PFMETAnalyzer::bookMonitorElement(std::string DirName,
				       bool        bLumiSecPlot = false)
{
  _dbe->setCurrentFolder(DirName);


  mePfMEx        = _dbe->book1D("METTask_PfMEx",        "pfmet.px()",           200, -500,  500); 
  mePfMEy        = _dbe->book1D("METTask_PfMEy",        "pfmet.py()",           200, -500,  500); 
  mePfMET        = _dbe->book1D("METTask_PfMET",        "pfmet.pt()",           200,    0, 1000); 
  mePfSumET      = _dbe->book1D("METTask_PfSumET",      "pfmet.sumEt()",        400,    0, 4000); 
  mePfMETSig     = _dbe->book1D("METTask_PfMETSig",     "pfmet.mEtSig()",        51,    0,   51);
  mePfMETPhi     = _dbe->book1D("METTask_PfMETPhi",     "pfmet.phi()",           60, -3.2,  3.2);
  mePfMET_logx   = _dbe->book1D("METTask_PfMET_logx",   "log10(pfmet.pt())",     40,   -1,    7);
  mePfSumET_logx = _dbe->book1D("METTask_PfSumET_logx", "log10(pfmet.sumEt())",  40,   -1,    7);


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


  if (_allhist) {
    if (bLumiSecPlot) {

      mePfMExLS = _dbe->book2D("METTask_PfMEx_LS", "METTask_PfMEx_LS", 200, -200, 200, 50, 0, 500);
      mePfMEyLS = _dbe->book2D("METTask_PfMEy_LS", "METTask_PfMEy_LS", 200, -200, 200, 50, 0, 500);

      mePfMExLS->setAxisTitle("pfmet.px()", 1);
      mePfMEyLS->setAxisTitle("pfmet.px()", 1);

      mePfMExLS->setAxisTitle("event.luminosityBlock()", 2);
      mePfMEyLS->setAxisTitle("event.luminosityBlock()", 2);
    }
  }


  // Book NPV profiles
  //----------------------------------------------------------------------------
  mePfMEx_profile   = _dbe->bookProfile("METTask_PfMEx_profile",   "pfmet.px()",    nbinsPV, PVlow, PVup, 200, -500,  500);
  mePfMEy_profile   = _dbe->bookProfile("METTask_PfMEy_profile",   "pfmet.py()",    nbinsPV, PVlow, PVup, 200, -500,  500); 
  mePfMET_profile   = _dbe->bookProfile("METTask_PfMET_profile",   "pfmet.pt()",    nbinsPV, PVlow, PVup, 200,    0, 1000); 
  mePfSumET_profile = _dbe->bookProfile("METTask_PfSumET_profile", "pfmet.sumEt()", nbinsPV, PVlow, PVup, 400,    0, 4000); 

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


  // Set NPV profiles x-axis title
  //----------------------------------------------------------------------------
  mePfMEx_profile  ->setAxisTitle("nvtx", 1);
  mePfMEy_profile  ->setAxisTitle("nvtx", 1);
  mePfMET_profile  ->setAxisTitle("nvtx", 1);
  mePfSumET_profile->setAxisTitle("nvtx", 1);

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


//------------------------------------------------------------------------------
// beginRun
//------------------------------------------------------------------------------
void PFMETAnalyzer::beginRun(const edm::Run&        iRun,
			     const edm::EventSetup& iSetup)
{
  if (_HighPtJetEventFlag->on()) _HighPtJetEventFlag->initRun(iRun, iSetup);
  if (_LowPtJetEventFlag ->on()) _LowPtJetEventFlag ->initRun(iRun, iSetup);
  if (_MinBiasEventFlag  ->on()) _MinBiasEventFlag  ->initRun(iRun, iSetup);
  if (_HighMETEventFlag  ->on()) _HighMETEventFlag  ->initRun(iRun, iSetup);
  if (_EleEventFlag      ->on()) _EleEventFlag      ->initRun(iRun, iSetup);
  if (_MuonEventFlag     ->on()) _MuonEventFlag     ->initRun(iRun, iSetup);

  if (_HighPtJetEventFlag->on() && _HighPtJetEventFlag->expressionsFromDB(_HighPtJetEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    highPtJetExpr_ = _HighPtJetEventFlag->expressionsFromDB(_HighPtJetEventFlag->hltDBKey(), iSetup);

  if (_LowPtJetEventFlag->on() && _LowPtJetEventFlag->expressionsFromDB(_LowPtJetEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    lowPtJetExpr_ = _LowPtJetEventFlag->expressionsFromDB(_LowPtJetEventFlag->hltDBKey(), iSetup);

  if (_HighMETEventFlag->on() && _HighMETEventFlag->expressionsFromDB(_HighMETEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    highMETExpr_ = _HighMETEventFlag->expressionsFromDB(_HighMETEventFlag->hltDBKey(), iSetup);

  if (_MuonEventFlag->on() && _MuonEventFlag->expressionsFromDB(_MuonEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    muonExpr_ = _MuonEventFlag->expressionsFromDB(_MuonEventFlag->hltDBKey(), iSetup);

  if (_EleEventFlag->on() && _EleEventFlag->expressionsFromDB(_EleEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    elecExpr_ = _EleEventFlag->expressionsFromDB(_EleEventFlag->hltDBKey(), iSetup);

  if (_MinBiasEventFlag->on() && _MinBiasEventFlag->expressionsFromDB(_MinBiasEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    minbiasExpr_ = _MinBiasEventFlag->expressionsFromDB(_MinBiasEventFlag->hltDBKey(), iSetup);
}


//------------------------------------------------------------------------------
// endRun
//------------------------------------------------------------------------------
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

  
  edm::Handle<bool> HBHENoiseFilterResultHandle;
  iEvent.getByLabel(HBHENoiseFilterResultTag, HBHENoiseFilterResultHandle);
  bool HBHENoiseFilterResult = *HBHENoiseFilterResultHandle;
  if (!HBHENoiseFilterResultHandle.isValid()) {
    LogDebug("") << "PFMETAnalyzer: Could not find HBHENoiseFilterResult" << std::endl;
    if (_verbose) std::cout << "PFMETAnalyzer: Could not find HBHENoiseFilterResult" << std::endl;
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
  
  bool bHcalNoiseFilter = HBHENoiseFilterResult;

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
  if (_trig_Ele)
    fillMonitorElement(iEvent,DirName,"Ele",pfmet,false);
  if (_trig_Muon)
    fillMonitorElement(iEvent,DirName,"Muon",pfmet,false);
}


//------------------------------------------------------------------------------
// fillMonitorElement
//------------------------------------------------------------------------------
void PFMETAnalyzer::fillMonitorElement(const edm::Event&  iEvent,
				       std::string        DirName, 
				       std::string        TriggerTypeName, 
				       const reco::PFMET& pfmet,
				       bool               bLumiSecPlot)
{
  if (TriggerTypeName == "HighPtJet") {
    if (!selectHighPtJetEvent(iEvent)) return;
  }
  else if (TriggerTypeName == "LowPtJet") {
    if (!selectLowPtJetEvent(iEvent)) return;
  }
  else if (TriggerTypeName == "HighMET") {
    if (pfmet.pt()<_highPFMETThreshold) return;
  }
  else if (TriggerTypeName == "Ele") {
    if (!selectWElectronEvent(iEvent)) return;
  }
  else if (TriggerTypeName == "Muon") {
    if (!selectWMuonEvent(iEvent)) return;
  }

  if (TriggerTypeName != "") DirName = DirName + "/" + TriggerTypeName;


  // Reconstructed PFMET information
  //----------------------------------------------------------------------------
  double pfSumET  = pfmet.sumEt();
  double pfMETSig = pfmet.mEtSig();
  double pfMET    = pfmet.pt();
  double pfMEx    = pfmet.px();
  double pfMEy    = pfmet.py();
  double pfMETPhi = pfmet.phi();


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


  if (pfSumET > _etThreshold) {
    
    mePfMEx        = _dbe->get(DirName + "/METTask_PfMEx");
    mePfMEy        = _dbe->get(DirName + "/METTask_PfMEy");
    mePfMET        = _dbe->get(DirName + "/METTask_PfMET");
    mePfMETPhi     = _dbe->get(DirName + "/METTask_PfMETPhi");
    mePfSumET      = _dbe->get(DirName + "/METTask_PfSumET");
    mePfMETSig     = _dbe->get(DirName + "/METTask_PfMETSig");
    mePfMET_logx   = _dbe->get(DirName + "/METTask_PfMET_logx");
    mePfSumET_logx = _dbe->get(DirName + "/METTask_PfSumET_logx");

    if (mePfMEx        && mePfMEx       ->getRootObject()) mePfMEx       ->Fill(pfMEx);
    if (mePfMEy        && mePfMEy       ->getRootObject()) mePfMEy       ->Fill(pfMEy);
    if (mePfMET        && mePfMET       ->getRootObject()) mePfMET       ->Fill(pfMET);
    if (mePfMETPhi     && mePfMETPhi    ->getRootObject()) mePfMETPhi    ->Fill(pfMETPhi);
    if (mePfSumET      && mePfSumET     ->getRootObject()) mePfSumET     ->Fill(pfSumET);
    if (mePfMETSig     && mePfMETSig    ->getRootObject()) mePfMETSig    ->Fill(pfMETSig);
    if (mePfMET_logx   && mePfMET_logx  ->getRootObject()) mePfMET_logx  ->Fill(log10(pfMET));
    if (mePfSumET_logx && mePfSumET_logx->getRootObject()) mePfSumET_logx->Fill(log10(pfSumET));


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


    if (_allhist) {
      if (bLumiSecPlot) {

	mePfMExLS = _dbe->get(DirName + "/METTask_PfMExLS");
	mePfMEyLS = _dbe->get(DirName + "/METTask_PfMEyLS");

	if (mePfMExLS && mePfMExLS->getRootObject()) mePfMExLS->Fill(pfMEx, iEvent.luminosityBlock());
	if (mePfMEyLS && mePfMEyLS->getRootObject()) mePfMEyLS->Fill(pfMEy, iEvent.luminosityBlock());
      }
    }


    // Fill NPV profiles
    //--------------------------------------------------------------------------
    mePfMEx_profile   = _dbe->get(DirName + "/METTask_PfMEx_profile");
    mePfMEy_profile   = _dbe->get(DirName + "/METTask_PfMEy_profile");
    mePfMET_profile   = _dbe->get(DirName + "/METTask_PfMET_profile");
    mePfSumET_profile = _dbe->get(DirName + "/METTask_PfSumET_profile");

    if (mePfMEx_profile   && mePfMEx_profile  ->getRootObject()) mePfMEx_profile  ->Fill(_numPV, pfMEx);
    if (mePfMEy_profile   && mePfMEy_profile  ->getRootObject()) mePfMEy_profile  ->Fill(_numPV, pfMEy);
    if (mePfMET_profile   && mePfMET_profile  ->getRootObject()) mePfMET_profile  ->Fill(_numPV, pfMET);
    if (mePfSumET_profile && mePfSumET_profile->getRootObject()) mePfSumET_profile->Fill(_numPV, pfSumET);


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
}


//------------------------------------------------------------------------------
// selectHighPtJetEvent
//------------------------------------------------------------------------------
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


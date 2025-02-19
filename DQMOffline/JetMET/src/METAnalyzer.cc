/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/05/20 13:12:05 $
 *  $Revision: 1.50 $
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
METAnalyzer::~METAnalyzer() { 

  delete _HighPtJetEventFlag;
  delete _LowPtJetEventFlag;
  delete _MinBiasEventFlag;
  delete _HighMETEventFlag;
  //  delete _LowMETEventFlag;
  delete _EleEventFlag;
  delete _MuonEventFlag;

}

void METAnalyzer::beginJob(DQMStore * dbe) {

  evtCounter = 0;
  metname = "METAnalyzer";

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

  // MET information
  theMETCollectionLabel       = parameters.getParameter<edm::InputTag>("METCollectionLabel");
  _source                     = parameters.getParameter<std::string>("Source");

  if (theMETCollectionLabel.label() == "tcMet" ) {
    inputTrackLabel         = parameters.getParameter<edm::InputTag>("InputTrackLabel");    
    inputMuonLabel          = parameters.getParameter<edm::InputTag>("InputMuonLabel");
    inputElectronLabel      = parameters.getParameter<edm::InputTag>("InputElectronLabel");
    inputBeamSpotLabel      = parameters.getParameter<edm::InputTag>("InputBeamSpotLabel");
  }

  // Other data collections
  theJetCollectionLabel       = parameters.getParameter<edm::InputTag>("JetCollectionLabel");
  HcalNoiseRBXCollectionTag   = parameters.getParameter<edm::InputTag>("HcalNoiseRBXCollection");
  BeamHaloSummaryTag          = parameters.getParameter<edm::InputTag>("BeamHaloSummaryLabel");
  HBHENoiseFilterResultTag    = parameters.getParameter<edm::InputTag>("HBHENoiseFilterResultLabel");

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
  jetID = new reco::helper::JetIDHelper(parameters.getParameter<ParameterSet>("JetIDParams"));

  // DQStore stuff
  LogTrace(metname)<<"[METAnalyzer] Parameters initialization";
  std::string DirName = _FolderName+_source;
  dbe->setCurrentFolder(DirName);

  hmetME = dbe->book1D("metReco", "metReco", 4, 1, 5);
  hmetME->setBinLabel(2,"MET",1);

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
}

// ***********************************************************
void METAnalyzer::endJob() {

  delete jetID;
  delete DCSFilter;

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
  hMETSig    ->setAxisTitle("CaloMETSig",       1);
  hMETPhi    ->setAxisTitle("METPhi [rad]",     1);
  hMET_logx  ->setAxisTitle("log(MET) [GeV]",   1);
  hSumET_logx->setAxisTitle("log(SumET) [GeV]", 1);


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

  if (theMETCollectionLabel.label() == "tcMet" ) {
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
  }

  hMExCorrection       = _dbe->book1D("METTask_MExCorrection", "METTask_MExCorrection", 100, -500.0,500.0);
  hMEyCorrection       = _dbe->book1D("METTask_MEyCorrection", "METTask_MEyCorrection", 100, -500.0,500.0);
  hMuonCorrectionFlag  = _dbe->book1D("METTask_CorrectionFlag","METTask_CorrectionFlag", 5, -0.5, 4.5);

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

  std::string dirName = _FolderName+_source+"/";
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
void METAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
			    const edm::TriggerResults& triggerResults) {

  if (_verbose) std::cout << "METAnalyzer analyze" << std::endl;

  std::string DirName = _FolderName+_source;

  LogTrace(metname)<<"[METAnalyzer] Analyze MET";

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

    edm::LogInfo("MetAnalyzer") << "TriggerResults::HLT not found, "
      "automatically select events"; 

    // TriggerResults object not found. Look at all events.    
    _trig_JetMB=1;
  }

  // ==========================================================
  // MET information
  
  // **** Get the MET container  
  edm::Handle<reco::METCollection> metcoll;
  iEvent.getByLabel(theMETCollectionLabel, metcoll);
  
  if(!metcoll.isValid()) {
    std::cout<<"Unable to find MET results for MET collection "<<theMETCollectionLabel<<std::endl;
    return;
  }

  const METCollection *metcol = metcoll.product();
  const MET *met;
  met = &(metcol->front());
    
  LogTrace(metname)<<"[METAnalyzer] Call to the MET analyzer";

  // ==========================================================
  // TCMET 

  if (theMETCollectionLabel.label() == "tcMet" ) {

    iEvent.getByLabel(inputMuonLabel, muon_h);
    iEvent.getByLabel(inputTrackLabel, track_h);
    iEvent.getByLabel(inputElectronLabel, electron_h);
    iEvent.getByLabel(inputBeamSpotLabel, beamSpot_h);
    iEvent.getByLabel("muonTCMETValueMapProducer" , "muCorrData", tcMet_ValueMap_Handle);

    if(!muon_h.isValid())     edm::LogInfo("OutputInfo") << "falied to retrieve muon data require by MET Task";
    if(!track_h.isValid())    edm::LogInfo("OutputInfo") << "falied to retrieve track data require by MET Task";
    if(!electron_h.isValid()) edm::LogInfo("OutputInfo") << "falied to retrieve electron data require by MET Task";
    if(!beamSpot_h.isValid()) edm::LogInfo("OutputInfo") << "falied to retrieve beam spot data require by MET Task";

    bspot = ( beamSpot_h.isValid() ) ? beamSpot_h->position() : math::XYZPoint(0, 0, 0);
    
  }

  // ==========================================================
  //

  edm::Handle<HcalNoiseRBXCollection> HRBXCollection;
  iEvent.getByLabel(HcalNoiseRBXCollectionTag,HRBXCollection);
  if (!HRBXCollection.isValid()) {
    LogDebug("") << "METAnalyzer: Could not find HcalNoiseRBX Collection" << std::endl;
    if (_verbose) std::cout << "METAnalyzer: Could not find HcalNoiseRBX Collection" << std::endl;
  }
  

  edm::Handle<bool> HBHENoiseFilterResultHandle;
  iEvent.getByLabel(HBHENoiseFilterResultTag, HBHENoiseFilterResultHandle);
  bool HBHENoiseFilterResult = *HBHENoiseFilterResultHandle;
  if (!HBHENoiseFilterResultHandle.isValid()) {
    LogDebug("") << "METAnalyzer: Could not find HBHENoiseFilterResult" << std::endl;
    if (_verbose) std::cout << "METAnalyzer: Could not find HBHENoiseFilterResult" << std::endl;
  }


  edm::Handle<reco::CaloJetCollection> caloJets;
  iEvent.getByLabel(theJetCollectionLabel, caloJets);
  if (!caloJets.isValid()) {
    LogDebug("") << "METAnalyzer: Could not find jet product" << std::endl;
    if (_verbose) std::cout << "METAnalyzer: Could not find jet product" << std::endl;
  }

  // ==========================================================
  // MET sanity check

  //   if (_source=="MET") validateMET(*met, tcCandidates);
  
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
  // --- Loose cuts, not  specific for now!
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
	      && fabs(vertex_Z)<_vtxz_max ) 
	  bPrimaryVertex = true;
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
  bool bJetID       = true;

  bool bPhysicsDeclared = true;
  if(_doHLTPhysicsOn) bPhysicsDeclared =_trig_PhysDec;


  if      (_tightBHFiltering)       bBeamHaloID = bBeamHaloIDTightPass;

  if      (_tightJetIDFiltering==1)  bJetID      = bJetIDMinimal;
  else if (_tightJetIDFiltering==2)  bJetID      = bJetIDLoose;
  else if (_tightJetIDFiltering==3)  bJetID      = bJetIDTight;
  else if (_tightJetIDFiltering==-1) bJetID      = true;

  bool bBasicCleanup = bTechTriggers && bPrimaryVertex && bPhysicsDeclared;
  bool bExtraCleanup = bBasicCleanup && bHcalNoise && bJetID && bBeamHaloID;

  //std::string DirName = _FolderName+_source;
  
  for (std::vector<std::string>::const_iterator ic = _FolderNames.begin(); 
       ic != _FolderNames.end(); ic++){
    if (*ic=="All")                                             fillMESet(iEvent, DirName+"/"+*ic, *met);
    if (DCSFilter->filter(iEvent, iSetup)) {
    if (_cleanupSelection){
    if (*ic=="BasicCleanup" && bBasicCleanup)                   fillMESet(iEvent, DirName+"/"+*ic, *met);
    if (*ic=="ExtraCleanup" && bExtraCleanup)                   fillMESet(iEvent, DirName+"/"+*ic, *met);
    }
    if (_allSelection) {
      if (*ic=="HcalNoiseFilter"      && bHcalNoiseFilter )       fillMESet(iEvent, DirName+"/"+*ic, *met);
      if (*ic=="JetIDMinimal"         && bJetIDMinimal)           fillMESet(iEvent, DirName+"/"+*ic, *met);
      if (*ic=="JetIDLoose"           && bJetIDLoose)             fillMESet(iEvent, DirName+"/"+*ic, *met);
      if (*ic=="JetIDTight"           && bJetIDTight)             fillMESet(iEvent, DirName+"/"+*ic, *met);
      if (*ic=="BeamHaloIDTightPass"  && bBeamHaloIDTightPass)    fillMESet(iEvent, DirName+"/"+*ic, *met);
      if (*ic=="BeamHaloIDLoosePass"  && bBeamHaloIDLoosePass)    fillMESet(iEvent, DirName+"/"+*ic, *met);
      if (*ic=="Triggers"             && bTechTriggers)           fillMESet(iEvent, DirName+"/"+*ic, *met);
      if (*ic=="PV"                   && bPrimaryVertex)          fillMESet(iEvent, DirName+"/"+*ic, *met);
    }
    } // DCS
  }
}


// ***********************************************************
void METAnalyzer::fillMESet(const edm::Event& iEvent, std::string DirName, 
			      const reco::MET& met)
{

  _dbe->setCurrentFolder(DirName);

  bool bLumiSecPlot=false;
  if (DirName.find("All")) bLumiSecPlot=true;

  if (_trig_JetMB)
    fillMonitorElement(iEvent,DirName,"",met, bLumiSecPlot);
  if (_trig_HighPtJet)
    fillMonitorElement(iEvent,DirName,"HighPtJet",met,false);
  if (_trig_LowPtJet)
    fillMonitorElement(iEvent,DirName,"LowPtJet",met,false);
  if (_trig_MinBias)
    fillMonitorElement(iEvent,DirName,"MinBias",met,false);
  if (_trig_HighMET)
    fillMonitorElement(iEvent,DirName,"HighMET",met,false);
  //  if (_trig_LowMET)
  //    fillMonitorElement(iEvent,DirName,"LowMET",met,false);
  if (_trig_Ele)
    fillMonitorElement(iEvent,DirName,"Ele",met,false);
  if (_trig_Muon)
    fillMonitorElement(iEvent,DirName,"Muon",met,false);
}

// ***********************************************************
void METAnalyzer::fillMonitorElement(const edm::Event& iEvent, std::string DirName, 
					 std::string TriggerTypeName, 
					 const reco::MET& met, bool bLumiSecPlot)
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
    
    hMEx    = _dbe->get(DirName+"/"+"METTask_MEx");     if (hMEx           && hMEx->getRootObject())     hMEx          ->Fill(MEx);
    hMEy    = _dbe->get(DirName+"/"+"METTask_MEy");     if (hMEy           && hMEy->getRootObject())     hMEy          ->Fill(MEy);
    hMET    = _dbe->get(DirName+"/"+"METTask_MET");     if (hMET           && hMET->getRootObject())     hMET          ->Fill(MET);
    hMETPhi = _dbe->get(DirName+"/"+"METTask_METPhi");  if (hMETPhi        && hMETPhi->getRootObject())  hMETPhi       ->Fill(METPhi);
    hSumET  = _dbe->get(DirName+"/"+"METTask_SumET");   if (hSumET         && hSumET->getRootObject())   hSumET        ->Fill(SumET);
    hMETSig = _dbe->get(DirName+"/"+"METTask_METSig");  if (hMETSig        && hMETSig->getRootObject())  hMETSig       ->Fill(METSig);
    //hEz     = _dbe->get(DirName+"/"+"METTask_Ez");      if (hEz            && hEz->getRootObject())      hEz           ->Fill(Ez);

    hMET_logx   = _dbe->get(DirName+"/"+"METTask_MET_logx");    if (hMET_logx      && hMET_logx->getRootObject())    hMET_logx->Fill(log10(MET));
    hSumET_logx = _dbe->get(DirName+"/"+"METTask_SumET_logx");  if (hSumET_logx    && hSumET_logx->getRootObject())  hSumET_logx->Fill(log10(SumET));

    //hMETIonFeedbck = _dbe->get(DirName+"/"+"METTask_METIonFeedbck");  if (hMETIonFeedbck && hMETIonFeedbck->getRootObject())  hMETIonFeedbck->Fill(MET);
    //hMETHPDNoise   = _dbe->get(DirName+"/"+"METTask_METHPDNoise");    if (hMETHPDNoise   && hMETHPDNoise->getRootObject())    hMETHPDNoise->Fill(MET);
       
    if (_allhist){
      if (bLumiSecPlot){
	hMExLS = _dbe->get(DirName+"/"+"METTask_MExLS"); if (hMExLS  &&  hMExLS->getRootObject())   hMExLS->Fill(MEx,myLuminosityBlock);
	hMEyLS = _dbe->get(DirName+"/"+"METTask_MEyLS"); if (hMEyLS  &&  hMEyLS->getRootObject())   hMEyLS->Fill(MEy,myLuminosityBlock);
      }
    } // _allhist

    ////////////////////////////////////
    if (theMETCollectionLabel.label() == "tcMet" ) {
    
      if(track_h.isValid()) {
	for( edm::View<reco::Track>::const_iterator trkit = track_h->begin(); trkit != track_h->end(); trkit++ ) {
	  htrkPt    = _dbe->get(DirName+"/"+"METTask_trackPt");     if (htrkPt    && htrkPt->getRootObject())     htrkPt->Fill( trkit->pt() );
	  htrkEta   = _dbe->get(DirName+"/"+"METTask_trackEta");    if (htrkEta   && htrkEta->getRootObject())    htrkEta->Fill( trkit->eta() );
	  htrkNhits = _dbe->get(DirName+"/"+"METTask_trackNhits");  if (htrkNhits && htrkNhits->getRootObject())  htrkNhits->Fill( trkit->numberOfValidHits() );
	  htrkChi2  = _dbe->get(DirName+"/"+"METTask_trackNormalizedChi2");   if (htrkChi2  && htrkChi2->getRootObject())   htrkChi2->Fill( trkit->chi2() / trkit->ndof() );
	  double d0 = -1 * trkit->dxy( bspot );
	  htrkD0    = _dbe->get(DirName+"/"+"METTask_trackD0");     if (htrkD0 && htrkD0->getRootObject())        htrkD0->Fill( d0 );
	}
      }
      
      if(electron_h.isValid()) {
	for( edm::View<reco::GsfElectron>::const_iterator eleit = electron_h->begin(); eleit != electron_h->end(); eleit++ ) {
	  helePt  = _dbe->get(DirName+"/"+"METTask_electronPt");   if (helePt  && helePt->getRootObject())   helePt->Fill( eleit->p4().pt() );  
	  heleEta = _dbe->get(DirName+"/"+"METTask_electronEta");  if (heleEta && heleEta->getRootObject())  heleEta->Fill( eleit->p4().eta() );
	  heleHoE = _dbe->get(DirName+"/"+"METTask_electronHoverE");  if (heleHoE && heleHoE->getRootObject())  heleHoE->Fill( eleit->hadronicOverEm() );
	}
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
      }
    }

  } // et threshold cut

}

// ***********************************************************
bool METAnalyzer::selectHighPtJetEvent(const edm::Event& iEvent){

  bool return_value=false;

  edm::Handle<reco::CaloJetCollection> caloJets;
  iEvent.getByLabel(theJetCollectionLabel, caloJets);
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
  
  return return_value;
}

// // ***********************************************************
bool METAnalyzer::selectLowPtJetEvent(const edm::Event& iEvent){

  bool return_value=false;

  edm::Handle<reco::CaloJetCollection> caloJets;
  iEvent.getByLabel(theJetCollectionLabel, caloJets);
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


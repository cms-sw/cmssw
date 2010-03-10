/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/03/04 05:40:43 $
 *  $Revision: 1.34 $
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
using namespace std;
using namespace edm;

// ***********************************************************
CaloMETAnalyzer::CaloMETAnalyzer(const edm::ParameterSet& pSet) {

  parameters = pSet;

}

// ***********************************************************
CaloMETAnalyzer::~CaloMETAnalyzer() { }

// ***********************************************************
void CaloMETAnalyzer::beginJob(DQMStore * dbe) {

  evtCounter = 0;
  metname = "caloMETAnalyzer";

  // trigger information
  HLTPathsJetMBByName_ = parameters.getParameter<std::vector<std::string > >("HLTPathsJetMB");

  _hlt_HighPtJet = parameters.getParameter<std::string>("HLT_HighPtJet");
  _hlt_LowPtJet  = parameters.getParameter<std::string>("HLT_LowPtJet");
  _hlt_HighMET   = parameters.getParameter<std::string>("HLT_HighMET");
  _hlt_LowMET    = parameters.getParameter<std::string>("HLT_LowMET");
  _hlt_Ele       = parameters.getParameter<std::string>("HLT_Ele");
  _hlt_Muon      = parameters.getParameter<std::string>("HLT_Muon");

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

  // Other data collections
  theCaloTowersLabel          = parameters.getParameter<edm::InputTag>("CaloTowersLabel");
  theJetCollectionLabel       = parameters.getParameter<edm::InputTag>("JetCollectionLabel");
  HcalNoiseRBXCollectionTag   = parameters.getParameter<edm::InputTag>("HcalNoiseRBXCollection");
  HcalNoiseSummaryTag         = parameters.getParameter<edm::InputTag>("HcalNoiseSummary");
  BeamHaloSummaryTag          = parameters.getParameter<edm::InputTag>("BeamHaloSummaryLabel");

  // misc
  _verbose     = parameters.getParameter<int>("verbose");
  _print       = parameters.getParameter<int>("printOut");
  _etThreshold = parameters.getParameter<double>("etThreshold"); // MET threshold
  _allhist     = parameters.getParameter<bool>("allHist");       // Full set of monitoring histograms
  _allSelection= parameters.getParameter<bool>("allSelection");  // Plot with all sets of event selection

  _highPtJetThreshold = parameters.getParameter<double>("HighPtJetThreshold"); // High Pt Jet threshold
  _lowPtJetThreshold = parameters.getParameter<double>("LowPtJetThreshold"); // Low Pt Jet threshold
  _highMETThreshold = parameters.getParameter<double>("HighMETThreshold"); // High MET threshold
  _lowMETThreshold = parameters.getParameter<double>("LowMETThreshold"); // Low MET threshold

  //
  jetID = new reco::helper::JetIDHelper(parameters.getParameter<ParameterSet>("JetIDParams"));

  // DQStore stuff
  LogTrace(metname)<<"[CaloMETAnalyzer] Parameters initialization";
  std::string DirName = "JetMET/MET/"+_source;
  dbe->setCurrentFolder(DirName);

  me[DirName+"/metME"] = dbe->book1D("metReco", "metReco", 4, 1, 5);
  me[DirName+"/metME"]->setBinLabel(1,"CaloMET",1);

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
    if (*ic=="BasicCleanup")    bookMESet(DirName+"/"+*ic);
    if (*ic=="ExtraCleanup")    bookMESet(DirName+"/"+*ic);
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
void CaloMETAnalyzer::endJob() {

  delete jetID;

}

// ***********************************************************
void CaloMETAnalyzer::bookMESet(std::string DirName)
{

  bool bLumiSecPlot=false;
  if (DirName.find("All")!=std::string::npos) bLumiSecPlot=true;

  bookMonitorElement(DirName,bLumiSecPlot);

  if (_hlt_HighPtJet.size()){
    bookMonitorElement(DirName+"/"+"HighPtJet",false);
    me[DirName+"/TriggerName_HighPtJet"] = _dbe->bookString("triggerName_HighPtJet", _hlt_HighPtJet);
  }  

  if (_hlt_LowPtJet.size()){
    bookMonitorElement(DirName+"/"+"LowPtJet",false);
    me[DirName+"/TriggerName_LowPtJet"] = _dbe->bookString("triggerName_LowPtJet", _hlt_LowPtJet);
  }

  if (_hlt_HighMET.size()){
    bookMonitorElement(DirName+"/"+"HighMET",false);
    me[DirName+"/TriggerName_HighMET"] = _dbe->bookString("triggerName_HighMET", _hlt_HighMET);
  }

  if (_hlt_LowMET.size()){
    bookMonitorElement(DirName+"/"+"LowMET",false);
    me[DirName+"/TriggerName_LowMET"] = _dbe->bookString("triggerName_LowMET", _hlt_LowMET);
  }

  if (_hlt_Ele.size()){
    bookMonitorElement(DirName+"/"+"Ele",false);
    me[DirName+"/TriggerName_Ele"] = _dbe->bookString("triggerName_Ele", _hlt_Ele);
  }

  if (_hlt_Muon.size()){
    bookMonitorElement(DirName+"/"+"Muon",false);
    me[DirName+"/TriggerName_Muon"] = _dbe->bookString("triggerName_Muon", _hlt_Muon);
  }
}

// ***********************************************************
void CaloMETAnalyzer::bookMonitorElement(std::string DirName, bool bLumiSecPlot=false)
{

  if (_verbose) std::cout << "bookMonitorElement " << DirName << std::endl;
  _dbe->setCurrentFolder(DirName);
 
  me[DirName+"/Nevents"]                = _dbe->book1D("METTask_Nevents",   "METTask_Nevents"   ,1,0,1);
  me[DirName+"/CaloMEx"]                = _dbe->book1D("METTask_CaloMEx",   "METTask_CaloMEx"   ,500,-500,500);
  me[DirName+"/CaloMEx"]->setAxisTitle("MEx [GeV]",1);
  me[DirName+"/CaloMEy"]                = _dbe->book1D("METTask_CaloMEy",   "METTask_CaloMEy"   ,500,-500,500);
  me[DirName+"/CaloMEy"]->setAxisTitle("MEy [GeV]",1);
  me[DirName+"/CaloEz"]                 = _dbe->book1D("METTask_CaloEz",    "METTask_CaloEz"    ,500,-500,500);
  me[DirName+"/CaloEz"]->setAxisTitle("Ez [GeV]",1);
  me[DirName+"/CaloMETSig"]             = _dbe->book1D("METTask_CaloMETSig","METTask_CaloMETSig",51,0,51);
  me[DirName+"/CaloMETSig"]->setAxisTitle("METSig",1);
  me[DirName+"/CaloMET"]                = _dbe->book1D("METTask_CaloMET",   "METTask_CaloMET"   ,500,0,1000);
  me[DirName+"/CaloMET"]->setAxisTitle("MET [GeV]",1);
  //meCaloMET->getTH1F()->SetStats(111111);
  //meCaloMET->getTH1F()->SetOption("logy");
  me[DirName+"/CaloMETPhi"]             = _dbe->book1D("METTask_CaloMETPhi","METTask_CaloMETPhi",80,-TMath::Pi(),TMath::Pi());
  me[DirName+"/CaloMETPhi"]->setAxisTitle("METPhi [rad]",1);
  me[DirName+"/CaloSumET"]              = _dbe->book1D("METTask_CaloSumET", "METTask_CaloSumET" ,500,0,2000);
  me[DirName+"/CaloSumET"]->setAxisTitle("SumET [GeV]",1);

  me[DirName+"/CaloMET_logx"]           = _dbe->book1D("METTask_CaloMET_logx",   "METTask_CaloMET_logx"   ,40,-1.,7.);
  me[DirName+"/CaloMET_logx"]->setAxisTitle("log(MET) [GeV]",1);
  me[DirName+"/CaloSumET_logx"]         = _dbe->book1D("METTask_CaloSumET_logx", "METTask_CaloSumET_logx" ,40,-1.,7.);
  me[DirName+"/CaloSumET_logx"]->setAxisTitle("log(SumET) [GeV]",1);

  me[DirName+"/CaloMETIonFeedbck"]      = _dbe->book1D("METTask_CaloMETIonFeedbck", "METTask_CaloMETIonFeedbck" ,500,0,1000);
  me[DirName+"/CaloMETIonFeedbck"]->setAxisTitle("MET [GeV]",1);
  me[DirName+"/CaloMETHPDNoise"]        = _dbe->book1D("METTask_CaloMETHPDNoise",   "METTask_CaloMETHPDNoise"   ,500,0,1000);
  me[DirName+"/CaloMETHPDNoise"]->setAxisTitle("MET [GeV]",1);
  me[DirName+"/CaloMETRBXNoise"]        = _dbe->book1D("METTask_CaloMETRBXNoise",   "METTask_CaloMETRBXNoise"   ,500,0,1000);
  me[DirName+"/CaloMETRBXNoise"]->setAxisTitle("MET [GeV]",1);

  me[DirName+"/CaloMETPhi002"]          = _dbe->book1D("METTask_CaloMETPhi002","METTask_CaloMETPhi002",72,-TMath::Pi(),TMath::Pi());
  me[DirName+"/CaloMETPhi002"]->setAxisTitle("METPhi [rad] (MET>2 GeV)",1);
  me[DirName+"/CaloMETPhi010"]          = _dbe->book1D("METTask_CaloMETPhi010","METTask_CaloMETPhi010",72,-TMath::Pi(),TMath::Pi());
  me[DirName+"/CaloMETPhi010"]->setAxisTitle("METPhi [rad] (MET>10 GeV)",1);
  me[DirName+"/CaloMETPhi020"]          = _dbe->book1D("METTask_CaloMETPhi020","METTask_CaloMETPhi020",72,-TMath::Pi(),TMath::Pi());
  me[DirName+"/CaloMETPhi020"]->setAxisTitle("METPhi [rad] (MET>20 GeV)",1);

  if (_allhist){
    if (bLumiSecPlot){
      me[DirName+"/CaloMExLS"]              = _dbe->book2D("METTask_CaloMEx_LS","METTask_CaloMEx_LS",200,-200,200,50,0.,500.);
      me[DirName+"/CaloMExLS"]->setAxisTitle("MEx [GeV]",1);
      me[DirName+"/CaloMExLS"]->setAxisTitle("Lumi Section",2);
      me[DirName+"/CaloMEyLS"]              = _dbe->book2D("METTask_CaloMEy_LS","METTask_CaloMEy_LS",200,-200,200,50,0.,500.);
      me[DirName+"/CaloMEyLS"]->setAxisTitle("MEy [GeV]",1);
      me[DirName+"/CaloMEyLS"]->setAxisTitle("Lumi Section",2);
    }

    me[DirName+"/CaloMaxEtInEmTowers"]    = _dbe->book1D("METTask_CaloMaxEtInEmTowers",   "METTask_CaloMaxEtInEmTowers"   ,100,0,2000);
    me[DirName+"/CaloMaxEtInEmTowers"]->setAxisTitle("Et(Max) in EM Tower [GeV]",1);
    me[DirName+"/CaloMaxEtInHadTowers"]   = _dbe->book1D("METTask_CaloMaxEtInHadTowers",  "METTask_CaloMaxEtInHadTowers"  ,100,0,2000);
    me[DirName+"/CaloMaxEtInHadTowers"]->setAxisTitle("Et(Max) in Had Tower [GeV]",1);
    me[DirName+"/CaloEtFractionHadronic"] = _dbe->book1D("METTask_CaloEtFractionHadronic","METTask_CaloEtFractionHadronic",100,0,1);
    me[DirName+"/CaloEtFractionHadronic"]->setAxisTitle("Hadronic Et Fraction",1);
    me[DirName+"/CaloEmEtFraction"]       = _dbe->book1D("METTask_CaloEmEtFraction",      "METTask_CaloEmEtFraction"      ,100,0,1);
    me[DirName+"/CaloEmEtFraction"]->setAxisTitle("EM Et Fraction",1);

    me[DirName+"/CaloEmEtFraction002"]    = _dbe->book1D("METTask_CaloEmEtFraction002",   "METTask_CaloEmEtFraction002"      ,100,0,1);
    me[DirName+"/CaloEmEtFraction002"]->setAxisTitle("EM Et Fraction (MET>2 GeV)",1);
    me[DirName+"/CaloEmEtFraction010"]    = _dbe->book1D("METTask_CaloEmEtFraction010",   "METTask_CaloEmEtFraction010"      ,100,0,1);
    me[DirName+"/CaloEmEtFraction010"]->setAxisTitle("EM Et Fraction (MET>10 GeV)",1);
    me[DirName+"/CaloEmEtFraction020"]    = _dbe->book1D("METTask_CaloEmEtFraction020",   "METTask_CaloEmEtFraction020"      ,100,0,1);
    me[DirName+"/CaloEmEtFraction020"]->setAxisTitle("EM Et Fraction (MET>20 GeV)",1);

    me[DirName+"/CaloHadEtInHB"]          = _dbe->book1D("METTask_CaloHadEtInHB","METTask_CaloHadEtInHB",100,0,2000);
    me[DirName+"/CaloHadEtInHB"]->setAxisTitle("Had Et [GeV]",1);
    me[DirName+"/CaloHadEtInHO"]          = _dbe->book1D("METTask_CaloHadEtInHO","METTask_CaloHadEtInHO",100,0,2000);
    me[DirName+"/CaloHadEtInHO"]->setAxisTitle("Had Et [GeV]",1);
    me[DirName+"/CaloHadEtInHE"]          = _dbe->book1D("METTask_CaloHadEtInHE","METTask_CaloHadEtInHE",100,0,2000);
    me[DirName+"/CaloHadEtInHE"]->setAxisTitle("Had Et [GeV]",1);
    me[DirName+"/CaloHadEtInHF"]          = _dbe->book1D("METTask_CaloHadEtInHF","METTask_CaloHadEtInHF",100,0,2000);
    me[DirName+"/CaloHadEtInHF"]->setAxisTitle("Had Et [GeV]",1);
    me[DirName+"/CaloEmEtInHF"]           = _dbe->book1D("METTask_CaloEmEtInHF" ,"METTask_CaloEmEtInHF" ,100,0,2000);
    me[DirName+"/CaloEmEtInHF"]->setAxisTitle("EM Et [GeV]",1);
    me[DirName+"/CaloEmEtInEE"]           = _dbe->book1D("METTask_CaloEmEtInEE" ,"METTask_CaloEmEtInEE" ,100,0,2000);
    me[DirName+"/CaloEmEtInEE"]->setAxisTitle("EM Et [GeV],1");
    me[DirName+"/CaloEmEtInEB"]           = _dbe->book1D("METTask_CaloEmEtInEB" ,"METTask_CaloEmEtInEB" ,100,0,2000);
    me[DirName+"/CaloEmEtInEB"]->setAxisTitle("EM Et [GeV]",1);

    me[DirName+"/CaloEmMEx"]= _dbe->book1D("METTask_CaloEmMEx","METTask_CaloEmMEx",500,-500,500);
    me[DirName+"/CaloEmMEx"]->setAxisTitle("EM MEx [GeV]",1);
    me[DirName+"/CaloEmMEy"]= _dbe->book1D("METTask_CaloEmMEy","METTask_CaloEmMEy",500,-500,500);
    me[DirName+"/CaloEmMEy"]->setAxisTitle("EM MEy [GeV]",1);
    me[DirName+"/CaloEmEz"]= _dbe->book1D("METTask_CaloEmEz","METTask_CaloEmEz",500,-500,500);
    me[DirName+"/CaloEmEz"]->setAxisTitle("EM Ez [GeV]",1);
    me[DirName+"/CaloEmMET"]= _dbe->book1D("METTask_CaloEmMET","METTask_CaloEmMET",500,0,1000);
    me[DirName+"/CaloEmMET"]->setAxisTitle("EM MET [GeV]",1);
    me[DirName+"/CaloEmMETPhi"]= _dbe->book1D("METTask_CaloEmMETPhi","METTask_CaloEmMETPhi",80,-TMath::Pi(),TMath::Pi());
    me[DirName+"/CaloEmMETPhi"]->setAxisTitle("EM METPhi [rad]",1);
    me[DirName+"/CaloEmSumET"]= _dbe->book1D("METTask_CaloEmSumET","METTask_CaloEmSumET",500,0,2000);
    me[DirName+"/CaloEmSumET"]->setAxisTitle("EM SumET [GeV]",1);

    me[DirName+"/CaloHaMEx"]= _dbe->book1D("METTask_CaloHaMEx","METTask_CaloHaMEx",500,-500,500);
    me[DirName+"/CaloHaMEx"]->setAxisTitle("HA MEx [GeV]",1);
    me[DirName+"/CaloHaMEy"]= _dbe->book1D("METTask_CaloHaMEy","METTask_CaloHaMEy",500,-500,500);
    me[DirName+"/CaloHaMEy"]->setAxisTitle("HA MEy [GeV]",1);
    me[DirName+"/CaloHaEz"]= _dbe->book1D("METTask_CaloHaEz","METTask_CaloHaEz",500,-500,500);
    me[DirName+"/CaloHaEz"]->setAxisTitle("HA Ez [GeV]",1);
    me[DirName+"/CaloHaMET"]= _dbe->book1D("METTask_CaloHaMET","METTask_CaloHaMET",500,0,1000);
    me[DirName+"/CaloHaMET"]->setAxisTitle("HA MET [GeV]",1);
    me[DirName+"/CaloHaMETPhi"]= _dbe->book1D("METTask_CaloHaMETPhi","METTask_CaloHaMETPhi",80,-TMath::Pi(),TMath::Pi());
    me[DirName+"/CaloHaMETPhi"]->setAxisTitle("HA METPhi [rad]",1);
    me[DirName+"/CaloHaSumET"]= _dbe->book1D("METTask_CaloHaSumET","METTask_CaloHaSumET",500,0,2000);
    me[DirName+"/CaloHaSumET"]->setAxisTitle("HA SumET [GeV]",1);

  }

}

// ***********************************************************
void CaloMETAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{

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
      if (_hlt_HighPtJet.size()) makeRatePlot(DirName+"/"+_hlt_HighPtJet,totltime);
      if (_hlt_LowPtJet.size())  makeRatePlot(DirName+"/"+_hlt_LowPtJet,totltime);
      if (_hlt_HighMET.size())   makeRatePlot(DirName+"/"+_hlt_HighMET,totltime);
      if (_hlt_LowMET.size())    makeRatePlot(DirName+"/"+_hlt_LowMET,totltime);
      if (_hlt_Ele.size())       makeRatePlot(DirName+"/"+_hlt_Ele,totltime);
      if (_hlt_Muon.size())      makeRatePlot(DirName+"/"+_hlt_Muon,totltime);
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
      me[DirName+"/CaloMETRate"] = _dbe->book1D("METTask_CaloMETRate",tCaloMETRate);
      me[DirName+"/CaloMETRate"]->setAxisTitle("MET Threshold [GeV]",1);
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

  me[DirName+"/metME"]->Fill(1);

  // ==========================================================  
  // Trigger information 
  //
  _trig_JetMB=0;
  _trig_HighPtJet=0;
  _trig_LowPtJet=0;
  _trig_HighMET=0;
  _trig_LowMET=0;
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
    if (_verbose) std::cout << "triggerNames size" << " " << triggerNames.size() << std::endl;
    if (_verbose) std::cout << _hlt_HighPtJet << " " << triggerNames.triggerIndex(_hlt_HighPtJet) << std::endl;
    if (_verbose) std::cout << _hlt_LowPtJet  << " " << triggerNames.triggerIndex(_hlt_LowPtJet)  << std::endl;
    if (_verbose) std::cout << _hlt_HighMET   << " " << triggerNames.triggerIndex(_hlt_HighMET)   << std::endl;
    if (_verbose) std::cout << _hlt_LowMET    << " " << triggerNames.triggerIndex(_hlt_LowMET)    << std::endl;
    if (_verbose) std::cout << _hlt_Ele       << " " << triggerNames.triggerIndex(_hlt_Ele)       << std::endl;
    if (_verbose) std::cout << _hlt_Muon      << " " << triggerNames.triggerIndex(_hlt_Muon)      << std::endl;
    if (_verbose) std::cout << _hlt_PhysDec   << " " << triggerNames.triggerIndex(_hlt_PhysDec)   << std::endl;

    if (triggerNames.triggerIndex(_hlt_HighPtJet) != triggerNames.size() &&
	triggerResults.accept(triggerNames.triggerIndex(_hlt_HighPtJet))) _trig_HighPtJet=1;

    if (triggerNames.triggerIndex(_hlt_LowPtJet)  != triggerNames.size() &&
	triggerResults.accept(triggerNames.triggerIndex(_hlt_LowPtJet)))  _trig_LowPtJet=1;

    if (triggerNames.triggerIndex(_hlt_HighMET)   != triggerNames.size() &&
        triggerResults.accept(triggerNames.triggerIndex(_hlt_HighMET)))   _trig_HighMET=1;

    if (triggerNames.triggerIndex(_hlt_LowMET)    != triggerNames.size() &&
        triggerResults.accept(triggerNames.triggerIndex(_hlt_LowMET)))    _trig_LowMET=1;

    if (triggerNames.triggerIndex(_hlt_Ele)       != triggerNames.size() &&
        triggerResults.accept(triggerNames.triggerIndex(_hlt_Ele)))       _trig_Ele=1;

    if (triggerNames.triggerIndex(_hlt_Muon)      != triggerNames.size() &&
        triggerResults.accept(triggerNames.triggerIndex(_hlt_Muon)))      _trig_Muon=1;

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

  if(!calometcoll.isValid()) return;

  const CaloMETCollection *calometcol = calometcoll.product();
  const reco::CaloMET *calomet;
  calomet = &(calometcol->front());
  
  LogTrace(metname)<<"[CaloMETAnalyzer] Call to the CaloMET analyzer";

  // ==========================================================
  //
  edm::Handle<HcalNoiseRBXCollection> HRBXCollection;
  iEvent.getByLabel(HcalNoiseRBXCollectionTag,HRBXCollection);
  if (!HRBXCollection.isValid()) {
      LogDebug("") << "CaloMETAnalyzer: Could not find HcalNoiseRBX Collection" << std::endl;
      if (_verbose) std::cout << "CaloMETAnalyzer: Could not find HcalNoiseRBX Collection" << std::endl;
  }
  
  edm::Handle<HcalNoiseSummary> HNoiseSummary;
  iEvent.getByLabel(HcalNoiseSummaryTag,HNoiseSummary);
  if (!HNoiseSummary.isValid()) {
    LogDebug("") << "CaloMETAnalyzer: Could not find Hcal NoiseSummary product" << std::endl;
    if (_verbose) std::cout << "CaloMETAnalyzer: Could not find Hcal NoiseSummary product" << std::endl;
  }
  
  edm::Handle<reco::CaloJetCollection> caloJets;
  iEvent.getByLabel(theJetCollectionLabel, caloJets);
  if (!caloJets.isValid()) {
    LogDebug("") << "CaloMETAnalyzer: Could not find jet product" << std::endl;
    if (_verbose) std::cout << "CaloMETAnalyzer: Could not find jet product" << std::endl;
  }

  edm::Handle<edm::View<Candidate> > towers;
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
  
  bool bHcalNoiseFilter      = HNoiseSummary->passLooseNoiseFilter();
  bool bHcalNoiseFilterTight = HNoiseSummary->passTightNoiseFilter();

  if (_verbose) std::cout << "HcalNoiseFilter Summary ends" << std::endl;

  // ==========================================================
  // Get BeamHaloSummary
  edm::Handle<BeamHaloSummary> TheBeamHaloSummary ;
  iEvent.getByLabel(BeamHaloSummaryTag, TheBeamHaloSummary) ;

  bool bBeamHaloIDTightPass = true;
  bool bBeamHaloIDLoosePass = true;
  
  if(TheBeamHaloSummary.isValid()) {
    
    const BeamHaloSummary TheSummary = (*TheBeamHaloSummary.product() );
    
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
      double vertex_chi2    = v->normalizedChi2();
      //double vertex_d0      = sqrt(v->x()*v->x()+v->y()*v->y());
      //double vertex_numTrks = v->tracksSize();
      double vertex_ndof    = v->ndof();
      bool   fakeVtx        = v->isFake();
      double vertex_sumTrks = 0.0;
      double vertex_Z       = v->z();
      for (Vertex::trackRef_iterator vertex_curTrack = v->tracks_begin(); vertex_curTrack!=v->tracks_end(); vertex_curTrack++) {
	vertex_sumTrks += (*vertex_curTrack)->pt();
      }
      
      if (  !fakeVtx
	    && vertex_number>=_nvtx_min
	    //&& vertex_numTrks>_nvtxtrks_min
	    && vertex_ndof   >_vtxndof_min
	    && vertex_chi2   <_vtxchi2_max
	    && fabs(vertex_Z)<_vtxz_max ) bPrimaryVertex = true;
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
    
    for (unsigned ttr = 0; ttr != _techTrigsAND.size(); ttr++) {
      bTechTriggersAND = bTechTriggersAND && technicalTriggerWordBeforeMask.at(_techTrigsAND.at(ttr));
    }
    
    for (unsigned ttr = 0; ttr != _techTrigsOR.size(); ttr++) {
      bTechTriggersOR = bTechTriggersOR || technicalTriggerWordBeforeMask.at(_techTrigsOR.at(ttr));
    }
    
    for (unsigned ttr = 0; ttr != _techTrigsNOT.size(); ttr++) {
      bTechTriggersNOT = bTechTriggersNOT || technicalTriggerWordBeforeMask.at(_techTrigsNOT.at(ttr));
    }
  }
    
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

  //std::string DirName = "JetMET/MET/"+_source;
  
  for (std::vector<std::string>::const_iterator ic = _FolderNames.begin(); 
       ic != _FolderNames.end(); ic++){
    if (*ic=="All")                                             fillMESet(iEvent, DirName+"/"+*ic, *calomet);
    if (*ic=="BasicCleanup"   && bBasicCleanup)                 fillMESet(iEvent, DirName+"/"+*ic, *calomet);
    if (*ic=="ExtraCleanup"   && bExtraCleanup)                 fillMESet(iEvent, DirName+"/"+*ic, *calomet);
    if (_allSelection) {
      if (*ic=="HcalNoiseFilter"      && bHcalNoiseFilter )       fillMESet(iEvent, DirName+"/"+*ic, *calomet);
      if (*ic=="HcalNoiseFilterTight" && bHcalNoiseFilterTight )  fillMESet(iEvent, DirName+"/"+*ic, *calomet);
      if (*ic=="JetIDMinimal"         && bJetIDMinimal)           fillMESet(iEvent, DirName+"/"+*ic, *calomet);
      if (*ic=="JetIDLoose"           && bJetIDLoose)             fillMESet(iEvent, DirName+"/"+*ic, *calomet);
      if (*ic=="JetIDTight"           && bJetIDTight)             fillMESet(iEvent, DirName+"/"+*ic, *calomet);
      if (*ic=="BeamHaloIDTightPass"  && bBeamHaloIDTightPass)    fillMESet(iEvent, DirName+"/"+*ic, *calomet);
      if (*ic=="BeamHaloIDLoosePass"  && bBeamHaloIDLoosePass)    fillMESet(iEvent, DirName+"/"+*ic, *calomet);
      if (*ic=="Triggers"             && bTechTriggers)           fillMESet(iEvent, DirName+"/"+*ic, *calomet);
      if (*ic=="PV"                   && bPrimaryVertex)          fillMESet(iEvent, DirName+"/"+*ic, *calomet);
    }
  }
}

// ***********************************************************
void CaloMETAnalyzer::computeEmHaMET(edm::Handle<edm::View<Candidate> > towers)
{

  edm::View<Candidate>::const_iterator towerCand = towers->begin();
  
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
      const Candidate* candidate = &(*towerCand);
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
  _HaMetPhi   = atan2( _EmMEy, _EmMEx ); 
  //
  _HaMEx = -sum_ha_ex;
  _HaMEy = -sum_ha_ex;
  _HaMET = pow(_HaMEx*_HaMEx+_HaMEy*_HaMEy,0.5);
  _HaCaloEz = sum_ha_ez;
  _HaSumEt  = sum_ha_et;
  _HaMetPhi   = atan2( _HaMEy, _HaMEx ); 
  
}
// ***********************************************************
void CaloMETAnalyzer::validateMET(const reco::CaloMET& calomet, 
				  edm::Handle<edm::View<Candidate> > towers)
{

  edm::View<Candidate>::const_iterator towerCand = towers->begin();
  
  double sum_et = 0.0;
  double sum_ex = 0.0;
  double sum_ey = 0.0;
  double sum_ez = 0.0;
  
  for ( ; towerCand != towers->end(); towerCand++)
    {
      const Candidate* candidate = &(*towerCand);
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
				const reco::CaloMET& calomet)
{

  _dbe->setCurrentFolder(DirName);

  bool bLumiSecPlot=false;
  if (DirName.find("All")) bLumiSecPlot=true;

  if (_trig_JetMB) fillMonitorElement(iEvent,DirName,"",calomet, bLumiSecPlot);
  if (_hlt_HighPtJet.size() && _trig_HighPtJet) fillMonitorElement(iEvent,DirName,"HighPtJet",calomet,false);
  if (_hlt_LowPtJet.size() && _trig_LowPtJet) fillMonitorElement(iEvent,DirName,"LowPtJet",calomet,false);
  if (_hlt_HighMET.size() && _trig_HighMET) fillMonitorElement(iEvent,DirName,"HighMET",calomet,false);
  if (_hlt_LowMET.size() && _trig_LowMET) fillMonitorElement(iEvent,DirName,"LowMET",calomet,false);
  if (_hlt_Ele.size() && _trig_Ele) fillMonitorElement(iEvent,DirName,"Ele",calomet,false);
  if (_hlt_Muon.size() && _trig_Muon) fillMonitorElement(iEvent,DirName,"Muon",calomet,false);

}

// ***********************************************************
void CaloMETAnalyzer::fillMonitorElement(const edm::Event& iEvent, std::string DirName, 
					 std::string TriggerTypeName, 
					 const reco::CaloMET& calomet, bool bLumiSecPlot)
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
  else if (TriggerTypeName=="LowMET") {
    if (calomet.pt()<_lowMETThreshold) return;
  }
  else if (TriggerTypeName=="Ele") {
    if (!selectWElectronEvent(iEvent)) return;
  }
  else if (TriggerTypeName=="Muon") {
    if (!selectWMuonEvent(iEvent)) return;
  }

  double caloSumET  = calomet.sumEt();
  double caloMETSig = calomet.mEtSig();
  double caloEz     = calomet.e_longitudinal();
  double caloMET    = calomet.pt();
  double caloMEx    = calomet.px();
  double caloMEy    = calomet.py();
  double caloMETPhi = calomet.phi();

  if (_verbose) std::cout << _source << " " << caloMET << std::endl;

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

  //
  int myLuminosityBlock;
  //  myLuminosityBlock = (evtCounter++)/1000;
  myLuminosityBlock = iEvent.luminosityBlock();
  //

  if (TriggerTypeName!="") DirName = DirName +"/"+TriggerTypeName;

  if (_verbose) std::cout << "_etThreshold = " << _etThreshold << std::endl;
  if (caloSumET>_etThreshold){

    me[DirName+"/CaloMEx"]->Fill(caloMEx);
    me[DirName+"/CaloMEy"]->Fill(caloMEy);
    me[DirName+"/CaloMET"]->Fill(caloMET);
    me[DirName+"/CaloMETPhi"]->Fill(caloMETPhi);
    me[DirName+"/CaloSumET"]->Fill(caloSumET);
    me[DirName+"/CaloMETSig"]->Fill(caloMETSig);
    me[DirName+"/CaloEz"]->Fill(caloEz);

    me[DirName+"/CaloMET_logx"]->Fill(log10(caloMET));
    me[DirName+"/CaloSumET_logx"]->Fill(log10(caloSumET));

    me[DirName+"/CaloMETIonFeedbck"]->Fill(caloMET);
    me[DirName+"/CaloMETHPDNoise"]->Fill(caloMET);

    if (caloMET>  2.){ me[DirName+"/CaloMETPhi002"]->Fill(caloMETPhi);}
    if (caloMET> 10.){ me[DirName+"/CaloMETPhi010"]->Fill(caloMETPhi);}
    if (caloMET> 20.){ me[DirName+"/CaloMETPhi020"]->Fill(caloMETPhi);}

    if (_allhist){
      if (bLumiSecPlot){
	if ( (_dbe->get(DirName+"/"+"METTask_CaloMExLS")) &&
	     ((_dbe->get(DirName+"/"+"METTask_CaloMExLS"))->getRootObject()))
	     me[DirName+"/CaloMExLS"]->Fill(caloMEx,myLuminosityBlock);
	if ( (_dbe->get(DirName+"/"+"METTask_CaloMEyLS")) &&
	     ((_dbe->get(DirName+"/"+"METTask_CaloMEyLS"))->getRootObject()))
	     me[DirName+"/CaloMEyLS"]->Fill(caloMEy,myLuminosityBlock);
      }
      
      me[DirName+"/CaloEtFractionHadronic"]->Fill(caloEtFractionHadronic);
      me[DirName+"/CaloEmEtFraction"]->Fill(caloEmEtFraction);

      if (caloMET>  2.){ me[DirName+"/CaloEmEtFraction002"]->Fill(caloEmEtFraction);}
      if (caloMET> 10.){ me[DirName+"/CaloEmEtFraction010"]->Fill(caloEmEtFraction);}
      if (caloMET> 20.){ me[DirName+"/CaloEmEtFraction020"]->Fill(caloEmEtFraction);}

      me[DirName+"/CaloMaxEtInEmTowers"]->Fill(caloMaxEtInEMTowers);
      me[DirName+"/CaloMaxEtInHadTowers"]->Fill(caloMaxEtInHadTowers);

      me[DirName+"/CaloHadEtInHB"]->Fill(caloHadEtInHB);
      me[DirName+"/CaloHadEtInHO"]->Fill(caloHadEtInHO);
      me[DirName+"/CaloHadEtInHE"]->Fill(caloHadEtInHE);
      me[DirName+"/CaloHadEtInHF"]->Fill(caloHadEtInHF);
      me[DirName+"/CaloEmEtInEB"]->Fill(caloEmEtInEB);
      me[DirName+"/CaloEmEtInEE"]->Fill(caloEmEtInEE);
      me[DirName+"/CaloEmEtInHF"]->Fill(caloEmEtInHF);

      me[DirName+"/CaloEmMEx"]->Fill(_EmMEx);
      me[DirName+"/CaloEmMEy"]->Fill(_EmMEy);
      me[DirName+"/CaloEmEz"]->Fill(_EmCaloEz);
      me[DirName+"/CaloEmMET"]->Fill(_EmMET);
      me[DirName+"/CaloEmMETPhi"]->Fill(_EmMetPhi);
      me[DirName+"/CaloEmSumET"]->Fill(_EmSumEt);

      me[DirName+"/CaloHaMEx"]->Fill(_HaMEx);
      me[DirName+"/CaloHaMEy"]->Fill(_HaMEy);
      me[DirName+"/CaloHaEz"]->Fill(_HaCaloEz);
      me[DirName+"/CaloHaMET"]->Fill(_HaMET);
      me[DirName+"/CaloHaMETPhi"]->Fill(_HaMetPhi);
      me[DirName+"/CaloHaSumET"]->Fill(_HaSumEt);

    } // _allhist

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

  bool return_value=false;

  /*
    W-electron event selection comes here
   */

  return return_value;

}

// ***********************************************************
bool CaloMETAnalyzer::selectWMuonEvent(const edm::Event& iEvent){

  bool return_value=false;

  /*
    W-muon event selection comes here
   */

  return return_value;

}


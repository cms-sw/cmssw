/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/05/20 13:12:05 $
 *  $Revision: 1.13 $
 *  \author A.Apresyan - Caltech
 */

#include "DQMOffline/JetMET/interface/TcMETAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include <string>
using namespace edm;
using namespace reco;
using namespace math;

// ***********************************************************
TcMETAnalyzer::TcMETAnalyzer(const edm::ParameterSet& pSet) {

  parameters = pSet;

}

// ***********************************************************
TcMETAnalyzer::~TcMETAnalyzer() { }

void TcMETAnalyzer::beginJob(DQMStore * dbe) {

  evtCounter = 0;
  metname = "tcMETAnalyzer";

  // trigger information
  HLTPathsJetMBByName_ = parameters.getParameter<std::vector<std::string > >("HLTPathsJetMB");

  _hlt_HighPtJet = parameters.getParameter<std::string>("HLT_HighPtJet");
  _hlt_LowPtJet  = parameters.getParameter<std::string>("HLT_LowPtJet");
  _hlt_HighMET   = parameters.getParameter<std::string>("HLT_HighMET");
  //  _hlt_LowMET    = parameters.getParameter<std::string>("HLT_LowMET");
  _hlt_Ele       = parameters.getParameter<std::string>("HLT_Ele");
  _hlt_Muon      = parameters.getParameter<std::string>("HLT_Muon");

  // TcMET information
  theTcMETCollectionLabel       = parameters.getParameter<edm::InputTag>("TcMETCollectionLabel");
  _source                       = parameters.getParameter<std::string>("Source");

  // Other data collections
  HcalNoiseRBXCollectionTag   = parameters.getParameter<edm::InputTag>("HcalNoiseRBXCollection");
  theJetCollectionLabel       = parameters.getParameter<edm::InputTag>("JetCollectionLabel");
  HBHENoiseFilterResultTag    = parameters.getParameter<edm::InputTag>("HBHENoiseFilterResultLabel");

  // misc
  _verbose     = parameters.getParameter<int>("verbose");
  _etThreshold = parameters.getParameter<double>("etThreshold"); // MET threshold
  _allhist     = parameters.getParameter<bool>("allHist");       // Full set of monitoring histograms
  _allSelection= parameters.getParameter<bool>("allSelection");  // Plot with all sets of event selection

  _highPtTcJetThreshold = parameters.getParameter<double>("HighPtTcJetThreshold"); // High Pt Jet threshold
  _lowPtTcJetThreshold = parameters.getParameter<double>("LowPtTcJetThreshold");   // Low Pt Jet threshold
  _highTcMETThreshold = parameters.getParameter<double>("HighTcMETThreshold");     // High MET threshold
  _lowTcMETThreshold = parameters.getParameter<double>("LowTcMETThreshold");       // Low MET threshold

  //
  jetID = new reco::helper::JetIDHelper(parameters.getParameter<ParameterSet>("JetIDParams"));

  // DQStore stuff
  LogTrace(metname)<<"[TcMETAnalyzer] Parameters initialization";
  std::string DirName = "JetMET/MET/"+_source;
  dbe->setCurrentFolder(DirName);

  metME = dbe->book1D("metReco", "metReco", 4, 1, 5);
  metME->setBinLabel(2,"TcMET",1);

  _dbe = dbe;

  _FolderNames.push_back("All");
  _FolderNames.push_back("Cleanup");
  _FolderNames.push_back("HcalNoiseFilter");
  _FolderNames.push_back("JetID");
  _FolderNames.push_back("JetIDTight");

  for (std::vector<std::string>::const_iterator ic = _FolderNames.begin(); 
       ic != _FolderNames.end(); ic++){
    if (*ic=="All")                  bookMESet(DirName+"/"+*ic);
    if (*ic=="Cleanup")              bookMESet(DirName+"/"+*ic);
    if (_allSelection){
    if (*ic=="HcalNoiseFilter")      bookMESet(DirName+"/"+*ic);
    if (*ic=="JetID")                bookMESet(DirName+"/"+*ic);
    if (*ic=="JetIDTight")           bookMESet(DirName+"/"+*ic);
    }
  }
}

// ***********************************************************
void TcMETAnalyzer::endJob() {

  delete jetID;

}

// ***********************************************************
void TcMETAnalyzer::bookMESet(std::string DirName)
{

  bool bLumiSecPlot=false;
  if (DirName.find("All")!=std::string::npos) bLumiSecPlot=true;

  bookMonitorElement(DirName,bLumiSecPlot);

  if (_hlt_HighPtJet.size()){
    bookMonitorElement(DirName+"/"+"HighPtJet",false);
    meTriggerName_HighPtJet = _dbe->bookString("triggerName_HighPtJet", _hlt_HighPtJet);
  }  

  if (_hlt_LowPtJet.size()){
    bookMonitorElement(DirName+"/"+"LowPtJet",false);
    meTriggerName_LowPtJet = _dbe->bookString("triggerName_LowPtJet", _hlt_LowPtJet);
  }

  if (_hlt_HighMET.size()){
    bookMonitorElement(DirName+"/"+"HighMET",false);
    meTriggerName_HighMET = _dbe->bookString("triggerName_HighMET", _hlt_HighMET);
  }

  //  if (_hlt_LowMET.size()){
  //    bookMonitorElement(DirName+"/"+"LowMET",false);
  //    meTriggerName_LowMET = _dbe->bookString("triggerName_LowMET", _hlt_LowMET);
  //  }

  if (_hlt_Ele.size()){
    bookMonitorElement(DirName+"/"+"Ele",false);
    meTriggerName_Ele = _dbe->bookString("triggerName_Ele", _hlt_Ele);
  }

  if (_hlt_Muon.size()){
    bookMonitorElement(DirName+"/"+"Muon",false);
    meTriggerName_Muon = _dbe->bookString("triggerName_Muon", _hlt_Muon);
  }

}

// ***********************************************************
void TcMETAnalyzer::bookMonitorElement(std::string DirName, bool bLumiSecPlot=false)
{

  if (_verbose) std::cout << "booMonitorElement " << DirName << std::endl;
  _dbe->setCurrentFolder(DirName);
 
  meTcMEx    = _dbe->book1D("METTask_TcMEx",    "METTask_TcMEx",    200, -500,  500);
  meTcMEy    = _dbe->book1D("METTask_TcMEy",    "METTask_TcMEy",    200, -500,  500);
  meTcEz     = _dbe->book1D("METTask_TcEz",     "METTask_TcEz",     200, -500,  500);
  meTcMETSig = _dbe->book1D("METTask_TcMETSig", "METTask_TcMETSig",  51,    0,   51);
  meTcMET    = _dbe->book1D("METTask_TcMET",    "METTask_TcMET",    200,    0, 1000);
  meTcMETPhi = _dbe->book1D("METTask_TcMETPhi", "METTask_TcMETPhi",  60, -3.2,  3.2);
  meTcSumET  = _dbe->book1D("METTask_TcSumET",  "METTask_TcSumET",  400,    0, 4000);

  meTcNeutralEMFraction  = _dbe->book1D("METTask_TcNeutralEMFraction", "METTask_TcNeutralEMFraction" ,50,0.,1.);
  meTcNeutralHadFraction = _dbe->book1D("METTask_TcNeutralHadFraction","METTask_TcNeutralHadFraction",50,0.,1.);
  meTcChargedEMFraction  = _dbe->book1D("METTask_TcChargedEMFraction", "METTask_TcChargedEMFraction" ,50,0.,1.);
  meTcChargedHadFraction = _dbe->book1D("METTask_TcChargedHadFraction","METTask_TcChargedHadFraction",50,0.,1.);
  meTcMuonFraction       = _dbe->book1D("METTask_TcMuonFraction",      "METTask_TcMuonFraction"      ,50,0.,1.);

  meTcMETIonFeedbck      = _dbe->book1D("METTask_TcMETIonFeedbck", "METTask_TcMETIonFeedbck" ,500,0,1000);
  meTcMETHPDNoise        = _dbe->book1D("METTask_TcMETHPDNoise",   "METTask_TcMETHPDNoise"   ,500,0,1000);
  meTcMETRBXNoise        = _dbe->book1D("METTask_TcMETRBXNoise",   "METTask_TcMETRBXNoise"   ,500,0,1000);

  if (_allhist){
    if (bLumiSecPlot){
      meTcMExLS              = _dbe->book2D("METTask_TcMEx_LS","METTask_TcMEx_LS",200,-200,200,50,0.,500.);
      meTcMEyLS              = _dbe->book2D("METTask_TcMEy_LS","METTask_TcMEy_LS",200,-200,200,50,0.,500.);
    }
  }
}

// ***********************************************************
void TcMETAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{

}

// ***********************************************************
void TcMETAnalyzer::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup, DQMStore * dbe)
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
      //      if (_hlt_LowMET.size())    makeRatePlot(DirName+"/"+_hlt_LowMET,totltime);
      if (_hlt_Ele.size())       makeRatePlot(DirName+"/"+_hlt_Ele,totltime);
      if (_hlt_Muon.size())      makeRatePlot(DirName+"/"+_hlt_Muon,totltime);

    }
}


// ***********************************************************
void TcMETAnalyzer::makeRatePlot(std::string DirName, double totltime)
{

  _dbe->setCurrentFolder(DirName);
  MonitorElement *meTcMET = _dbe->get(DirName+"/"+"METTask_TcMET");

  TH1F* tTcMET;
  TH1F* tTcMETRate;

  if ( meTcMET )
    if ( meTcMET->getRootObject() ) {
      tTcMET     = meTcMET->getTH1F();
      
      // Integral plot & convert number of events to rate (hz)
      tTcMETRate = (TH1F*) tTcMET->Clone("METTask_TcMETRate");
      for (int i = tTcMETRate->GetNbinsX()-1; i>=0; i--){
	tTcMETRate->SetBinContent(i+1,tTcMETRate->GetBinContent(i+2)+tTcMET->GetBinContent(i+1));
      }
      for (int i = 0; i<tTcMETRate->GetNbinsX(); i++){
	tTcMETRate->SetBinContent(i+1,tTcMETRate->GetBinContent(i+1)/double(totltime));
      }      

      meTcMETRate      = _dbe->book1D("METTask_TcMETRate",tTcMETRate);
      
    }

}

// ***********************************************************
void TcMETAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
			    const edm::TriggerResults& triggerResults) {

  if (_verbose) std::cout << "TcMETAnalyzer analyze" << std::endl;

  LogTrace(metname)<<"[TcMETAnalyzer] Analyze TcMET";

  metME->Fill(2);

  // ==========================================================  
  // Trigger information 
  //
  _trig_JetMB=0;
  _trig_HighPtJet=0;
  _trig_LowPtJet=0;
  _trig_HighMET=0;
  //  _trig_LowMET=0;

  if (&triggerResults) {   

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
    //    if (_verbose) std::cout << _hlt_LowMET    << " " << triggerNames.triggerIndex(_hlt_LowMET)    << std::endl;
    if (_verbose) std::cout << _hlt_Ele       << " " << triggerNames.triggerIndex(_hlt_Ele)       << std::endl;
    if (_verbose) std::cout << _hlt_Muon      << " " << triggerNames.triggerIndex(_hlt_Muon)      << std::endl;

    if (triggerNames.triggerIndex(_hlt_HighPtJet) != triggerNames.size() &&
	triggerResults.accept(triggerNames.triggerIndex(_hlt_HighPtJet))) _trig_HighPtJet=1;

    if (triggerNames.triggerIndex(_hlt_LowPtJet)  != triggerNames.size() &&
	triggerResults.accept(triggerNames.triggerIndex(_hlt_LowPtJet)))  _trig_LowPtJet=1;

    if (triggerNames.triggerIndex(_hlt_HighMET)   != triggerNames.size() &&
        triggerResults.accept(triggerNames.triggerIndex(_hlt_HighMET)))   _trig_HighMET=1;

    //    if (triggerNames.triggerIndex(_hlt_LowMET)    != triggerNames.size() &&
    //        triggerResults.accept(triggerNames.triggerIndex(_hlt_LowMET)))    _trig_LowMET=1;

    if (triggerNames.triggerIndex(_hlt_Ele)       != triggerNames.size() &&
        triggerResults.accept(triggerNames.triggerIndex(_hlt_Ele)))       _trig_Ele=1;

    if (triggerNames.triggerIndex(_hlt_Muon)      != triggerNames.size() &&
        triggerResults.accept(triggerNames.triggerIndex(_hlt_Muon)))      _trig_Muon=1;
    
  } else {

    edm::LogInfo("TcMetAnalyzer") << "TriggerResults::HLT not found, "
      "automatically select events"; 

    // TriggerResults object not found. Look at all events.    
    _trig_JetMB=1;
  }

  // ==========================================================
  // TcMET information
  
  // **** Get the MET container  
  edm::Handle<reco::METCollection> tcmetcoll;
  iEvent.getByLabel(theTcMETCollectionLabel, tcmetcoll);
  
  if(!tcmetcoll.isValid()) return;

  const METCollection *tcmetcol = tcmetcoll.product();
  const MET *tcmet;
  tcmet = &(tcmetcol->front());
    
  LogTrace(metname)<<"[TcMETAnalyzer] Call to the TcMET analyzer";

  // ==========================================================
  //
  edm::Handle<HcalNoiseRBXCollection> HRBXCollection;
  iEvent.getByLabel(HcalNoiseRBXCollectionTag,HRBXCollection);
  if (!HRBXCollection.isValid()) {
    LogDebug("") << "TcMETAnalyzer: Could not find HcalNoiseRBX Collection" << std::endl;
    if (_verbose) std::cout << "TcMETAnalyzer: Could not find HcalNoiseRBX Collection" << std::endl;
  }


  edm::Handle<bool> HBHENoiseFilterResultHandle;
  iEvent.getByLabel(HBHENoiseFilterResultTag, HBHENoiseFilterResultHandle);
  bool HBHENoiseFilterResult = *HBHENoiseFilterResultHandle;
  if (!HBHENoiseFilterResultHandle.isValid()) {
    LogDebug("") << "TcMETAnalyzer: Could not find HBHENoiseFilterResult" << std::endl;
    if (_verbose) std::cout << "TcMETAnalyzer: Could not find HBHENoiseFilterResult" << std::endl;
  }


  edm::Handle<reco::CaloJetCollection> caloJets;
  iEvent.getByLabel(theJetCollectionLabel, caloJets);
  if (!caloJets.isValid()) {
    LogDebug("") << "TcMETAnalyzer: Could not find jet product" << std::endl;
    if (_verbose) std::cout << "TcMETAnalyzer: Could not find jet product" << std::endl;
  }

  // ==========================================================
  // TcMET sanity check

  //   if (_source=="TcMET") validateMET(*tcmet, tcCandidates);
  
  // ==========================================================
  // JetID 

  if (_verbose) std::cout << "JetID starts" << std::endl;
  
  //
  // --- Loose cuts, not Tc specific for now!
  //
  bool bJetID=true;
  for (reco::CaloJetCollection::const_iterator cal = caloJets->begin(); 
       cal!=caloJets->end(); ++cal){ 
    jetID->calculate(iEvent, *cal);
    if (_verbose) std::cout << jetID->n90Hits() << " " 
			    << jetID->restrictedEMF() << " "
			    << cal->pt() << std::endl;
    if (cal->pt()>10.){
      //
      // for all regions
      if (jetID->n90Hits()<2)  bJetID=false; 
      if (jetID->fHPD()>=0.98) bJetID=false; 
      //if (jetID->restrictedEMF()<0.01) bJetID=false; 
      //
      // for non-forward
      if (fabs(cal->eta())<2.55){
	if (cal->emEnergyFraction()<=0.01) bJetID=false; 
      }
      // for forward
      else {
	if (cal->emEnergyFraction()<=-0.9) bJetID=false; 
	if (cal->pt()>80.){
	if (cal->emEnergyFraction()>= 1.0) bJetID=false; 
	}
      } // forward vs non-forward
    }   // pt>10 GeV/c
  }     // calor-jets loop
 
  //
  // --- Tight cuts
  //
  bool bJetIDTight=true;
  bJetIDTight=bJetID;
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
  // Reconstructed MET Information - fill MonitorElements
  
  std::string DirName = "JetMET/MET/"+_source;
  
  for (std::vector<std::string>::const_iterator ic = _FolderNames.begin(); 
       ic != _FolderNames.end(); ic++){
    if (*ic=="All")                                   fillMESet(iEvent, DirName+"/"+*ic, *tcmet);
    if (*ic=="Cleanup" && bHcalNoiseFilter && bJetID) fillMESet(iEvent, DirName+"/"+*ic, *tcmet);
    if (_allSelection) {
    if (*ic=="HcalNoiseFilter"      && bHcalNoiseFilter )       fillMESet(iEvent, DirName+"/"+*ic, *tcmet);
    if (*ic=="JetID"      && bJetID)                            fillMESet(iEvent, DirName+"/"+*ic, *tcmet);
    if (*ic=="JetIDTight" && bJetIDTight)                       fillMESet(iEvent, DirName+"/"+*ic, *tcmet);
    }
  }
}

// ***********************************************************
void TcMETAnalyzer::fillMESet(const edm::Event& iEvent, std::string DirName, 
			      const reco::MET& tcmet)
{

  _dbe->setCurrentFolder(DirName);

  bool bLumiSecPlot=false;
  if (DirName.find("All")) bLumiSecPlot=true;

  if (_trig_JetMB) fillMonitorElement(iEvent,DirName,"",tcmet, bLumiSecPlot);
  if (_hlt_HighPtJet.size() && _trig_HighPtJet) fillMonitorElement(iEvent,DirName,"HighPtJet",tcmet,false);
  if (_hlt_LowPtJet.size() && _trig_LowPtJet) fillMonitorElement(iEvent,DirName,"LowPtJet",tcmet,false);
  if (_hlt_HighMET.size() && _trig_HighMET) fillMonitorElement(iEvent,DirName,"HighMET",tcmet,false);
  //  if (_hlt_LowMET.size() && _trig_LowMET) fillMonitorElement(iEvent,DirName,"LowMET",tcmet,false);
  if (_hlt_Ele.size() && _trig_Ele) fillMonitorElement(iEvent,DirName,"Ele",tcmet,false);
  if (_hlt_Muon.size() && _trig_Muon) fillMonitorElement(iEvent,DirName,"Muon",tcmet,false);
}

// ***********************************************************
void TcMETAnalyzer::fillMonitorElement(const edm::Event& iEvent, std::string DirName, 
					 std::string TriggerTypeName, 
					 const reco::MET& tcmet, bool bLumiSecPlot)
{

  if (TriggerTypeName=="HighPtJet") {
    if (!selectHighPtJetEvent(iEvent)) return;
  }
  else if (TriggerTypeName=="LowPtJet") {
    if (!selectLowPtJetEvent(iEvent)) return;
  }
  else if (TriggerTypeName=="HighMET") {
    if (tcmet.pt()<_highTcMETThreshold) return;
  }
  //  else if (TriggerTypeName=="LowMET") {
  //    if (tcmet.pt()<_lowTcMETThreshold) return;
  //  }
  else if (TriggerTypeName=="Ele") {
    if (!selectWElectronEvent(iEvent)) return;
  }
  else if (TriggerTypeName=="Muon") {
    if (!selectWMuonEvent(iEvent)) return;
  }
  
// Reconstructed MET Information
  double tcSumET  = tcmet.sumEt();
  double tcMETSig = tcmet.mEtSig();
  double tcEz     = tcmet.e_longitudinal();
  double tcMET    = tcmet.pt();
  double tcMEx    = tcmet.px();
  double tcMEy    = tcmet.py();
  double tcMETPhi = tcmet.phi();

  //
  int myLuminosityBlock;
  //  myLuminosityBlock = (evtCounter++)/1000;
  myLuminosityBlock = iEvent.luminosityBlock();
  //

  if (TriggerTypeName!="") DirName = DirName +"/"+TriggerTypeName;

  if (_verbose) std::cout << "_etThreshold = " << _etThreshold << std::endl;
  if (tcMET>_etThreshold){
    
    meTcMEx    = _dbe->get(DirName+"/"+"METTask_TcMEx");    if (meTcMEx    && meTcMEx->getRootObject())    meTcMEx->Fill(tcMEx);
    meTcMEy    = _dbe->get(DirName+"/"+"METTask_TcMEy");    if (meTcMEy    && meTcMEy->getRootObject())    meTcMEy->Fill(tcMEy);
    meTcMET    = _dbe->get(DirName+"/"+"METTask_TcMET");    if (meTcMET    && meTcMET->getRootObject())    meTcMET->Fill(tcMET);
    meTcMETPhi = _dbe->get(DirName+"/"+"METTask_TcMETPhi"); if (meTcMETPhi && meTcMETPhi->getRootObject()) meTcMETPhi->Fill(tcMETPhi);
    meTcSumET  = _dbe->get(DirName+"/"+"METTask_TcSumET");  if (meTcSumET  && meTcSumET->getRootObject())  meTcSumET->Fill(tcSumET);
    meTcMETSig = _dbe->get(DirName+"/"+"METTask_TcMETSig"); if (meTcMETSig && meTcMETSig->getRootObject()) meTcMETSig->Fill(tcMETSig);
    meTcEz     = _dbe->get(DirName+"/"+"METTask_TcEz");     if (meTcEz     && meTcEz->getRootObject())     meTcEz->Fill(tcEz);

    meTcMETIonFeedbck = _dbe->get(DirName+"/"+"METTask_TcMETIonFeedbck");  if (meTcMETIonFeedbck && meTcMETIonFeedbck->getRootObject()) meTcMETIonFeedbck->Fill(tcMET);
    meTcMETHPDNoise   = _dbe->get(DirName+"/"+"METTask_TcMETHPDNoise");    if (meTcMETHPDNoise   && meTcMETHPDNoise->getRootObject())   meTcMETHPDNoise->Fill(tcMET);
    meTcMETRBXNoise   = _dbe->get(DirName+"/"+"METTask_TcMETRBXNoise");    if (meTcMETRBXNoise   && meTcMETRBXNoise->getRootObject())   meTcMETRBXNoise->Fill(tcMET);
        
    if (_allhist){
      if (bLumiSecPlot){
	meTcMExLS = _dbe->get(DirName+"/"+"METTask_TcMExLS"); if (meTcMExLS && meTcMExLS->getRootObject()) meTcMExLS->Fill(tcMEx,myLuminosityBlock);
	meTcMEyLS = _dbe->get(DirName+"/"+"METTask_TcMEyLS"); if (meTcMEyLS && meTcMEyLS->getRootObject()) meTcMEyLS->Fill(tcMEy,myLuminosityBlock);
      }
    } // _allhist
  } // et threshold cut
}


// ***********************************************************
bool TcMETAnalyzer::selectHighPtJetEvent(const edm::Event& iEvent){

  bool return_value=false;

  edm::Handle<reco::CaloJetCollection> caloJets;
  iEvent.getByLabel(theJetCollectionLabel, caloJets);
  if (!caloJets.isValid()) {
    LogDebug("") << "TcMETAnalyzer: Could not find jet product" << std::endl;
    if (_verbose) std::cout << "TcMETAnalyzer: Could not find jet product" << std::endl;
  }

  for (reco::CaloJetCollection::const_iterator cal = caloJets->begin(); 
       cal!=caloJets->end(); ++cal){
    if (cal->pt()>_highPtTcJetThreshold){
      return_value=true;
    }
  }
  
  return return_value;
}

// // ***********************************************************
bool TcMETAnalyzer::selectLowPtJetEvent(const edm::Event& iEvent){

  bool return_value=false;

  edm::Handle<reco::CaloJetCollection> caloJets;
  iEvent.getByLabel(theJetCollectionLabel, caloJets);
  if (!caloJets.isValid()) {
    LogDebug("") << "TcMETAnalyzer: Could not find jet product" << std::endl;
    if (_verbose) std::cout << "TcMETAnalyzer: Could not find jet product" << std::endl;
  }

  for (reco::CaloJetCollection::const_iterator cal = caloJets->begin(); 
       cal!=caloJets->end(); ++cal){
    if (cal->pt()>_lowPtTcJetThreshold){
      return_value=true;
    }
  }

  return return_value;

}

// ***********************************************************
bool TcMETAnalyzer::selectWElectronEvent(const edm::Event& iEvent){

  bool return_value=false;

  /*
    W-electron event selection comes here
   */

  return return_value;

}

// ***********************************************************
bool TcMETAnalyzer::selectWMuonEvent(const edm::Event& iEvent){

  bool return_value=false;

  /*
    W-muon event selection comes here
   */

  return return_value;

}

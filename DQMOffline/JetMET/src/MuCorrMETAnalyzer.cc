/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/05/20 13:12:05 $
 *  $Revision: 1.11 $
 *  \author A.Apresyan - Caltech
 */

#include "DQMOffline/JetMET/interface/MuCorrMETAnalyzer.h"
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
MuCorrMETAnalyzer::MuCorrMETAnalyzer(const edm::ParameterSet& pSet) {

  parameters = pSet;

}

// ***********************************************************
MuCorrMETAnalyzer::~MuCorrMETAnalyzer() { }

void MuCorrMETAnalyzer::beginJob(DQMStore * dbe) {

  evtCounter = 0;
  metname = "muonMETAnalyzer";

  // trigger information
  HLTPathsJetMBByName_ = parameters.getParameter<std::vector<std::string > >("HLTPathsJetMB");

  _hlt_HighPtJet = parameters.getParameter<std::string>("HLT_HighPtJet");
  _hlt_LowPtJet  = parameters.getParameter<std::string>("HLT_LowPtJet");
  _hlt_HighMET   = parameters.getParameter<std::string>("HLT_HighMET");
  //  _hlt_LowMET    = parameters.getParameter<std::string>("HLT_LowMET");
  _hlt_Ele       = parameters.getParameter<std::string>("HLT_Ele");
  _hlt_Muon      = parameters.getParameter<std::string>("HLT_Muon");

  // MuCorrMET information
  theMuCorrMETCollectionLabel       = parameters.getParameter<edm::InputTag>("MuCorrMETCollectionLabel");
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

  _highPtMuCorrJetThreshold = parameters.getParameter<double>("HighPtMuCorrJetThreshold"); // High Pt Jet threshold
  _lowPtMuCorrJetThreshold = parameters.getParameter<double>("LowPtMuCorrJetThreshold");   // Low Pt Jet threshold
  _highMuCorrMETThreshold = parameters.getParameter<double>("HighMuCorrMETThreshold");     // High MET threshold
  //  _lowMuCorrMETThreshold = parameters.getParameter<double>("LowMuCorrMETThreshold");       // Low MET threshold

  //
  jetID = new reco::helper::JetIDHelper(parameters.getParameter<ParameterSet>("JetIDParams"));

  // DQStore stuff
  LogTrace(metname)<<"[MuCorrMETAnalyzer] Parameters initialization";
  std::string DirName = "JetMET/MET/"+_source;
  dbe->setCurrentFolder(DirName);

  metME = dbe->book1D("metReco", "metReco", 4, 1, 5);
  metME->setBinLabel(4,"MuCorrMET",1);

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
void MuCorrMETAnalyzer::endJob() {

  delete jetID;

}

// ***********************************************************
void MuCorrMETAnalyzer::bookMESet(std::string DirName)
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
void MuCorrMETAnalyzer::bookMonitorElement(std::string DirName, bool bLumiSecPlot=false)
{
  if (_verbose) std::cout << "booMonitorElement " << DirName << std::endl;

  _dbe->setCurrentFolder(DirName);

  meMuCorrMEx    = _dbe->book1D("METTask_MuCorrMEx",    "METTask_MuCorrMEx"   , 200, -500,  500);
  meMuCorrMEy    = _dbe->book1D("METTask_MuCorrMEy",    "METTask_MuCorrMEy"   , 200, -500,  500);
  meMuCorrMET    = _dbe->book1D("METTask_MuCorrMET",    "METTask_MuCorrMET"   , 200,    0, 1000);
  meMuCorrSumET  = _dbe->book1D("METTask_MuCorrSumET",  "METTask_MuCorrSumET" , 400,    0, 4000);
  meMuCorrMETSig = _dbe->book1D("METTask_MuCorrMETSig", "METTask_MuCorrMETSig",  51,    0,   51);
  meMuCorrMETPhi = _dbe->book1D("METTask_MuCorrMETPhi", "METTask_MuCorrMETPhi",  60, -3.2,  3.2);

  meMuCorrMETIonFeedbck = _dbe->book1D("METTask_MuCorrMETIonFeedbck", "METTask_MuCorrMETIonFeedbck", 200, 0, 1000);
  meMuCorrMETHPDNoise   = _dbe->book1D("METTask_MuCorrMETHPDNoise",   "METTask_MuCorrMETHPDNoise",   200, 0, 1000);
  meMuCorrMETRBXNoise   = _dbe->book1D("METTask_MuCorrMETRBXNoise",   "METTask_MuCorrMETRBXNoise",   200, 0, 1000);

  if (_allhist) {
    if (bLumiSecPlot) {
      meMuCorrMExLS = _dbe->book2D("METTask_MuCorrMEx_LS","METTask_MuCorrMEx_LS", 200, -200, 200, 50, 0, 500);
      meMuCorrMEyLS = _dbe->book2D("METTask_MuCorrMEy_LS","METTask_MuCorrMEy_LS", 200, -200, 200, 50, 0, 500);
    }
  }
}


// ***********************************************************
void MuCorrMETAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{

}

// ***********************************************************
void MuCorrMETAnalyzer::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup, DQMStore * dbe)
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
void MuCorrMETAnalyzer::makeRatePlot(std::string DirName, double totltime)
{

  _dbe->setCurrentFolder(DirName);
  MonitorElement *meMuCorrMET = _dbe->get(DirName+"/"+"METTask_MuCorrMET");

  TH1F* tMuCorrMET;
  TH1F* tMuCorrMETRate;

  if ( meMuCorrMET )
    if ( meMuCorrMET->getRootObject() ) {
      tMuCorrMET     = meMuCorrMET->getTH1F();
      
      // Integral plot & convert number of events to rate (hz)
      tMuCorrMETRate = (TH1F*) tMuCorrMET->Clone("METTask_MuCorrMETRate");
      for (int i = tMuCorrMETRate->GetNbinsX()-1; i>=0; i--){
	tMuCorrMETRate->SetBinContent(i+1,tMuCorrMETRate->GetBinContent(i+2)+tMuCorrMET->GetBinContent(i+1));
      }
      for (int i = 0; i<tMuCorrMETRate->GetNbinsX(); i++){
	tMuCorrMETRate->SetBinContent(i+1,tMuCorrMETRate->GetBinContent(i+1)/double(totltime));
      }      

      meMuCorrMETRate      = _dbe->book1D("METTask_MuCorrMETRate",tMuCorrMETRate);
      
    }

}

// ***********************************************************
void MuCorrMETAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
			    const edm::TriggerResults& triggerResults) {

  if (_verbose) std::cout << "MuCorrMETAnalyzer analyze" << std::endl;

  LogTrace(metname)<<"[MuCorrMETAnalyzer] Analyze MuCorrMET";

  metME->Fill(4);

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

    edm::LogInfo("MuCorrMetAnalyzer") << "TriggerResults::HLT not found, "
      "automatically select events"; 

    // TriggerResults object not found. Look at all events.    
    _trig_JetMB=1;
  }

  // ==========================================================
  // MuCorrMET information
  
  // **** Get the MET container  
  edm::Handle<reco::CaloMETCollection> muCorrmetcoll;
  iEvent.getByLabel("corMetGlobalMuons", muCorrmetcoll);
  
  if(!muCorrmetcoll.isValid()) return;

  const CaloMETCollection *muCorrmetcol = muCorrmetcoll.product();
  const CaloMET *muCorrmet;
  muCorrmet = &(muCorrmetcol->front());
    
  LogTrace(metname)<<"[MuCorrMETAnalyzer] Call to the MuCorrMET analyzer";

  // ==========================================================
  //
  edm::Handle<HcalNoiseRBXCollection> HRBXCollection;
  iEvent.getByLabel(HcalNoiseRBXCollectionTag,HRBXCollection);
  if (!HRBXCollection.isValid()) {
    LogDebug("") << "MuCorrMETAnalyzer: Could not find HcalNoiseRBX Collection" << std::endl;
    if (_verbose) std::cout << "MuCorrMETAnalyzer: Could not find HcalNoiseRBX Collection" << std::endl;
  }
  

  edm::Handle<bool> HBHENoiseFilterResultHandle;
  iEvent.getByLabel(HBHENoiseFilterResultTag, HBHENoiseFilterResultHandle);
  bool HBHENoiseFilterResult = *HBHENoiseFilterResultHandle;
  if (!HBHENoiseFilterResultHandle.isValid()) {
    LogDebug("") << "MuCorrMETAnalyzer: Could not find HBHENoiseFilterResult" << std::endl;
    if (_verbose) std::cout << "MuCorrMETAnalyzer: Could not find HBHENoiseFilterResult" << std::endl;
  }


  edm::Handle<reco::CaloJetCollection> caloJets;
  iEvent.getByLabel(theJetCollectionLabel, caloJets);
  if (!caloJets.isValid()) {
    LogDebug("") << "MuCorrMETAnalyzer: Could not find jet product" << std::endl;
    if (_verbose) std::cout << "MuCorrMETAnalyzer: Could not find jet product" << std::endl;
  }

  // ==========================================================
  // MuCorrMET sanity check

  //   if (_source=="MuCorrMET") validateMET(*muCorrmet, tcCandidates);
  
  // ==========================================================
  // JetID 

  if (_verbose) std::cout << "JetID starts" << std::endl;
  
  //
  // --- Loose cuts, not Muon specific for now!
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
    if (*ic=="All")                                   fillMESet(iEvent, DirName+"/"+*ic, *muCorrmet);
    if (*ic=="Cleanup" && bHcalNoiseFilter && bJetID) fillMESet(iEvent, DirName+"/"+*ic, *muCorrmet);
    if (_allSelection) {
    if (*ic=="HcalNoiseFilter"      && bHcalNoiseFilter )       fillMESet(iEvent, DirName+"/"+*ic, *muCorrmet);
    if (*ic=="JetID"      && bJetID)                            fillMESet(iEvent, DirName+"/"+*ic, *muCorrmet);
    if (*ic=="JetIDTight" && bJetIDTight)                       fillMESet(iEvent, DirName+"/"+*ic, *muCorrmet);
    }
  }
}

// ***********************************************************
void MuCorrMETAnalyzer::fillMESet(const edm::Event& iEvent, std::string DirName, 
			      const reco::CaloMET& muCorrmet)
{

  _dbe->setCurrentFolder(DirName);

  bool bLumiSecPlot=false;
  if (DirName.find("All")) bLumiSecPlot=true;

  if (_trig_JetMB) fillMonitorElement(iEvent,DirName,"",muCorrmet, bLumiSecPlot);
  if (_hlt_HighPtJet.size() && _trig_HighPtJet) fillMonitorElement(iEvent,DirName,"HighPtJet",muCorrmet,false);
  if (_hlt_LowPtJet.size() && _trig_LowPtJet) fillMonitorElement(iEvent,DirName,"LowPtJet",muCorrmet,false);
  if (_hlt_HighMET.size() && _trig_HighMET) fillMonitorElement(iEvent,DirName,"HighMET",muCorrmet,false);
  //  if (_hlt_LowMET.size() && _trig_LowMET) fillMonitorElement(iEvent,DirName,"LowMET",muCorrmet,false);
  if (_hlt_Ele.size() && _trig_Ele) fillMonitorElement(iEvent,DirName,"Ele",muCorrmet,false);
  if (_hlt_Muon.size() && _trig_Muon) fillMonitorElement(iEvent,DirName,"Muon",muCorrmet,false);
}

// ***********************************************************
void MuCorrMETAnalyzer::fillMonitorElement(const edm::Event& iEvent, std::string DirName, 
					 std::string TriggerTypeName, 
					 const reco::CaloMET& muCorrmet, bool bLumiSecPlot)
{

  if (TriggerTypeName=="HighPtJet") {
    if (!selectHighPtJetEvent(iEvent)) return;
  }
  else if (TriggerTypeName=="LowPtJet") {
    if (!selectLowPtJetEvent(iEvent)) return;
  }
  else if (TriggerTypeName=="HighMET") {
    if (muCorrmet.pt()<_highMuCorrMETThreshold) return;
  }
  //  else if (TriggerTypeName=="LowMET") {
  //    if (muCorrmet.pt()<_lowMuCorrMETThreshold) return;
  //  }
  else if (TriggerTypeName=="Ele") {
    if (!selectWElectronEvent(iEvent)) return;
  }
  else if (TriggerTypeName=="Muon") {
    if (!selectWMuonEvent(iEvent)) return;
  }
  

  // Reconstructed MET Information
  double mucorrSumET  = muCorrmet.sumEt();
  double mucorrmetSig = muCorrmet.mEtSig();
  //  double mucorrEz     = muCorrmet.e_longitudinal();
  double mucorrmet    = muCorrmet.pt();
  double mucorrMEx    = muCorrmet.px();
  double mucorrMEy    = muCorrmet.py();
  double mucorrmetPhi = muCorrmet.phi();

  //
  int myLuminosityBlock;
  //  myLuminosityBlock = (evtCounter++)/1000;
  myLuminosityBlock = iEvent.luminosityBlock();
  //

  if (TriggerTypeName!="") DirName = DirName +"/"+TriggerTypeName;

  if (_verbose) std::cout << "_etThreshold = " << _etThreshold << std::endl;
  if (mucorrmet>_etThreshold){
    
    meMuCorrMEx    = _dbe->get(DirName+"/"+"METTask_MuCorrMEx");    if (meMuCorrMEx    && meMuCorrMEx->getRootObject())    meMuCorrMEx->Fill(mucorrMEx);
    meMuCorrMEy    = _dbe->get(DirName+"/"+"METTask_MuCorrMEy");    if (meMuCorrMEy    && meMuCorrMEy->getRootObject())    meMuCorrMEy->Fill(mucorrMEy);
    meMuCorrMET    = _dbe->get(DirName+"/"+"METTask_MuCorrMET");    if (meMuCorrMET    && meMuCorrMET->getRootObject())    meMuCorrMET->Fill(mucorrmet);
    meMuCorrMETPhi = _dbe->get(DirName+"/"+"METTask_MuCorrMETPhi"); if (meMuCorrMETPhi && meMuCorrMETPhi->getRootObject()) meMuCorrMETPhi->Fill(mucorrmetPhi);
    meMuCorrSumET  = _dbe->get(DirName+"/"+"METTask_MuCorrSumET");  if (meMuCorrSumET  && meMuCorrSumET->getRootObject())  meMuCorrSumET->Fill(mucorrSumET);
    meMuCorrMETSig = _dbe->get(DirName+"/"+"METTask_MuCorrMETSig"); if (meMuCorrMETSig && meMuCorrMETSig->getRootObject()) meMuCorrMETSig->Fill(mucorrmetSig);

    meMuCorrMETIonFeedbck = _dbe->get(DirName+"/"+"METTask_MuCorrMETIonFeedbck");  if (meMuCorrMETIonFeedbck && meMuCorrMETIonFeedbck->getRootObject()) meMuCorrMETIonFeedbck->Fill(mucorrmet);
    meMuCorrMETHPDNoise   = _dbe->get(DirName+"/"+"METTask_MuCorrMETHPDNoise");    if (meMuCorrMETHPDNoise   && meMuCorrMETHPDNoise->getRootObject())   meMuCorrMETHPDNoise->Fill(mucorrmet);
    meMuCorrMETRBXNoise   = _dbe->get(DirName+"/"+"METTask_MuCorrMETRBXNoise");    if (meMuCorrMETRBXNoise   && meMuCorrMETRBXNoise->getRootObject())   meMuCorrMETRBXNoise->Fill(mucorrmet);
        
    if (_allhist){
      if (bLumiSecPlot){
	meMuCorrMExLS = _dbe->get(DirName+"/"+"METTask_MuCorrMExLS"); if (meMuCorrMExLS && meMuCorrMExLS->getRootObject()) meMuCorrMExLS->Fill(mucorrMEx,myLuminosityBlock);
	meMuCorrMEyLS = _dbe->get(DirName+"/"+"METTask_MuCorrMEyLS"); if (meMuCorrMEyLS && meMuCorrMEyLS->getRootObject()) meMuCorrMEyLS->Fill(mucorrMEy,myLuminosityBlock);
      }
    } // _allhist
  } // et threshold cut
}

// ***********************************************************
bool MuCorrMETAnalyzer::selectHighPtJetEvent(const edm::Event& iEvent){

  bool return_value=false;

  edm::Handle<reco::CaloJetCollection> caloJets;
  iEvent.getByLabel(theJetCollectionLabel, caloJets);
  if (!caloJets.isValid()) {
    LogDebug("") << "MuCorrMETAnalyzer: Could not find jet product" << std::endl;
    if (_verbose) std::cout << "MuCorrMETAnalyzer: Could not find jet product" << std::endl;
  }

  for (reco::CaloJetCollection::const_iterator cal = caloJets->begin(); 
       cal!=caloJets->end(); ++cal){
    if (cal->pt()>_highPtMuCorrJetThreshold){
      return_value=true;
    }
  }
  
  return return_value;
}

// // ***********************************************************
bool MuCorrMETAnalyzer::selectLowPtJetEvent(const edm::Event& iEvent){

  bool return_value=false;

  edm::Handle<reco::CaloJetCollection> caloJets;
  iEvent.getByLabel(theJetCollectionLabel, caloJets);
  if (!caloJets.isValid()) {
    LogDebug("") << "MuCorrMETAnalyzer: Could not find jet product" << std::endl;
    if (_verbose) std::cout << "MuCorrMETAnalyzer: Could not find jet product" << std::endl;
  }

  for (reco::CaloJetCollection::const_iterator cal = caloJets->begin(); 
       cal!=caloJets->end(); ++cal){
    if (cal->pt()>_lowPtMuCorrJetThreshold){
      return_value=true;
    }
  }

  return return_value;

}

// ***********************************************************
bool MuCorrMETAnalyzer::selectWElectronEvent(const edm::Event& iEvent){

  bool return_value=false;

  /*
    W-electron event selection comes here
   */

  return return_value;

}

// ***********************************************************
bool MuCorrMETAnalyzer::selectWMuonEvent(const edm::Event& iEvent){

  bool return_value=false;

  /*
    W-muon event selection comes here
   */

  return return_value;

}

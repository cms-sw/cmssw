/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/10/08 10:14:36 $
 *  $Revision: 1.3 $
 *  \author K. Hatakeyama - Rockefeller University
 *          A.Apresyan - Caltech
 */

#include "DQMOffline/JetMET/interface/PFMETAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include <string>
using namespace std;
using namespace edm;
using namespace reco;
using namespace math;

// ***********************************************************
PFMETAnalyzer::PFMETAnalyzer(const edm::ParameterSet& pSet) {

  parameters = pSet;

}

// ***********************************************************
PFMETAnalyzer::~PFMETAnalyzer() { }

void PFMETAnalyzer::beginJob(edm::EventSetup const& iSetup,DQMStore * dbe) {

  evtCounter = 0;
  metname = "pfMETAnalyzer";

  // trigger information
  HLTPathsJetMBByName_ = parameters.getParameter<std::vector<std::string > >("HLTPathsJetMB");

  _hlt_HighPtJet = parameters.getParameter<std::string>("HLT_HighPtJet");
  _hlt_LowPtJet  = parameters.getParameter<std::string>("HLT_LowPtJet");
  _hlt_HighMET   = parameters.getParameter<std::string>("HLT_HighMET");
  _hlt_LowMET    = parameters.getParameter<std::string>("HLT_LowMET");
  _hlt_Ele       = parameters.getParameter<std::string>("HLT_Ele");
  _hlt_Muon      = parameters.getParameter<std::string>("HLT_Muon");

  // PFMET information
  thePfMETCollectionLabel       = parameters.getParameter<edm::InputTag>("PfMETCollectionLabel");
  thePfJetCollectionLabel       = parameters.getParameter<edm::InputTag>("PfJetCollectionLabel");
  _source                       = parameters.getParameter<std::string>("Source");

  // Other data collections
  HcalNoiseRBXCollectionTag   = parameters.getParameter<edm::InputTag>("HcalNoiseRBXCollection");
  HcalNoiseSummaryTag         = parameters.getParameter<edm::InputTag>("HcalNoiseSummary");
  theJetCollectionLabel       = parameters.getParameter<edm::InputTag>("JetCollectionLabel");
  PFCandidatesTag             = parameters.getParameter<edm::InputTag>("PFCandidates");

  // misc
  _verbose     = parameters.getParameter<int>("verbose");
  _etThreshold = parameters.getParameter<double>("etThreshold"); // MET threshold
  _allhist     = parameters.getParameter<bool>("allHist");       // Full set of monitoring histograms
  _allSelection= parameters.getParameter<bool>("allSelection");  // Plot with all sets of event selection

  _highPtPFJetThreshold = parameters.getParameter<double>("HighPtPFJetThreshold"); // High Pt Jet threshold
  _lowPtPFJetThreshold = parameters.getParameter<double>("LowPtPFJetThreshold");   // Low Pt Jet threshold
  _highPFMETThreshold = parameters.getParameter<double>("HighPFMETThreshold");     // High MET threshold
  _lowPFMETThreshold = parameters.getParameter<double>("LowPFMETThreshold");       // Low MET threshold

  // DQStore stuff
  LogTrace(metname)<<"[PFMETAnalyzer] Parameters initialization";
  std::string DirName = "JetMET/MET/"+_source;
  dbe->setCurrentFolder(DirName);

  metME = dbe->book1D("metReco", "metReco", 4, 1, 5);
  metME->setBinLabel(3,"PFMET",1);

  _dbe = dbe;

  _FolderNames.push_back("All");
  _FolderNames.push_back("Cleanup");
  _FolderNames.push_back("HcalNoiseFilter");
  _FolderNames.push_back("HcalNoiseFilterTight");
  _FolderNames.push_back("JetID");
  _FolderNames.push_back("JetIDTight");

  for (std::vector<std::string>::const_iterator ic = _FolderNames.begin(); 
       ic != _FolderNames.end(); ic++){
    if (*ic=="All")             bookMESet(DirName+"/"+*ic);
    if (*ic=="Cleanup")         bookMESet(DirName+"/"+*ic);
    if (_allSelection){
    if (*ic=="HcalNoiseFilter")      bookMESet(DirName+"/"+*ic);
    if (*ic=="HcalNoiseFilterTight") bookMESet(DirName+"/"+*ic);
    if (*ic=="JetID")                bookMESet(DirName+"/"+*ic);
    if (*ic=="JetIDTight")           bookMESet(DirName+"/"+*ic);
    }
  }
}

// ***********************************************************
void PFMETAnalyzer::bookMESet(std::string DirName)
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

  if (_hlt_LowMET.size()){
    bookMonitorElement(DirName+"/"+"LowMET",false);
    meTriggerName_LowMET = _dbe->bookString("triggerName_LowMET", _hlt_LowMET);
  }

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
void PFMETAnalyzer::bookMonitorElement(std::string DirName, bool bLumiSecPlot=false)
{

  if (_verbose) std::cout << "booMonitorElement " << DirName << std::endl;
  _dbe->setCurrentFolder(DirName);
 
  meNevents              = _dbe->book1D("METTask_Nevents", "METTask_Nevents"   ,1,0,1);
  mePfMEx                = _dbe->book1D("METTask_PfMEx",   "METTask_PfMEx"   ,500,-500,500);
  mePfMEy                = _dbe->book1D("METTask_PfMEy",   "METTask_PfMEy"   ,500,-500,500);
  mePfEz                 = _dbe->book1D("METTask_PfEz",    "METTask_PfEz"    ,500,-500,500);
  mePfMETSig             = _dbe->book1D("METTask_PfMETSig","METTask_PfMETSig",51,0,51);
  mePfMET                = _dbe->book1D("METTask_PfMET",   "METTask_PfMET"   ,500,0,1000);
  mePfMETPhi             = _dbe->book1D("METTask_PfMETPhi","METTask_PfMETPhi",80,-TMath::Pi(),TMath::Pi());
  mePfSumET              = _dbe->book1D("METTask_PfSumET", "METTask_PfSumET" ,500,0,2000);

  mePfNeutralEMFraction  = _dbe->book1D("METTask_PfNeutralEMFraction", "METTask_PfNeutralEMFraction" ,50,0.,1.);
  mePfNeutralHadFraction = _dbe->book1D("METTask_PfNeutralHadFraction","METTask_PfNeutralHadFraction",50,0.,1.);
  mePfChargedEMFraction  = _dbe->book1D("METTask_PfChargedEMFraction", "METTask_PfChargedEMFraction" ,50,0.,1.);
  mePfChargedHadFraction = _dbe->book1D("METTask_PfChargedHadFraction","METTask_PfChargedHadFraction",50,0.,1.);
  mePfMuonFraction       = _dbe->book1D("METTask_PfMuonFraction",      "METTask_PfMuonFraction"      ,50,0.,1.);

  mePfMETIonFeedbck      = _dbe->book1D("METTask_PfMETIonFeedbck", "METTask_PfMETIonFeedbck" ,500,0,1000);
  mePfMETHPDNoise        = _dbe->book1D("METTask_PfMETHPDNoise",   "METTask_PfMETHPDNoise"   ,500,0,1000);
  mePfMETRBXNoise        = _dbe->book1D("METTask_PfMETRBXNoise",   "METTask_PfMETRBXNoise"   ,500,0,1000);

  if (_allhist){
    if (bLumiSecPlot){
      mePfMExLS              = _dbe->book2D("METTask_PfMEx_LS","METTask_PfMEx_LS",200,-200,200,50,0.,500.);
      mePfMEyLS              = _dbe->book2D("METTask_PfMEy_LS","METTask_PfMEy_LS",200,-200,200,50,0.,500.);
    }
  }
}

// ***********************************************************
void PFMETAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{

  //
  //--- htlConfig_
  
//   hltConfig_.init(processname_);
//   if (!hltConfig_.init(processname_)) {
//     processname_ = "FU";
//     if (!hltConfig_.init(processname_)){
//       LogDebug("PFMETAnalyzer") << "HLTConfigProvider failed to initialize.";
//     }
//   }

//   if (_verbose) std::cout << hltConfig_.triggerIndex(_hlt_HighPtJet) << std::endl;
//   if (_verbose) std::cout << hltConfig_.triggerIndex(_hlt_LowPtJet)  << std::endl;
//   if (_verbose) std::cout << hltConfig_.triggerIndex(_hlt_HighMET)   << std::endl;
//   if (_verbose) std::cout << hltConfig_.triggerIndex(_hlt_LowMET)    << std::endl;
//   if (_verbose) std::cout << hltConfig_.triggerIndex(_hlt_Ele)       << std::endl;
//   if (_verbose) std::cout << hltConfig_.triggerIndex(_hlt_Muon)      << std::endl;

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
      if (_hlt_HighPtJet.size()) makeRatePlot(DirName+"/"+_hlt_HighPtJet,totltime);
      if (_hlt_LowPtJet.size())  makeRatePlot(DirName+"/"+_hlt_LowPtJet,totltime);
      if (_hlt_HighMET.size())   makeRatePlot(DirName+"/"+_hlt_HighMET,totltime);
      if (_hlt_LowMET.size())    makeRatePlot(DirName+"/"+_hlt_LowMET,totltime);
      if (_hlt_Ele.size())       makeRatePlot(DirName+"/"+_hlt_Ele,totltime);
      if (_hlt_Muon.size())      makeRatePlot(DirName+"/"+_hlt_Muon,totltime);

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
    edm::TriggerNames triggerNames; // TriggerNames class
    triggerNames.init(triggerResults);
    
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
  // --- Loose cuts, not PF specific for now!
  //
  bool bJetID=true;
  for (reco::CaloJetCollection::const_iterator cal = caloJets->begin(); 
       cal!=caloJets->end(); ++cal){ 
    jetID.calculate(iEvent, *cal);
    if (_verbose) std::cout << jetID.n90Hits() << " " 
			    << jetID.restrictedEMF() << " "
			    << cal->pt() << std::endl;
    if (cal->pt()>10.){
      //
      // for all regions
      if (jetID.n90Hits()<2)  bJetID=false; 
      if (jetID.fHPD()>=0.98) bJetID=false; 
      //if (jetID.restrictedEMF()<0.01) bJetID=false; 
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
    jetID.calculate(iEvent, *cal);
    if (cal->pt()>25.){
      //
      // for all regions
      if (jetID.fHPD()>=0.95) bJetIDTight=false; 
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
  // Reconstructed MET Information - fill MonitorElements
  
  std::string DirName = "JetMET/MET/"+_source;
  
  for (std::vector<std::string>::const_iterator ic = _FolderNames.begin(); 
       ic != _FolderNames.end(); ic++){
    if (*ic=="All")                                   fillMESet(iEvent, DirName+"/"+*ic, *pfmet);
    if (*ic=="Cleanup" && bHcalNoiseFilter && bJetID) fillMESet(iEvent, DirName+"/"+*ic, *pfmet);
    if (_allSelection) {
    if (*ic=="HcalNoiseFilter"      && bHcalNoiseFilter )       fillMESet(iEvent, DirName+"/"+*ic, *pfmet);
    if (*ic=="HcalNoiseFilterTight" && bHcalNoiseFilterTight )  fillMESet(iEvent, DirName+"/"+*ic, *pfmet);
    if (*ic=="JetID"      && bJetID)                            fillMESet(iEvent, DirName+"/"+*ic, *pfmet);
    if (*ic=="JetIDTight" && bJetIDTight)                       fillMESet(iEvent, DirName+"/"+*ic, *pfmet);
    }
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
	cout<<"PFSumEt: " << sumEt         <<", "<<"PFMETBlock: "<<pfmet.pt()<<endl;
	cout<<"PFMET: "   << missingEt.T() <<", "<<"PFMETBlock: "<<pfmet.pt()<<endl;
	cout<<"PFMETx: "  << missingEt.X() <<", "<<"PFMETBlockx: "<<pfmet.pt()<<endl;
	cout<<"PFMETy: "  << missingEt.Y() <<", "<<"PFMETBlocky: "<<pfmet.pt()<<endl;
      }
    }

// ***********************************************************
void PFMETAnalyzer::fillMESet(const edm::Event& iEvent, std::string DirName, 
			      const reco::PFMET& pfmet)
{

  _dbe->setCurrentFolder(DirName);

  bool bLumiSecPlot=false;
  if (DirName.find("All")) bLumiSecPlot=true;

  if (_trig_JetMB) fillMonitorElement(iEvent,DirName,"",pfmet, bLumiSecPlot);
  if (_hlt_HighPtJet.size() && _trig_HighPtJet) fillMonitorElement(iEvent,DirName,"HighPtJet",pfmet,false);
  if (_hlt_LowPtJet.size() && _trig_LowPtJet) fillMonitorElement(iEvent,DirName,"LowPtJet",pfmet,false);
  if (_hlt_HighMET.size() && _trig_HighMET) fillMonitorElement(iEvent,DirName,"HighMET",pfmet,false);
  if (_hlt_LowMET.size() && _trig_LowMET) fillMonitorElement(iEvent,DirName,"LowMET",pfmet,false);
  if (_hlt_Ele.size() && _trig_Ele) fillMonitorElement(iEvent,DirName,"Ele",pfmet,false);
  if (_hlt_Muon.size() && _trig_Muon) fillMonitorElement(iEvent,DirName,"Muon",pfmet,false);
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
  double pfEz     = pfmet.e_longitudinal();
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
  if (pfMET>_etThreshold){
    
    mePfMEx    = _dbe->get(DirName+"/"+"METTask_PfMEx");    if (mePfMEx    && mePfMEx->getRootObject())    mePfMEx->Fill(pfMEx);
    mePfMEy    = _dbe->get(DirName+"/"+"METTask_PfMEy");    if (mePfMEy    && mePfMEy->getRootObject())    mePfMEy->Fill(pfMEy);
    mePfMET    = _dbe->get(DirName+"/"+"METTask_PfMET");    if (mePfMET    && mePfMET->getRootObject())    mePfMET->Fill(pfMET);
    mePfMETPhi = _dbe->get(DirName+"/"+"METTask_PfMETPhi"); if (mePfMETPhi && mePfMETPhi->getRootObject()) mePfMETPhi->Fill(pfMETPhi);
    mePfSumET  = _dbe->get(DirName+"/"+"METTask_PfSumET");  if (mePfSumET  && mePfSumET->getRootObject())  mePfSumET->Fill(pfSumET);
    mePfMETSig = _dbe->get(DirName+"/"+"METTask_PfMETSig"); if (mePfMETSig && mePfMETSig->getRootObject()) mePfMETSig->Fill(pfMETSig);
    mePfEz     = _dbe->get(DirName+"/"+"METTask_PfEz");     if (mePfEz     && mePfEz->getRootObject())     mePfEz->Fill(pfEz);

    mePfMETIonFeedbck = _dbe->get(DirName+"/"+"METTask_PfMETIonFeedbck");  if (mePfMETIonFeedbck && mePfMETIonFeedbck->getRootObject()) mePfMETIonFeedbck->Fill(pfMET);
    mePfMETHPDNoise   = _dbe->get(DirName+"/"+"METTask_PfMETHPDNoise");    if (mePfMETHPDNoise   && mePfMETHPDNoise->getRootObject())   mePfMETHPDNoise->Fill(pfMET);
    mePfMETRBXNoise   = _dbe->get(DirName+"/"+"METTask_PfMETRBXNoise");    if (mePfMETRBXNoise   && mePfMETRBXNoise->getRootObject())   mePfMETRBXNoise->Fill(pfMET);
    
    mePfNeutralEMFraction = _dbe->get(DirName+"/"+"METTask_mePfNeutralEMFraction"); 
    if (mePfNeutralEMFraction   && mePfNeutralEMFraction->getRootObject()) mePfNeutralEMFraction->Fill(pfNeutralEMFraction);
    mePfNeutralHadFraction = _dbe->get(DirName+"/"+"METTask_mePfNeutralHadFraction"); 
    if (mePfNeutralHadFraction   && mePfNeutralHadFraction->getRootObject()) mePfNeutralHadFraction->Fill(pfNeutralHadFraction);
    mePfChargedEMFraction = _dbe->get(DirName+"/"+"METTask_mePfChargedEMFraction"); 
    if (mePfChargedEMFraction   && mePfChargedEMFraction->getRootObject()) mePfChargedEMFraction->Fill(pfChargedEMFraction);
    mePfChargedHadFraction = _dbe->get(DirName+"/"+"METTask_mePfChargedHadFraction"); 
    if (mePfChargedHadFraction   && mePfChargedHadFraction->getRootObject()) mePfChargedHadFraction->Fill(pfChargedHadFraction);
    mePfMuonFraction = _dbe->get(DirName+"/"+"METTask_mePfMuonFraction"); 
    if (mePfMuonFraction   && mePfMuonFraction->getRootObject()) mePfMuonFraction->Fill(pfMuonFraction);
    
    if (_allhist){
      if (bLumiSecPlot){
	mePfMExLS = _dbe->get(DirName+"/"+"METTask_PfMExLS"); if (mePfMExLS && mePfMExLS->getRootObject()) mePfMExLS->Fill(pfMEx,myLuminosityBlock);
	mePfMEyLS = _dbe->get(DirName+"/"+"METTask_PfMEyLS"); if (mePfMEyLS && mePfMEyLS->getRootObject()) mePfMEyLS->Fill(pfMEy,myLuminosityBlock);
      }
    } // _allhist
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

  bool return_value=false;

  /*
    W-electron event selection comes here
   */

  return return_value;

}

// ***********************************************************
bool PFMETAnalyzer::selectWMuonEvent(const edm::Event& iEvent){

  bool return_value=false;

  /*
    W-muon event selection comes here
   */

  return return_value;

}

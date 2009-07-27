/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/06/30 13:44:17 $
 *  $Revision: 1.11 $
 *  \author F. Chlebana - Fermilab
 *          K. Hatakeyama - Rockefeller University
 */

#include "DQMOffline/JetMET/interface/CaloMETAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/Math/interface/LorentzVector.h"

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
void CaloMETAnalyzer::beginJob(edm::EventSetup const& iSetup,DQMStore * dbe) {

  evtCounter = 0;
  metname = "caloMETAnalyzer";

  // trigger information
  HLTPathsJetMBByName_ = parameters.getParameter<std::vector<std::string > >("HLTPathsJetMB");
  nHLTPathsJetMB_=HLTPathsJetMBByName_.size();
  HLTPathsJetMBByIndex_.resize(nHLTPathsJetMB_);

  _hlt_HighPtJet = parameters.getParameter<std::string>("HLT_HighPtJet");
  _hlt_LowPtJet  = parameters.getParameter<std::string>("HLT_LowPtJet");
  _hlt_HighMET   = parameters.getParameter<std::string>("HLT_HighMET");
  _hlt_LowMET    = parameters.getParameter<std::string>("HLT_LowMET");

  processname_ = parameters.getParameter<std::string>("processname");

  // CaloMET information
  theCaloMETCollectionLabel       = parameters.getParameter<edm::InputTag>("CaloMETCollectionLabel");
  _source                         = parameters.getParameter<std::string>("Source");

  // Other data collections
  theCaloTowersLabel          = parameters.getParameter<edm::InputTag>("CaloTowersLabel");
  theJetCollectionLabel       = parameters.getParameter<edm::InputTag>("JetCollectionLabel");
  HcalNoiseRBXCollectionTag   = parameters.getParameter<edm::InputTag>("HcalNoiseRBXCollection");
  HcalNoiseSummaryTag         = parameters.getParameter<edm::InputTag>("HcalNoiseSummary");

  // misc
  _verbose     = parameters.getParameter<int>("verbose");
  _etThreshold = parameters.getParameter<double>("etThreshold"); // MET threshold
  _allhist     = parameters.getParameter<bool>("allHist");       // Full set of monitoring histograms
  _allSelection= parameters.getParameter<bool>("allSelection");  // Plot with all sets of event selection

  // DQStore stuff
  LogTrace(metname)<<"[CaloMETAnalyzer] Parameters initialization";
  std::string DirName = "JetMET/MET/"+_source;
  dbe->setCurrentFolder(DirName);

  jetME = dbe->book1D("metReco", "metReco", 4, 1, 5);
  jetME->setBinLabel(1,"CaloMET",1);

  _dbe = dbe;

  _FolderNames.push_back("All");
  _FolderNames.push_back("Cleanup");
  _FolderNames.push_back("HcalNoiseFilter");
  _FolderNames.push_back("JetID");

  for (std::vector<std::string>::const_iterator ic = _FolderNames.begin(); 
       ic != _FolderNames.end(); ic++){
    if (*ic=="All")             bookMESet(DirName+"/"+*ic);
    if (*ic=="Cleanup")         bookMESet(DirName+"/"+*ic);
    if (_allSelection){
    if (*ic=="HcalNoiseFilter") bookMESet(DirName+"/"+*ic);
    if (*ic=="JetID")           bookMESet(DirName+"/"+*ic);
    }
  }

}

// ***********************************************************
void CaloMETAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{

  //
  //--- htlConfig_
  processname_="HLT";
  hltConfig_.init(processname_);
  if (!hltConfig_.init(processname_)) {
    processname_ = "FU";
    if (!hltConfig_.init(processname_)){
      LogDebug("CaloMETAnalyzer") << "HLTConfigProvider failed to initialize.";
    }
  }

  if (_verbose) std::cout << hltConfig_.triggerIndex(_hlt_HighPtJet) << std::endl;
  if (_verbose) std::cout << hltConfig_.triggerIndex(_hlt_LowPtJet)  << std::endl;
  if (_verbose) std::cout << hltConfig_.triggerIndex(_hlt_HighMET)   << std::endl;
  if (_verbose) std::cout << hltConfig_.triggerIndex(_hlt_LowMET)    << std::endl;

}

// ***********************************************************
void CaloMETAnalyzer::bookMESet(std::string DirName)
{

  bool bLumiSecPlot=false;
  if (DirName.find("All")!=std::string::npos) bLumiSecPlot=true;
  //std::cout << DirName.find("All") << " " << DirName << " " << DirName.size() << " " << std::string::npos << std::endl;

  bookMonitorElement(DirName,bLumiSecPlot);
  if (_hlt_HighPtJet.size()) bookMonitorElement(DirName+"/"+_hlt_HighPtJet,false);
  if (_hlt_LowPtJet.size())  bookMonitorElement(DirName+"/"+_hlt_LowPtJet,false);
  if (_hlt_HighMET.size())   bookMonitorElement(DirName+"/"+_hlt_HighMET,false);
  if (_hlt_LowMET.size())    bookMonitorElement(DirName+"/"+_hlt_LowMET,false);

}

// ***********************************************************
void CaloMETAnalyzer::bookMonitorElement(std::string DirName, bool bLumiSecPlot=false)
{

  if (_verbose) std::cout << "bookMonitorElement " << DirName << std::endl;
  _dbe->setCurrentFolder(DirName);
  
  meNevents                = _dbe->book1D("METTask_Nevents",   "METTask_Nevents"   ,1,0,1);
  meCaloMEx                = _dbe->book1D("METTask_CaloMEx",   "METTask_CaloMEx"   ,500,-500,500);
  meCaloMEy                = _dbe->book1D("METTask_CaloMEy",   "METTask_CaloMEy"   ,500,-500,500);
  meCaloEz                 = _dbe->book1D("METTask_CaloEz",    "METTask_CaloEz"    ,500,-500,500);
  meCaloMETSig             = _dbe->book1D("METTask_CaloMETSig","METTask_CaloMETSig",51,0,51);
  meCaloMET                = _dbe->book1D("METTask_CaloMET",   "METTask_CaloMET"   ,500,0,1000);
  meCaloMETPhi             = _dbe->book1D("METTask_CaloMETPhi","METTask_CaloMETPhi",80,-4,4);
  meCaloSumET              = _dbe->book1D("METTask_CaloSumET", "METTask_CaloSumET" ,500,0,2000);

  if (_allhist){
    if (bLumiSecPlot){
      meCaloMExLS              = _dbe->book2D("METTask_CaloMEx_LS","METTask_CaloMEx_LS",200,-200,200,50,0.,500.);
      meCaloMEyLS              = _dbe->book2D("METTask_CaloMEy_LS","METTask_CaloMEy_LS",200,-200,200,50,0.,500.);
    }

    meCaloMaxEtInEmTowers    = _dbe->book1D("METTask_CaloMaxEtInEmTowers",   "METTask_CaloMaxEtInEmTowers"   ,100,0,2000);
    meCaloMaxEtInHadTowers   = _dbe->book1D("METTask_CaloMaxEtInHadTowers",  "METTask_CaloMaxEtInHadTowers"  ,100,0,2000);
    meCaloEtFractionHadronic = _dbe->book1D("METTask_CaloEtFractionHadronic","METTask_CaloEtFractionHadronic",100,0,1);
    meCaloEmEtFraction       = _dbe->book1D("METTask_CaloEmEtFraction",      "METTask_CaloEmEtFraction"      ,100,0,1);

    meCaloHadEtInHB          = _dbe->book1D("METTask_CaloHadEtInHB","METTask_CaloHadEtInHB",100,0,2000);
    meCaloHadEtInHO          = _dbe->book1D("METTask_CaloHadEtInHO","METTask_CaloHadEtInHO",100,0,2000);
    meCaloHadEtInHE          = _dbe->book1D("METTask_CaloHadEtInHE","METTask_CaloHadEtInHE",100,0,2000);
    meCaloHadEtInHF          = _dbe->book1D("METTask_CaloHadEtInHF","METTask_CaloHadEtInHF",100,0,2000);
    meCaloEmEtInHF           = _dbe->book1D("METTask_CaloEmEtInHF" ,"METTask_CaloEmEtInHF" ,100,0,2000);
    meCaloEmEtInEE           = _dbe->book1D("METTask_CaloEmEtInEE" ,"METTask_CaloEmEtInEE" ,100,0,2000);
    meCaloEmEtInEB           = _dbe->book1D("METTask_CaloEmEtInEB" ,"METTask_CaloEmEtInEB" ,100,0,2000);
  }

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
	tCaloMETRate->SetBinContent(i+1,tCaloMETRate->GetBinContent(i+1)/double(totltime));
      }
      
      meCaloMETRate      = _dbe->book1D("METTask_CaloMETRate",tCaloMETRate);
      
    }

}


// ***********************************************************
void CaloMETAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
			      const edm::TriggerResults& triggerResults) {

  LogTrace(metname)<<"[CaloMETAnalyzer] Analyze CaloMET";

  jetME->Fill(1);

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
    // Fill HLTPathsJetMBByIndex_[i]
    // If index=ntrigs, this HLT trigger doesn't exist in the HLT table for this data.
    edm::TriggerNames triggerNames; // TriggerNames class
    triggerNames.init(triggerResults);
    unsigned int n(nHLTPathsJetMB_);
    for (unsigned int i=0; i!=n; i++) {
      HLTPathsJetMBByIndex_[i]=triggerNames.triggerIndex(HLTPathsJetMBByName_[i]);
    }
    
    //
    //
    // for empty input vectors (n==0), use all HLT trigger paths!
    if (n==0) {
      n=triggerResults.size();
      HLTPathsJetMBByName_.resize(n);
      HLTPathsJetMBByIndex_.resize(n);
      for (unsigned int i=0; i!=n; i++) {
	  HLTPathsJetMBByName_[i]=triggerNames.triggerName(i);
	  HLTPathsJetMBByIndex_[i]=i;
      }
    }  

    //
    //
    // count number of requested Jet or MB HLT paths which have fired
    for (unsigned int i=0; i!=n; i++) {
      if (HLTPathsJetMBByIndex_[i]<triggerResults.size()) {
	if (triggerResults.accept(HLTPathsJetMBByIndex_[i])) {
	  _trig_JetMB++;
	}
      }
    }
      
    //
    if (triggerResults.accept(triggerNames.triggerIndex(_hlt_HighPtJet))) _trig_HighPtJet=1;
    if (triggerResults.accept(triggerNames.triggerIndex(_hlt_LowPtJet)))  _trig_LowPtJet=1;
    if (triggerResults.accept(triggerNames.triggerIndex(_hlt_HighMET)))   _trig_HighMET=1;
    if (triggerResults.accept(triggerNames.triggerIndex(_hlt_LowMET)))    _trig_LowMET=1;
    
  } else {

    edm::LogInfo("CaloMetAnalyzer") << "TriggerResults::HLT not found, "
	"automatically select events"; 
    //return;
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
  edm::Handle<HcalNoiseRBXCollection> RBXCollection;
  iEvent.getByLabel(HcalNoiseRBXCollectionTag,RBXCollection);
  if (!RBXCollection.isValid()) {
      LogDebug("") << "CaloMETAnalyzer: Could not find HcalNoiseRBX Collection" << std::endl;
      if (_verbose) std::cout << "CaloMETAnalyzer: Could not find HcalNoiseRBX Collection" << std::endl;
  }
  
  edm::Handle<HcalNoiseSummary> NoiseSummary;
  iEvent.getByLabel(HcalNoiseSummaryTag,NoiseSummary);
  if (!NoiseSummary.isValid()) {
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
  // JetID 

  bool bJetID=true;
  for (reco::CaloJetCollection::const_iterator cal = caloJets->begin(); 
       cal!=caloJets->end(); ++cal){
    jetID.calculate(iEvent, *cal);
    if (_verbose) std::cout << jetID.n90Hits() << " " 
			    << jetID.restrictedEMF() << " "
			    << cal->pt() << std::endl;
    if (cal->pt()>10.){
      if (jetID.n90Hits()<2 || jetID.restrictedEMF()<0.01) bJetID=false; 
    }
  }
     
  // ==========================================================
  // HCAL Noise filter
  
  double EMFractionMin = 0.1;
  double nRBXhitsMax = 50;
  double RBXhitThresh = 1.5;

  bool bHcalNoiseFilter=true;

  //std::cout << "NoiseSummary: " << NoiseSummary->eventEMFraction() << std::endl;
  if (NoiseSummary->eventEMFraction() < EMFractionMin) bHcalNoiseFilter = false;
 
  //std::cout << "Size of NoiseRBX collection:  " << RBXCollection->size() << std::endl;
  for (HcalNoiseRBXCollection::const_iterator rbx = RBXCollection->begin();
       rbx!=(RBXCollection->end()); rbx++) {
    int nRBXhits=rbx->numRecHits(RBXhitThresh);
    //std::cout << "\tNumber of RBX hits:     " << nRBXhits << std::endl;
    if (nRBXhits > nRBXhitsMax ) bHcalNoiseFilter = false;
  }

  // ==========================================================
  // Reconstructed MET Information - fill MonitorElements
  
  std::string DirName = "JetMET/MET/"+_source;
  
  for (std::vector<std::string>::const_iterator ic = _FolderNames.begin(); 
       ic != _FolderNames.end(); ic++){
    if (*ic=="All")                                   fillMESet(iEvent, DirName+"/"+*ic, *calomet);
    if (*ic=="Cleanup" && bHcalNoiseFilter && bJetID) fillMESet(iEvent, DirName+"/"+*ic, *calomet);
    if (_allSelection) {
    if (*ic=="HcalNoiseFilter" && bHcalNoiseFilter )  fillMESet(iEvent, DirName+"/"+*ic, *calomet);
    if (*ic=="JetID" && bJetID)                       fillMESet(iEvent, DirName+"/"+*ic, *calomet);
    }
  }



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
	    if (Tower_ET>0.5) {
	      
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

}

// ***********************************************************
void CaloMETAnalyzer::fillMESet(const edm::Event& iEvent, std::string DirName, 
				const reco::CaloMET& calomet)
{

  _dbe->setCurrentFolder(DirName);

  bool bLumiSecPlot=false;
  if (DirName.find("All")) bLumiSecPlot=true;

  if (_trig_JetMB) fillMonitorElement(iEvent,DirName, calomet, bLumiSecPlot);
  if (_hlt_HighPtJet.size() && _trig_HighPtJet) fillMonitorElement(iEvent,DirName+"/"+_hlt_HighPtJet,calomet,false);
  if (_hlt_LowPtJet.size() && _trig_LowPtJet) fillMonitorElement(iEvent,DirName+"/"+_hlt_LowPtJet,calomet,false);
  if (_hlt_HighMET.size() && _trig_HighMET) fillMonitorElement(iEvent,DirName+"/"+_hlt_HighMET,calomet,false);
  if (_hlt_LowMET.size() && _trig_LowMET) fillMonitorElement(iEvent,DirName+"/"+_hlt_LowMET,calomet,false);

}

// ***********************************************************
void CaloMETAnalyzer::fillMonitorElement(const edm::Event& iEvent, std::string DirName, 
					 const reco::CaloMET& calomet, bool bLumiSecPlot)
{

  _dbe->setCurrentFolder(DirName);

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

  if (_verbose) std::cout << "_etThreshold = " << _etThreshold << std::endl;
  if (caloMET>_etThreshold){

    meCaloMEx    = _dbe->get(DirName+"/"+"METTask_CaloMEx");    if (meCaloMEx    && meCaloMEx->getRootObject())    meCaloMEx->Fill(caloMEx);
    meCaloMEy    = _dbe->get(DirName+"/"+"METTask_CaloMEy");    if (meCaloMEy    && meCaloMEy->getRootObject())    meCaloMEy->Fill(caloMEy);
    meCaloMET    = _dbe->get(DirName+"/"+"METTask_CaloMET");    if (meCaloMET    && meCaloMET->getRootObject())    meCaloMET->Fill(caloMET);
    meCaloMETPhi = _dbe->get(DirName+"/"+"METTask_CaloMETPhi"); if (meCaloMETPhi && meCaloMETPhi->getRootObject()) meCaloMETPhi->Fill(caloMETPhi);
    meCaloSumET  = _dbe->get(DirName+"/"+"METTask_CaloSumET");  if (meCaloSumET  && meCaloSumET->getRootObject())  meCaloSumET->Fill(caloSumET);
    meCaloMETSig = _dbe->get(DirName+"/"+"METTask_CaloMETSig"); if (meCaloMETSig && meCaloMETSig->getRootObject()) meCaloMETSig->Fill(caloMETSig);
    meCaloEz     = _dbe->get(DirName+"/"+"METTask_CaloEz");     if (meCaloEz     && meCaloEz->getRootObject())     meCaloEz->Fill(caloEz);

    if (_allhist){
      if (bLumiSecPlot){
      meCaloMExLS = _dbe->get(DirName+"/"+"METTask_CaloMExLS"); if (meCaloMExLS && meCaloMExLS->getRootObject()) meCaloMExLS->Fill(caloMEx,myLuminosityBlock);
      meCaloMEyLS = _dbe->get(DirName+"/"+"METTask_CaloMEyLS"); if (meCaloMEyLS && meCaloMEyLS->getRootObject()) meCaloMEyLS->Fill(caloMEy,myLuminosityBlock);
      }
   
      meCaloEtFractionHadronic = _dbe->get(DirName+"/"+"METTask_CaloEtFractionHadronic"); 
        if (meCaloEtFractionHadronic && meCaloEtFractionHadronic->getRootObject()) meCaloEtFractionHadronic->Fill(caloEtFractionHadronic);
      meCaloEmEtFraction = _dbe->get(DirName+"/"+"METTask_CaloEmEtFraction"); 
        if (meCaloEmEtFraction && meCaloEmEtFraction->getRootObject()) meCaloEmEtFraction->Fill(caloEmEtFraction);

      meCaloMaxEtInEmTowers  = _dbe->get(DirName+"/"+"METTask_CaloMaxEtInEmTowers");  if (meCaloMaxEtInEmTowers  && meCaloMaxEtInEmTowers->getRootObject())  meCaloMaxEtInEmTowers->Fill(caloMaxEtInEMTowers);
      meCaloMaxEtInHadTowers = _dbe->get(DirName+"/"+"METTask_CaloMaxEtInHadTowers"); if (meCaloMaxEtInHadTowers && meCaloMaxEtInHadTowers->getRootObject()) meCaloMaxEtInHadTowers->Fill(caloMaxEtInHadTowers);

      meCaloHadEtInHB = _dbe->get(DirName+"/"+"METTask_CaloHadEtInHB"); if (meCaloHadEtInHB && meCaloHadEtInHB->getRootObject()) meCaloHadEtInHB->Fill(caloHadEtInHB);
      meCaloHadEtInHO = _dbe->get(DirName+"/"+"METTask_CaloHadEtInHO"); if (meCaloHadEtInHO && meCaloHadEtInHO->getRootObject()) meCaloHadEtInHO->Fill(caloHadEtInHO);
      meCaloHadEtInHE = _dbe->get(DirName+"/"+"METTask_CaloHadEtInHE"); if (meCaloHadEtInHE && meCaloHadEtInHE->getRootObject()) meCaloHadEtInHE->Fill(caloHadEtInHE);
      meCaloHadEtInHF = _dbe->get(DirName+"/"+"METTask_CaloHadEtInHF"); if (meCaloHadEtInHF && meCaloHadEtInHF->getRootObject()) meCaloHadEtInHF->Fill(caloHadEtInHF);
      meCaloEmEtInEB = _dbe->get(DirName+"/"+"METTask_CaloEmEtInEB"); if (meCaloEmEtInEB && meCaloEmEtInEB->getRootObject()) meCaloEmEtInEB->Fill(caloEmEtInEB);
      meCaloEmEtInEE = _dbe->get(DirName+"/"+"METTask_CaloEmEtInEE"); if (meCaloEmEtInEE && meCaloEmEtInEE->getRootObject()) meCaloEmEtInEE->Fill(caloEmEtInEE);
      meCaloEmEtInHF = _dbe->get(DirName+"/"+"METTask_CaloEmEtInHF"); if (meCaloEmEtInHF && meCaloEmEtInHF->getRootObject()) meCaloEmEtInHF->Fill(caloEmEtInHF);
    } // _allhist

  } // et threshold cut

}

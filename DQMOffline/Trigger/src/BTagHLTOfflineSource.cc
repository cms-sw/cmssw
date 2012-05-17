/*
  New version of HLT Offline DQM code for BTagMu paths
  responsible: Jyothsna Komaragiri
*/

#include "TMath.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMOffline/Trigger/interface/BTagHLTOfflineSource.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "math.h"
#include "TH1F.h"
#include "TProfile.h"
#include "TH2F.h"
#include "TPRegexp.h"

using namespace edm;
using namespace reco;
using namespace std;

  
BTagHLTOfflineSource::BTagHLTOfflineSource(const edm::ParameterSet& iConfig):
  isSetup_(false)
{


  SelectedCaloJets = new reco::CaloJetCollection;
  SelectedMuons = new reco::MuonCollection;

  LogDebug("BTagHLTOfflineSource") << "constructor....";

  dbe = Service < DQMStore > ().operator->();
  if ( ! dbe ) {
    LogDebug("BTagHLTOfflineSource") << "unabel to get DQMStore service?";
  }
  if (iConfig.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe->setVerbose(0);
  }
  

  dirname_ = iConfig.getUntrackedParameter("dirname",
					   std::string("HLT/BTag/"));
  
  processname_  = iConfig.getParameter<std::string>("processname");
  verbose_      = iConfig.getUntrackedParameter< bool >("verbose", false);
  plotEff_      = iConfig.getUntrackedParameter< bool >("plotEff", false);
  nameForEff_   = iConfig.getUntrackedParameter< bool >("nameForEff", true);

  jetID = new reco::helper::JetIDHelper(iConfig.getParameter<ParameterSet>("JetIDParams"));
  
  // plotting paramters
  MuonTrigPaths_ = iConfig.getUntrackedParameter<vector<std::string> >("pathnameMuon");
  MBTrigPaths_   = iConfig.getUntrackedParameter<vector<std::string> >("pathnameMB");
  caloJetsTag_   = iConfig.getParameter<edm::InputTag>("CaloJetCollectionLabel");
  muonTag_       = iConfig.getParameter<edm::InputTag>("MuonCollectionLabel");

  triggerSummaryLabel_ = iConfig.getParameter<edm::InputTag>("triggerSummaryLabel");
  triggerResultsLabel_ = iConfig.getParameter<edm::InputTag>("triggerResultsLabel");
  custompathname       = iConfig.getUntrackedParameter<vector<std::string> >("paths");

  //Jet selection cuts
  _jetpt   = iConfig.getUntrackedParameter< double >("jetpt", 20.);
  _jeteta  = iConfig.getUntrackedParameter< double >("jeteta", 2.40);
  _fEMF    = iConfig.getUntrackedParameter< double >("fEMF", 0.01);
  _fHPD    = iConfig.getUntrackedParameter< double >("fHPD", 0.98);
  _n90Hits = iConfig.getUntrackedParameter< double >("n90Hits", 1.0);

  //Muon selection cuts
  _mupt             = iConfig.getUntrackedParameter< double >("mupt", 6.);
  _mueta            = iConfig.getUntrackedParameter< double >("mueta", 2.40);
  _muonHits         = iConfig.getUntrackedParameter< int >("muonHits");
  _nMatches         = iConfig.getUntrackedParameter< int >("nMatches");
  _trackerHits      = iConfig.getUntrackedParameter< int >("trackerHits");
  _pixelHits        = iConfig.getUntrackedParameter< int >("pixelHits");
  _outerHits        = iConfig.getUntrackedParameter< int >("outerHits");
  _tknormalizedChi2 = iConfig.getUntrackedParameter< double >("tknormalizedChi2");
  _gmnormalizedChi2 = iConfig.getUntrackedParameter< double >("gmnormalizedChi2");
  _mudZ             = iConfig.getUntrackedParameter< double >("mudZ");
  _mujetdR          = iConfig.getUntrackedParameter< double >("mujetdR");


  // this is the list of paths to look at.
  std::vector<edm::ParameterSet> paths =  iConfig.getParameter<std::vector<edm::ParameterSet> >("pathPairs");
  for(std::vector<edm::ParameterSet>::iterator pathconf = paths.begin() ; pathconf != paths.end();  pathconf++) {
    std::pair<std::string, std::string> custompathnamepair;
    custompathnamepair.first =pathconf->getParameter<std::string>("pathname"); 
    custompathnamepair.second = pathconf->getParameter<std::string>("denompathname");   
    custompathnamepairs_.push_back(custompathnamepair);
  } 
  
}

//--------------------------------------------------------
BTagHLTOfflineSource::~BTagHLTOfflineSource() {
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
  delete SelectedCaloJets;
  delete SelectedMuons;

}

//--------------------------------------------------------
void BTagHLTOfflineSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  
  using namespace edm;
  using namespace trigger;
  using namespace reco;

  //---------- triggerResults ----------
  iEvent.getByLabel(triggerResultsLabel_, triggerResults_);
  if(!triggerResults_.isValid()) {
    edm::InputTag triggerResultsLabelFU(triggerResultsLabel_.label(),triggerResultsLabel_.instance(), "FU");
    iEvent.getByLabel(triggerResultsLabelFU,triggerResults_);
    if(!triggerResults_.isValid()) {
      edm::LogInfo("BTagHLTOfflineSource") << "TriggerResults not found, "
	"skipping event";
      return;
    }
  }
  
  int npath;
  if(&triggerResults_) {
  
    // Check how many HLT triggers are in triggerResults
    npath = triggerResults_->size();
    triggerNames_ = iEvent.triggerNames(*triggerResults_);
    
  } 
  else {
    
    edm::LogInfo("BTagHLTOfflineSource") << "TriggerResults::HLT not found, "
      "automatically select events";
    return;
  }

  //---------- triggerSummary ----------
  iEvent.getByLabel(triggerSummaryLabel_,triggerObj_);
  if(!triggerObj_.isValid()) {
    edm::InputTag triggerSummaryLabelFU(triggerSummaryLabel_.label(),triggerSummaryLabel_.instance(), "FU");
    iEvent.getByLabel(triggerSummaryLabelFU,triggerObj_);
    if(!triggerObj_.isValid()) {
      edm::LogInfo("BTagHLTOfflineSource") << "TriggerEvent not found, "
	"skipping event";
      return;
    }
  }
  
  //------------Offline Objects-------

  //Access the reco calo jets 
  edm::Handle<reco::CaloJetCollection> jetHandle;
  bool ValidJetColl_ = iEvent.getByLabel(caloJetsTag_, jetHandle);
  if(!ValidJetColl_) return;
  // get the selected jets
  selectJets(iEvent,jetHandle);

  //Access the reco muons
  edm::Handle<reco::MuonCollection> muonHandle;
  bool ValidMuColl_ = iEvent.getByLabel(muonTag_, muonHandle);
  if(!ValidMuColl_) return;
  // get the selected muons
  selectMuons(muonHandle);

  // Beam spot
  if (!iEvent.getByLabel(InputTag("offlineBeamSpot"), beamSpot_)) {
        edm::LogInfo("") << ">>> No beam spot found !!!";
  }

  fillMEforMonTriggerSummary();
  
  if(plotEff_)
    {
      fillMEforEffAllTrigger(iEvent); 
      fillMEforEffWrtMuTrigger(iEvent);
      fillMEforEffWrtMBTrigger(iEvent);
    }

  fillMEforTriggerNTfired();

}

//--------------------------------------------------------
void BTagHLTOfflineSource::fillMEforMonTriggerSummary(){
  // Trigger summary for all paths

  bool muTrig = false;
  bool mbTrig = false;

  for(size_t i=0;i<MuonTrigPaths_.size();++i){
    if(isHLTPathAccepted(MuonTrigPaths_[i])){
      muTrig = true;
      break;
    } 
  }
  for(size_t i=0;i<MBTrigPaths_.size();++i){
    if(isHLTPathAccepted(MBTrigPaths_[i])){
      mbTrig = true;
      break;
    }
  }

  for(PathInfoCollection::iterator v = hltPathsAll_.begin(); v!= hltPathsAll_.end(); ++v )
    {
      bool trigFirst= false;  
      double binV = TriggerPosition(v->getPath());
      
      if(isHLTPathAccepted(v->getPath())) trigFirst = true;
      if(!trigFirst)continue;
      if(trigFirst)
	{
	  rate_All->Fill(binV);
	  correlation_All->Fill(binV,binV);
	  if(muTrig){
	    rate_AllWrtMu->Fill(binV);
	    correlation_AllWrtMu->Fill(binV,binV);
	  }
	  if(mbTrig){
	    rate_AllWrtMB->Fill(binV);
	    correlation_AllWrtMB->Fill(binV,binV);
	  }
	}
      for(PathInfoCollection::iterator w = v+1; w!= hltPathsAll_.end(); ++w )
	{
	  bool trigSec = false; 
	  double binW = TriggerPosition(w->getPath()); 
	  if(isHLTPathAccepted(w->getPath()))trigSec = true;
	  if(trigSec && trigFirst)
	    {
	      correlation_All->Fill(binV,binW);
	      if(muTrig)correlation_AllWrtMu->Fill(binV,binW);
	      if(mbTrig)correlation_AllWrtMB->Fill(binV,binW); 
	    }
	  if(!trigSec && trigFirst)
	    {
	      correlation_All->Fill(binW,binV); 
	      if(muTrig)correlation_AllWrtMu->Fill(binW,binV);
	      if(mbTrig)correlation_AllWrtMB->Fill(binW,binV);

	    }
	}
    }
}

//--------------------------------------------------------
void BTagHLTOfflineSource::fillMEforTriggerNTfired(){

  int npath;
  if(&triggerResults_) {
    npath = triggerResults_->size();
  } else {
    return;
  }
  for(PathInfoCollection::iterator v = hltPathsAll_.begin(); v!= hltPathsAll_.end(); ++v )
    {
      unsigned index = triggerNames_.triggerIndex(v->getPath()); 
      if (index < triggerNames_.size() )
	{
	  v->getMEhisto_TriggerSummary()->Fill(0.);

	  edm::InputTag l1Tag(v->getl1Path(),"",processname_);
	  const int l1Index = triggerObj_->filterIndex(l1Tag);
	  bool l1found = false;

	  if ( l1Index < triggerObj_->sizeFilters() ) l1found = true;
	  if(!l1found)v->getMEhisto_TriggerSummary()->Fill(1.);
	  if(!l1found && !(triggerResults_->accept(index)))v->getMEhisto_TriggerSummary()->Fill(2.);
	  if(!l1found && (triggerResults_->accept(index)))v->getMEhisto_TriggerSummary()->Fill(3.);
	  if(l1found)v->getMEhisto_TriggerSummary()->Fill(4.);
	  if(l1found && (triggerResults_->accept(index)))v->getMEhisto_TriggerSummary()->Fill(5.); 
	  if(l1found && !(triggerResults_->accept(index)))v->getMEhisto_TriggerSummary()->Fill(6.);

	  if(!(triggerResults_->accept(index)) && l1found)
	    { 
	      if((v->getTriggerType().compare("BTagMu_Trigger") == 0) && (SelectedCaloJetsColl_.isValid()) && SelectedCaloJets->size())
		{
		  CaloJetCollection::const_iterator jet = SelectedCaloJets->begin();
		  v->getMEhisto_JetPt()->Fill(jet->pt());
		  v->getMEhisto_EtavsPt()->Fill(jet->eta(),jet->pt());
		  v->getMEhisto_PhivsPt()->Fill(jet->phi(),jet->pt());
                 
		}// BTagMu trigger is not fired

	    } // L1 is fired
	}//
    }// trigger not fired
 


}

//--------------------------------------------------------
void BTagHLTOfflineSource::fillMEforEffAllTrigger(const Event & iEvent){
  
  int npath;
  if(&triggerResults_) {
    npath = triggerResults_->size();
  } 
  else {
    return;
  }
  
  int num = -1;
  int denom = -1;

  for(PathInfoCollection::iterator v = hltPathsEff_.begin(); v!= hltPathsEff_.end(); ++v )
  {
    num++;
    denom++;

    bool denompassed = false;
    bool numpassed   = false; 
    
    unsigned indexNum = triggerNames_.triggerIndex(v->getPath());
    unsigned indexDenom = triggerNames_.triggerIndex(v->getDenomPath());
    
    if(indexNum < triggerNames_.size() && triggerResults_->accept(indexNum))       numpassed   = true;
    if(indexDenom < triggerNames_.size() && triggerResults_->accept(indexDenom))   denompassed = true; 
    
    if(denompassed)
      { //Denominator is fired
	
	if(SelectedCaloJetsColl_.isValid() && (v->getObjectType() == trigger::TriggerBJet)) {

	  if((v->getTriggerType().compare("BTagMu_Trigger") == 0) && SelectedCaloJets->size())
	    {
	      CaloJetCollection::const_iterator jet = SelectedCaloJets->begin();
	      
	      if (isMuonJet(*jet, SelectedMuons)){//mujet
		
		v->getMEhisto_DenominatorPt()->Fill(jet->pt());
		
		if (isBarrel(jet->eta()))  v->getMEhisto_DenominatorPtBarrel()->Fill(jet->pt());
		if (isEndCap(jet->eta()))  v->getMEhisto_DenominatorPtEndcap()->Fill(jet->pt());
		if (isForward(jet->eta())) v->getMEhisto_DenominatorPtForward()->Fill(jet->pt());
		
		v->getMEhisto_DenominatorEta()->Fill(jet->eta());
		v->getMEhisto_DenominatorPhi()->Fill(jet->phi());
		v->getMEhisto_DenominatorEtaPhi()->Fill(jet->eta(),jet->phi());             

	      }//mujet
	    }
	  
	}// Jet trigger and valid jet collection
	
	if (numpassed)
	  {//Numerator is fired
	    
	    if(SelectedCaloJetsColl_.isValid() && (v->getObjectType() == trigger::TriggerBJet)){

	      if((v->getTriggerType().compare("BTagMu_Trigger") == 0) && SelectedCaloJets->size())
		{
		  CaloJetCollection::const_iterator jet = SelectedCaloJets->begin();

		  if (isMuonJet(*jet, SelectedMuons)){//mujet
		    
		    v->getMEhisto_NumeratorPt()->Fill(jet->pt());
		    
		    if (isBarrel(jet->eta()))  v->getMEhisto_NumeratorPtBarrel()->Fill(jet->pt());
		    if (isEndCap(jet->eta()))  v->getMEhisto_NumeratorPtEndcap()->Fill(jet->pt());
		    if (isForward(jet->eta())) v->getMEhisto_NumeratorPtForward()->Fill(jet->pt());
		    
		    v->getMEhisto_NumeratorEta()->Fill(jet->eta());
		    v->getMEhisto_NumeratorPhi()->Fill(jet->phi());
		    v->getMEhisto_NumeratorEtaPhi()->Fill(jet->eta(),jet->phi());
		  }//mujet
		}
	      
	    }// Jet trigger and valid jet collection
	    
	  }//Numerator is fired
      }//Denominator is fired
    
  }// trigger under study

}

//--------------------------------------------------------
void BTagHLTOfflineSource::fillMEforEffWrtMuTrigger(const Event & iEvent){

  int npath;
  if(&triggerResults_) {
    npath = triggerResults_->size();
  } else {
    return;
  }
  bool muTrig = false;
  bool denompassed = false;
  for(size_t i=0;i<MuonTrigPaths_.size();++i){
    if(isHLTPathAccepted(MuonTrigPaths_[i])){
      muTrig = true;
      break;
    }
  }
  for(PathInfoCollection::iterator v = hltPathsEffWrtMu_.begin(); v!= hltPathsEffWrtMu_.end(); ++v )
  {
    bool numpassed   = false; 
    if(muTrig)denompassed = true;
     
    unsigned indexNum = triggerNames_.triggerIndex(v->getPath());
    if(indexNum < triggerNames_.size() && triggerResults_->accept(indexNum))numpassed   = true;

    if(denompassed){

      if(SelectedCaloJetsColl_.isValid() && (v->getObjectType() == trigger::TriggerBJet)){

	if((v->getTriggerType().compare("BTagMu_Trigger") == 0) && SelectedCaloJets->size())
	  {
	    CaloJetCollection::const_iterator jet = SelectedCaloJets->begin();
	    
	    if (isMuonJet(*jet, SelectedMuons)){//mujet

	      v->getMEhisto_DenominatorPt()->Fill(jet->pt());

	      if (isBarrel(jet->eta()))  v->getMEhisto_DenominatorPtBarrel()->Fill(jet->pt());
	      if (isEndCap(jet->eta()))  v->getMEhisto_DenominatorPtEndcap()->Fill(jet->pt());
	      if (isForward(jet->eta())) v->getMEhisto_DenominatorPtForward()->Fill(jet->pt());
	      
	      v->getMEhisto_DenominatorEta()->Fill(jet->eta());
	      v->getMEhisto_DenominatorPhi()->Fill(jet->phi());
	      v->getMEhisto_DenominatorEtaPhi()->Fill(jet->eta(),jet->phi());             
	      
	    }//mujet
	  }
            
      }// Jet trigger and valid jet collection

      if (numpassed)
      {
	if(SelectedCaloJetsColl_.isValid() && (v->getObjectType() == trigger::TriggerBJet)){
	  if((v->getTriggerType().compare("BTagMu_Trigger") == 0) && SelectedCaloJets->size())
	  {
	    CaloJetCollection::const_iterator jet = SelectedCaloJets->begin();

	    if (isMuonJet(*jet, SelectedMuons)){//mujet
	      
              v->getMEhisto_NumeratorPt()->Fill(jet->pt());

              if (isBarrel(jet->eta()))  v->getMEhisto_NumeratorPtBarrel()->Fill(jet->pt());
              if (isEndCap(jet->eta()))  v->getMEhisto_NumeratorPtEndcap()->Fill(jet->pt());
              if (isForward(jet->eta())) v->getMEhisto_NumeratorPtForward()->Fill(jet->pt());

              v->getMEhisto_NumeratorEta()->Fill(jet->eta());
              v->getMEhisto_NumeratorPhi()->Fill(jet->phi());
              v->getMEhisto_NumeratorEtaPhi()->Fill(jet->eta(),jet->phi());
	    }//mujet
	    
	  }
	}// Jet trigger and valid jet collection

      }//Numerator is fired
    }//Denominator is fired
  }// trigger under study


}

//--------------------------------------------------------
void BTagHLTOfflineSource::fillMEforEffWrtMBTrigger(const Event & iEvent){

  int npath;
  if(&triggerResults_) {
    npath = triggerResults_->size();
  } else {
    return;
  }
  bool mbTrig = false;
  bool denompassed = false;
  for(size_t i=0;i<MBTrigPaths_.size();++i){
    if(isHLTPathAccepted(MBTrigPaths_[i])){
      mbTrig = true;
      break;
    }
  }
  for(PathInfoCollection::iterator v = hltPathsEffWrtMB_.begin(); v!= hltPathsEffWrtMB_.end(); ++v )
  {
    bool numpassed   = false; 
    if(mbTrig)denompassed = true;

    unsigned indexNum = triggerNames_.triggerIndex(v->getPath());
    if(indexNum < triggerNames_.size() && triggerResults_->accept(indexNum))numpassed   = true;

    if(denompassed){

      if(SelectedCaloJetsColl_.isValid() && (v->getObjectType() == trigger::TriggerBJet)){
	if((v->getTriggerType().compare("BTagMu_Trigger") == 0) && SelectedCaloJets->size()) 
	  {
	    CaloJetCollection::const_iterator jet = SelectedCaloJets->begin();
	    if (isMuonJet(*jet, SelectedMuons)){//mujet

	      v->getMEhisto_DenominatorPt()->Fill(jet->pt());

	      if (isBarrel(jet->eta()))  v->getMEhisto_DenominatorPtBarrel()->Fill(jet->pt());
	      if (isEndCap(jet->eta()))  v->getMEhisto_DenominatorPtEndcap()->Fill(jet->pt());
	      if (isForward(jet->eta())) v->getMEhisto_DenominatorPtForward()->Fill(jet->pt());

	      v->getMEhisto_DenominatorEta()->Fill(jet->eta());
	      v->getMEhisto_DenominatorPhi()->Fill(jet->phi());
	      v->getMEhisto_DenominatorEtaPhi()->Fill(jet->eta(),jet->phi());             
	    }//mujet
	}

      }// Jet trigger and valid jet collection

      if (numpassed)
      {
	if(SelectedCaloJetsColl_.isValid() && (v->getObjectType() == trigger::TriggerBJet)) {
	  if((v->getTriggerType().compare("BTagMu_Trigger") == 0) && SelectedCaloJets->size())
	    {
	      CaloJetCollection::const_iterator jet = SelectedCaloJets->begin();
	      if (isMuonJet(*jet, SelectedMuons)){//mujet
		
		v->getMEhisto_NumeratorPt()->Fill(jet->pt());
		if (isBarrel(jet->eta()))  v->getMEhisto_NumeratorPtBarrel()->Fill(jet->pt());
		if (isEndCap(jet->eta()))  v->getMEhisto_NumeratorPtEndcap()->Fill(jet->pt());
		if (isForward(jet->eta())) v->getMEhisto_NumeratorPtForward()->Fill(jet->pt());
		v->getMEhisto_NumeratorEta()->Fill(jet->eta());
		v->getMEhisto_NumeratorPhi()->Fill(jet->phi());
		v->getMEhisto_NumeratorEtaPhi()->Fill(jet->eta(),jet->phi());

	      }//mujet
	    }
	  
	}// Jet trigger and valid jet collection

      }//Numerator is fired
    }//Denominator is fired
  }// trigger under study


}

// -- method called once each job just before starting event loop  --------
//--------------------------------------------------------
void BTagHLTOfflineSource::beginJob(){
 
}

//--------------------------------------------------------
// BeginRun
void BTagHLTOfflineSource::beginRun(const edm::Run& run, const edm::EventSetup& c){

  if(!isSetup_)
    { 
      DQMStore *dbe = 0;
      dbe = Service<DQMStore>().operator->();
      if (dbe) {
	dbe->setCurrentFolder(dirname_);
	dbe->rmdir(dirname_);
      }
      if (dbe) {
	dbe->setCurrentFolder(dirname_);
      }
      
      //--- htlConfig_
      bool changed(true);
      if (!hltConfig_.init(run, c, processname_, changed)) {
	LogDebug("HLTBTagDQMSource") << "HLTConfigProvider failed to initialize.";
      }
      
      /*
	Here we select the BTagMu Jets. BTagMu triggers are saved under same object type "TriggerBJet". 
	For the first trigger in the list, denominator trigger is dummy (empty) whereas for 
	other triggers denom is previous trigger of same type. e.g. DiJet20U has DiJet10U as denominator.
	For defining histos wrt muon trigger, denominator is always set "MuonTrigger". This string later can be checked and condition 
	can be applied on muon triggers.
      */

      const unsigned int n(hltConfig_.size());

      int btagmuJet = 0;
      
      for (unsigned int i=0; i!=n; ++i) { //Loop over paths

	bool denomFound = false;
	bool numFound = false; 
	bool mbFound = false;
	bool muFound = false; 

	std::string pathname = hltConfig_.triggerName(i);
	if(verbose_)cout << "==pathname==" << pathname << endl;

	std::string dpathname = MuonTrigPaths_[0];
	std::string l1pathname = "dummy";
	std::string denompathname = "";
	unsigned int usedPrescale = 1;
	unsigned int objectType = 0;
	std::string triggerType = "";
	std::string filtername("dummy");
	std::string Denomfiltername("denomdummy");
	
	if (pathname.find("BTagMu") != std::string::npos) {
	  triggerType = "BTagMu_Trigger";
	  objectType = trigger::TriggerBJet;
	}
	
	if( objectType == trigger::TriggerBJet  && !(pathname.find("BTagIP") != std::string::npos) )
	  {
	    btagmuJet++;
	    if(btagmuJet > 1)   dpathname = dpathname = hltConfig_.triggerName(i-1);
	    if(btagmuJet == 1)  dpathname = MuonTrigPaths_[0];
	  }
	
	// find L1 condition for numpath with numpath object type 
	// find PSet for L1 global seed for numpath, 
	// list module labels for numpath
        // Checking if the trigger exist in HLT table or not

	for (unsigned int i=0; i!=n; ++i) {
          std::string HLTname = hltConfig_.triggerName(i);
          if(HLTname == pathname)           numFound = true;
          if(HLTname == dpathname)          denomFound = true;
          if(HLTname == MBTrigPaths_[0])    mbFound = true;
          if(HLTname == MuonTrigPaths_[0])  muFound = true; 
        }
 
	if(numFound)//check whether the trigger exists in the menu
	  {
	    std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(pathname);

	    for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin(); numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
	      edm::InputTag testTag(*numpathmodule,"",processname_);
	      if ((hltConfig_.moduleType(*numpathmodule) == "HLT1CaloBJet") || (hltConfig_.moduleType(*numpathmodule) == "HLTPrescaler") ) filtername = *numpathmodule;
	      if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed") l1pathname = *numpathmodule;
	      
	    }
	  }

	if(objectType != 0 && denomFound)
	  {
	    std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(dpathname);
	    for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin(); numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
	      edm::InputTag testTag(*numpathmodule,"",processname_);
	      if ((hltConfig_.moduleType(*numpathmodule) == "HLT1CaloBJet") || (hltConfig_.moduleType(*numpathmodule) == "HLTPrescaler") ) Denomfiltername = *numpathmodule;
	    }
	  }

	if(objectType != 0 && numFound)
	  {
	    if(verbose_)
	      cout<<"==pathname=="<<pathname<<"==denompath=="<<dpathname<<"==filtername=="<<filtername<<"==denomfiltername=="<<Denomfiltername<<"==l1pathname=="<<l1pathname<<"==objectType=="<<objectType<<endl;    
	    
	    if(!((pathname.find("BTagIP") != std::string::npos) || (pathname.find("Quad") != std::string::npos)))
	      {     
		hltPathsAll_.push_back(PathInfo(usedPrescale, dpathname, pathname, l1pathname, filtername, Denomfiltername, processname_, objectType, triggerType));
		if(muFound)
		  hltPathsAllWrtMu_.push_back(PathInfo(usedPrescale, dpathname, pathname, l1pathname, filtername, Denomfiltername, processname_, objectType, triggerType));
		if(muFound)
		  hltPathsEffWrtMu_.push_back(PathInfo(usedPrescale, dpathname, pathname, l1pathname, filtername, Denomfiltername, processname_, objectType, triggerType));
		if(mbFound)
		  hltPathsEffWrtMB_.push_back(PathInfo(usedPrescale, dpathname, pathname, l1pathname, filtername, Denomfiltername, processname_, objectType, triggerType));
		if(!nameForEff_ && denomFound) 
		  hltPathsEff_.push_back(PathInfo(usedPrescale, dpathname, pathname, l1pathname, filtername, Denomfiltername, processname_, objectType, triggerType));
	      }
	    
	    hltPathsAllTriggerSummary_.push_back(PathInfo(usedPrescale, dpathname, pathname, l1pathname, filtername, Denomfiltername, processname_, objectType, triggerType));

	  }
      } //Loop over paths

      //---------bool to pick trigger names pair from config file-------------
      if(nameForEff_)
	{
	  std::string l1pathname = "dummy";
	  std::string denompathname = "";
	  unsigned int usedPrescale = 1;
	  unsigned int objectType = 0;
	  std::string triggerType = "";
	  std::string filtername("dummy");
	  std::string Denomfiltername("denomdummy");

	  for (std::vector<std::pair<std::string, std::string> >::iterator custompathnamepair = custompathnamepairs_.begin(); custompathnamepair != custompathnamepairs_.end(); ++custompathnamepair)
	    {
	      std::string pathname  = custompathnamepair->first;
	      std::string dpathname = custompathnamepair->second;
	      bool numFound = false;
	      bool denomFound = false;
	      // Check whether the trigger exist in HLT table or not
	      for (unsigned int i=0; i!=n; ++i) {
		std::string HLTname = hltConfig_.triggerName(i);
		if(HLTname == pathname)   numFound = true;
		if(HLTname == dpathname)  denomFound = true;
	      }
	      if(numFound && denomFound)
		{
		  if (pathname.find("BTagMu") != std::string::npos){
		    triggerType = "BTagMu_Trigger";
		    objectType = trigger::TriggerBJet;
		  }
		  std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(pathname);
		  for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin(); numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
		    edm::InputTag testTag(*numpathmodule,"",processname_);
		    if ((hltConfig_.moduleType(*numpathmodule) == "HLT1CaloBJet") || (hltConfig_.moduleType(*numpathmodule) == "HLTPrescaler") ) filtername = *numpathmodule;
		    if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed")l1pathname = *numpathmodule;
		  }

		  if(objectType != 0)
		    {
		      std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(dpathname);
		      for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin(); numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
			edm::InputTag testTag(*numpathmodule,"",processname_);
			if ((hltConfig_.moduleType(*numpathmodule) == "HLT1CaloBJet") || (hltConfig_.moduleType(*numpathmodule) == "HLTPrescaler") ) Denomfiltername = *numpathmodule;
		      }
     
		      if(verbose_)
			cout<<"==pathname=="<<pathname<<"==denompath=="<<dpathname<<"==filtername=="<<filtername<<"==denomfiltername=="<<Denomfiltername<<"==l1pathname=="<<l1pathname<<"==objectType=="<<objectType<<endl;
		      hltPathsEff_.push_back(PathInfo(usedPrescale, dpathname, pathname, l1pathname, filtername, Denomfiltername, processname_, objectType, triggerType)); 

		    }
		}
	    }
	}
      
      //-----------------------------------------------------------------
      //---book trigger summary histos
      if(!isSetup_)
	{
	  std::string foldernm = "/TriggerSummary/";
	  if (dbe)   {
	    dbe->setCurrentFolder(dirname_ + foldernm);
	  }
	  int     TrigBins_ = hltPathsAllTriggerSummary_.size();
	  double  TrigMin_  = -0.5;
	  double  TrigMax_  = hltPathsAllTriggerSummary_.size()-0.5;
	  
	  std::string histonm = "BTagMu_TriggerRate";
	  std::string histot = "BTagMu TriggerRate Summary";
     
	  rate_All = dbe->book1D(histonm.c_str(),histot.c_str(),TrigBins_,TrigMin_,TrigMax_);
	  
	  histonm = "BTagMu_TriggerRate_Correlation";
	  histot = "BTagMu TriggerRate Correlation Summary;y&&!x;x&&y";

	  correlation_All = dbe->book2D(histonm.c_str(),histot.c_str(),TrigBins_,TrigMin_,TrigMax_,TrigBins_,TrigMin_,TrigMax_);


	  histonm = "BTagMu_TriggerRate_WrtMuTrigger";
	  histot = "BTagMu TriggerRate Summary Wrt Muon Trigger ";
    
	  rate_AllWrtMu = dbe->book1D(histonm.c_str(),histot.c_str(),TrigBins_,TrigMin_,TrigMax_);


	  histonm = "BTagMu_TriggerRate_Correlation_WrtMuTrigger";
	  histot = "BTagMu TriggerRate Correlation Summary Wrt Muon Trigger;y&&!x;x&&y";

	  correlation_AllWrtMu = dbe->book2D(histonm.c_str(),histot.c_str(),TrigBins_,TrigMin_,TrigMax_,TrigBins_,TrigMin_,TrigMax_);

	  histonm = "BTagMu_TriggerRate_WrtMBTrigger";
	  histot = "BTagMu TriggerRate Summary Wrt MB Trigger";

	  rate_AllWrtMB = dbe->book1D(histonm.c_str(),histot.c_str(),TrigBins_,TrigMin_,TrigMax_);


	  histonm = "BTagMu_TriggerRate_Correlation_WrtMBTrigger";
	  histot = "BTagMu TriggerRate Correlation Wrt MB Trigger;y&&!x;x&&y";

	  correlation_AllWrtMB = dbe->book2D(histonm.c_str(),histot.c_str(),TrigBins_,TrigMin_,TrigMax_,TrigBins_,TrigMin_,TrigMax_);
	  isSetup_ = true;

	}

      //---Set bin label
      for(PathInfoCollection::iterator v = hltPathsAllTriggerSummary_.begin(); v!= hltPathsAllTriggerSummary_.end(); ++v ){
	
	std::string labelnm("dummy");
	labelnm = v->getPath(); 
	int nbins = rate_All->getTH1()->GetNbinsX();

	for(int ibin=1; ibin<nbins+1; ibin++)
	  {
	    const char * binLabel = rate_All->getTH1()->GetXaxis()->GetBinLabel(ibin);
	    std::string binLabel_str = string(binLabel);
	    if(binLabel_str.compare(labelnm)==0)break;
	    if(binLabel[0]=='\0')
	      {
		rate_All->setBinLabel(ibin,labelnm);
		rate_AllWrtMu->setBinLabel(ibin,labelnm);
		rate_AllWrtMB->setBinLabel(ibin,labelnm);
		correlation_All->setBinLabel(ibin,labelnm,1);
		correlation_AllWrtMu->setBinLabel(ibin,labelnm,1);
		correlation_AllWrtMB->setBinLabel(ibin,labelnm,1);
		correlation_All->setBinLabel(ibin,labelnm,2);
		correlation_AllWrtMu->setBinLabel(ibin,labelnm,2);
		correlation_AllWrtMB->setBinLabel(ibin,labelnm,2);
		break; 
	      } 
	  }     
	
      }

      //-------Now Efficiency histos--------
      if(plotEff_)
	{//plotEff_
	  
	  int Ptbins_      = 100;
	  int Etabins_     = 40;
	  int Phibins_     = 35;
	  double PtMin_    = 0.;
	  double PtMax_    = 200.;
	  double EtaMin_   = -5.;
	  double EtaMax_   =  5.;
	  double PhiMin_   = -3.14159;
	  double PhiMax_   =  3.14159;
	  
	  // Now define histos wrt lower threshold trigger
	  std::string dirName1 = dirname_ + "/RelativeTriggerEff/";
	  for(PathInfoCollection::iterator v = hltPathsEff_.begin(); v!= hltPathsEff_.end(); ++v ){
	    std::string labelname("ME") ;
	    std::string subdirName = dirName1 + v->getPath() + "_wrt_" + v->getDenomPath();
	    dbe->setCurrentFolder(subdirName);
	    std::string histoname(labelname+"");
	    std::string title(labelname+"");
	    
	    MonitorElement *dummy;
	    dummy =  dbe->bookFloat("dummy");   
	    
	    if((v->getObjectType() == trigger::TriggerBJet) && (v->getTriggerType().compare("BTagMu_Trigger") == 0))
	      {//Loop over BTagMu trigger
		
		histoname = labelname + "_NumeratorPt";
		title     = labelname + "NumeratorPt;Calo Pt[GeV/c]";
		MonitorElement * NumeratorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
		TH1 * h = NumeratorPt->getTH1();

		
		histoname = labelname + "_NumeratorPtBarrel";
		title     = labelname + "NumeratorPtBarrel;Calo Pt[GeV/c] ";
		MonitorElement * NumeratorPtBarrel =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
		h = NumeratorPtBarrel->getTH1();

		
		histoname = labelname + "_NumeratorPtEndcap";
		title     = labelname + "NumeratorPtEndcap;Calo Pt[GeV/c]";
		MonitorElement * NumeratorPtEndcap =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
		h = NumeratorPtEndcap->getTH1();

		
		histoname = labelname + "_NumeratorPtForward";
		title     = labelname + "NumeratorPtForward;Calo Pt[GeV/c]";
		MonitorElement * NumeratorPtForward =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
		h = NumeratorPtForward->getTH1();

		
		histoname = labelname + "_NumeratorEta";
		title     = labelname + "NumeratorEta;Calo #eta ";
		MonitorElement * NumeratorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
		h = NumeratorEta->getTH1();

		
		histoname = labelname + "_NumeratorPhi";
		title     = labelname + "NumeratorPhi;Calo #Phi";
		MonitorElement * NumeratorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
		h = NumeratorPhi->getTH1();

		
		histoname = labelname + "_NumeratorEtaPhi";
		title     = labelname + "NumeratorEtaPhi;Calo #eta;Calo #Phi";
		MonitorElement * NumeratorEtaPhi =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
		h = NumeratorEtaPhi->getTH1();

		
		histoname = labelname + "_DenominatorPt";
		title     = labelname + "DenominatorPt;Calo Pt[GeV/c]";
		MonitorElement * DenominatorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
		h = DenominatorPt->getTH1();

		
		histoname = labelname + "_DenominatorPtBarrel";
		title     = labelname + "DenominatorPtBarrel;Calo Pt[GeV/c]";
		MonitorElement * DenominatorPtBarrel =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
		h = DenominatorPtBarrel->getTH1();

		
		histoname = labelname + "_DenominatorPtEndcap";
		title     = labelname + "DenominatorPtEndcap;Calo Pt[GeV/c]";
		MonitorElement * DenominatorPtEndcap =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
		h = DenominatorPtEndcap->getTH1();

		
		histoname = labelname + "_DenominatorPtForward";
		title     = labelname + "DenominatorPtForward;Calo Pt[GeV/c] ";
		MonitorElement * DenominatorPtForward =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
		h = DenominatorPtForward->getTH1();

		
		histoname = labelname + "_DenominatorEta";
		title     = labelname + "DenominatorEta;Calo #eta ";
		MonitorElement * DenominatorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
		h = DenominatorEta->getTH1();

		
		histoname = labelname + "_DenominatorPhi";
		title     = labelname + "DenominatorPhi;Calo #Phi";
		MonitorElement * DenominatorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
		h = DenominatorPhi->getTH1();

		
		histoname = labelname + "_DenominatorEtaPhi";
		title     = labelname + "DenominatorEtaPhi;Calo #eta; Calo #Phi";
		MonitorElement * DenominatorEtaPhi =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
		h = DenominatorEtaPhi->getTH1();

		
		
		v->setEffHistos(  NumeratorPt,  NumeratorPtBarrel, NumeratorPtEndcap, NumeratorPtForward, NumeratorEta, NumeratorPhi, NumeratorEtaPhi,
				  DenominatorPt,  DenominatorPtBarrel, DenominatorPtEndcap, DenominatorPtForward, DenominatorEta, DenominatorPhi, DenominatorEtaPhi);

	      }// Loop over BTagMu Trigger
	    
	  }

	  //------Efficiency wrt Muon trigger-----------------------
	  std::string dirName2 = dirname_ + "/EffWrtMuonTrigger/";
	  for(PathInfoCollection::iterator v = hltPathsEffWrtMu_.begin(); v!= hltPathsEffWrtMu_.end(); ++v ){
	    std::string labelname("ME") ;
	    std::string subdirName = dirName2 + v->getPath();
	    std::string histoname(labelname+"");
	    std::string title(labelname+"");
	    dbe->setCurrentFolder(subdirName);
	    
	    MonitorElement *dummy;
	    dummy =  dbe->bookFloat("dummy");
	    if((v->getObjectType() == trigger::TriggerBJet) && (v->getTriggerType().compare("BTagMu_Trigger") == 0))
	      {// Loop over BTagMu Trigger

		histoname = labelname+"_NumeratorPt";
		title     = labelname+"NumeratorPt;Pt[GeV/c]";
		MonitorElement * NumeratorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
		TH1 * h = NumeratorPt->getTH1();

		
		histoname = labelname+"_NumeratorPtBarrel";
		title     = labelname+"NumeratorPtBarrel;Calo Pt[GeV/c]";
		MonitorElement * NumeratorPtBarrel =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
		h = NumeratorPtBarrel->getTH1();

		
		histoname = labelname+"_NumeratorPtEndcap";
		title     = labelname+"NumeratorPtEndcap;Calo Pt[GeV/c]";
		MonitorElement * NumeratorPtEndcap =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
		h = NumeratorPtEndcap->getTH1();

		
		histoname = labelname+"_NumeratorPtForward";
		title     = labelname+"NumeratorPtForward;Calo Pt[GeV/c]";
		MonitorElement * NumeratorPtForward =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
		h = NumeratorPtForward->getTH1();

		
		histoname = labelname+"_NumeratorEta";
		title     = labelname+"NumeratorEta;Calo #eta ";
		MonitorElement * NumeratorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
		h = NumeratorEta->getTH1();

		
		histoname = labelname+"_NumeratorPhi";
		title     = labelname+"NumeratorPhi;Calo #Phi";
		MonitorElement * NumeratorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
		h = NumeratorPhi->getTH1();

		
		histoname = labelname+"_NumeratorEtaPhi";
		title     = labelname+"NumeratorEtaPhi;Calo #eta;Calo #Phi";
		MonitorElement * NumeratorEtaPhi =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
		h = NumeratorEtaPhi->getTH1();

		
		histoname = labelname+"_DenominatorPt";
		title     = labelname+"DenominatorPt;Calo Pt[GeV/c]";
		MonitorElement * DenominatorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
		h = DenominatorPt->getTH1();

		
		histoname = labelname+"_DenominatorPtBarrel";
		title     = labelname+"DenominatorPtBarrel;Calo Pt[GeV/c]";
		MonitorElement * DenominatorPtBarrel =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
		h = DenominatorPtBarrel->getTH1();

		
		histoname = labelname+"_DenominatorPtEndcap";
		title     = labelname+"DenominatorPtEndcap;Calo Pt[GeV/c]";
		MonitorElement * DenominatorPtEndcap =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
		h = DenominatorPtEndcap->getTH1();

		
		histoname = labelname+"_DenominatorPtForward";
		title     = labelname+"DenominatorPtForward;Calo Pt[GeV/c] ";
		MonitorElement * DenominatorPtForward =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
		h = DenominatorPtForward->getTH1();

		
		histoname = labelname+"_DenominatorEta";
		title     = labelname+"DenominatorEta;Calo #eta";
		MonitorElement * DenominatorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
		h = DenominatorEta->getTH1();

		
		histoname = labelname+"_DenominatorPhi";
		title     = labelname+"DenominatorPhi;Calo #Phi";
		MonitorElement * DenominatorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
		h = DenominatorPhi->getTH1();

		
		histoname = labelname+"_DenominatorEtaPhi";
		title     = labelname+"DenominatorEtaPhi;Calo #eta (IC5);Calo #Phi ";
		MonitorElement * DenominatorEtaPhi =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
		h = DenominatorEtaPhi->getTH1();

		
		
		v->setEffHistos(  NumeratorPt,  NumeratorPtBarrel, NumeratorPtEndcap, NumeratorPtForward, NumeratorEta, NumeratorPhi, NumeratorEtaPhi,
				  DenominatorPt,  DenominatorPtBarrel, DenominatorPtEndcap, DenominatorPtForward, DenominatorEta, DenominatorPhi, DenominatorEtaPhi);

	      }// Loop over BTagMu Trigger
	    
	  }
	  
	  //--------Efficiency  wrt MiniBias trigger---------
	  std::string dirName3  = dirname_ + "/EffWrtMBTrigger/";
	  for(PathInfoCollection::iterator v = hltPathsEffWrtMB_.begin(); v!= hltPathsEffWrtMB_.end(); ++v ){
	    std::string labelname("ME") ;
	    std::string subdirName = dirName3 + v->getPath() ;
	    std::string histoname(labelname+"");
	    std::string title(labelname+"");
	    dbe->setCurrentFolder(subdirName);
	    MonitorElement *dummy;
	    dummy =  dbe->bookFloat("dummy");   

	    if((v->getObjectType() == trigger::TriggerBJet) && (v->getTriggerType().compare("BTagMu_Trigger") == 0))
	      { // Loop over BTagMu Trigger
		
		histoname = labelname+"_NumeratorPt";
		title     = labelname+"NumeratorPt;Calo Pt[GeV/c] ";
		MonitorElement * NumeratorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
		TH1 * h = NumeratorPt->getTH1();

		
		histoname = labelname+"_NumeratorPtBarrel";
		title     = labelname+"NumeratorPtBarrel;Calo Pt[GeV/c]";
		MonitorElement * NumeratorPtBarrel =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
		h = NumeratorPtBarrel->getTH1();

		
		histoname = labelname+"_NumeratorPtEndcap";
		title     = labelname+"NumeratorPtEndcap; Calo Pt[GeV/c] ";
		MonitorElement * NumeratorPtEndcap =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
		h = NumeratorPtEndcap->getTH1();

		
		histoname = labelname+"_NumeratorPtForward";
		title     = labelname+"NumeratorPtForward;Calo Pt[GeV/c]";
		MonitorElement * NumeratorPtForward =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
		h = NumeratorPtForward->getTH1();

		
		histoname = labelname+"_NumeratorEta";
		title     = labelname+"NumeratorEta;Calo #eta ";
		MonitorElement * NumeratorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
		h = NumeratorEta->getTH1();

		
		histoname = labelname+"_NumeratorPhi";
		title     = labelname+"NumeratorPhi;Calo #Phi";
		MonitorElement * NumeratorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
		h = NumeratorPhi->getTH1();

		
		histoname = labelname+"_NumeratorEtaPhi";
		title     = labelname+"NumeratorEtaPhi;Calo #eta;Calo #Phi ";
		MonitorElement * NumeratorEtaPhi =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
		h = NumeratorEtaPhi->getTH1();

		
		histoname = labelname+"_DenominatorPt";
		title     = labelname+"DenominatorPt;Calo Pt[GeV/c]";
		MonitorElement * DenominatorPt =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
		h = DenominatorPt->getTH1();

		
		histoname = labelname+"_DenominatorPtBarrel";
		title     = labelname+"DenominatorPtBarrel;Calo Pt[GeV/c]";
		MonitorElement * DenominatorPtBarrel =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
		h = DenominatorPtBarrel->getTH1();

		
		histoname = labelname+"_DenominatorPtEndcap";
		title     = labelname+"DenominatorPtEndcap;Calo Pt[GeV/c]";
		MonitorElement * DenominatorPtEndcap =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
		h = DenominatorPtEndcap->getTH1();

		
		histoname = labelname+"_DenominatorPtForward";
		title     = labelname+"DenominatorPtForward;Calo Pt[GeV/c]";
		MonitorElement * DenominatorPtForward =  dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
		h = DenominatorPtForward->getTH1();

		
		histoname = labelname+"_DenominatorEta";
		title     = labelname+"DenominatorEta;Calo #eta ";
		MonitorElement * DenominatorEta =  dbe->book1D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_);
		h = DenominatorEta->getTH1();

		
		histoname = labelname+"_DenominatorPhi";
		title     = labelname+"DenominatorPhi;Calo #Phi";
		MonitorElement * DenominatorPhi =  dbe->book1D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_);
		h = DenominatorPhi->getTH1();

		
		histoname = labelname+"_DenominatorEtaPhi";
		title     = labelname+"DenominatorEtaPhi;Calo #eta ;Calo #Phi ";
		MonitorElement * DenominatorEtaPhi =  dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Phibins_,PhiMin_,PhiMax_);
		h = DenominatorEtaPhi->getTH1();

		
		
		v->setEffHistos(  NumeratorPt,  NumeratorPtBarrel, NumeratorPtEndcap, NumeratorPtForward, NumeratorEta, NumeratorPhi, NumeratorEtaPhi,
				  DenominatorPt,  DenominatorPtBarrel, DenominatorPtEndcap, DenominatorPtForward, DenominatorEta, DenominatorPhi, DenominatorEtaPhi);
		
	      }// Loop over BTagMu Trigger
	    
	  }
	  
	}// This is loop over all efficiency plots
      
      //--------Histos to see WHY trigger is NOT fired----------
      /*
      int Nbins_       = 10;
      int Nmin_        = 0;
      int Nmax_        = 10;
      */
      int Ptbins_      = 100;
      int Etabins_     = 40;
      int Phibins_     = 35;
      double PtMin_    = 0.;
      double PtMax_    = 200.;
      double EtaMin_   = -5.;
      double EtaMax_   =  5.;
      double PhiMin_   = -3.14159;
      double PhiMax_   =  3.14159;
      
      std::string dirName4_ = dirname_ + "/TriggerNotFired/";
      for(PathInfoCollection::iterator v = hltPathsAll_.begin(); v!= hltPathsAll_.end(); ++v ){
	
	MonitorElement *dummy;
	dummy =  dbe->bookFloat("dummy");
	
	std::string labelname("ME") ;
	std::string histoname(labelname+"");
	std::string title(labelname+"");
	dbe->setCurrentFolder(dirName4_ + v->getPath());
	
	histoname = labelname+"_TriggerSummary";
	title     = labelname+"Summary of trigger levels"; 
	MonitorElement * TriggerSummary = dbe->book1D(histoname.c_str(),title.c_str(),7, -0.5,6.5);
	
	std::vector<std::string> trigger;
	trigger.push_back("Nevt");
	trigger.push_back("L1 failed");
	trigger.push_back("L1 & HLT failed");
	trigger.push_back("L1 failed but not HLT");
	trigger.push_back("L1 passed");
	trigger.push_back("L1 & HLT passed");
	trigger.push_back("L1 passed but not HLT");

	for(unsigned int i =0; i < trigger.size(); i++) TriggerSummary->setBinLabel(i+1, trigger[i]);
	
	if((v->getTriggerType().compare("BTagMu_Trigger") == 0))
	  {// BTagMu trigger
	    
	    histoname = labelname+"_JetPt"; 
	    title     = labelname+"Leading jet pT;Pt[GeV/c]";
	    MonitorElement * JetPt = dbe->book1D(histoname.c_str(),title.c_str(),Ptbins_,PtMin_,PtMax_);
	    TH1 * h = JetPt->getTH1();

	    
	    histoname = labelname+"_JetEtaVsPt";
	    title     = labelname+"Leading jet #eta vs pT;#eta;Pt[GeV/c]";
	    MonitorElement * JetEtaVsPt = dbe->book2D(histoname.c_str(),title.c_str(),Etabins_,EtaMin_,EtaMax_,Ptbins_,PtMin_,PtMax_);
	    h = JetEtaVsPt->getTH1();

	    
	    histoname = labelname+"_JetPhiVsPt";
	    title     = labelname+"Leading jet #Phi vs pT;#Phi;Pt[GeV/c]";
	    MonitorElement * JetPhiVsPt = dbe->book2D(histoname.c_str(),title.c_str(),Phibins_,PhiMin_,PhiMax_,Ptbins_,PtMin_,PtMax_);
	    h = JetPhiVsPt->getTH1();

	    
	    v->setDgnsHistos( TriggerSummary, dummy, JetPt, JetEtaVsPt, JetPhiVsPt, dummy, dummy, dummy, dummy, dummy, dummy); 

	  }// BTagMu trigger  
	
      }
      
    }  
}

//--------------------------------------------------------
void BTagHLTOfflineSource::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
						const EventSetup& context) {
}
//--------------------------------------------------------
void BTagHLTOfflineSource::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
					      const EventSetup& context) {
}

//--------------------------------------------------------
// - method called once each job just after ending the event loop  ------------
void BTagHLTOfflineSource::endJob() {
  delete jetID;
}

//--------------------------------------------------------
// EndRun
void BTagHLTOfflineSource::endRun(const edm::Run& run, const edm::EventSetup& c){
  if (verbose_) std::cout << "endRun, run " << run.id() << std::endl;
}

//--------------------------------------------------------
bool BTagHLTOfflineSource::isBarrel(double eta){
  bool output = false;
  if (fabs(eta)<=1.3) output=true;
  return output;
}

//--------------------------------------------------------
bool BTagHLTOfflineSource::isEndCap(double eta){
  bool output = false;
  if (fabs(eta)<=3.0 && fabs(eta)>1.3) output=true;
  return output;
}

//--------------------------------------------------------
bool BTagHLTOfflineSource::isForward(double eta){
  bool output = false;
  if (fabs(eta)>3.0) output=true;
  return output;
}

//--------------------------------------------------------
bool BTagHLTOfflineSource::validPathHLT(std::string pathname){
  // hltConfig_ has to be defined first before calling this method
  bool output=false;
  for (unsigned int j=0; j!=hltConfig_.size(); ++j) {
    if (hltConfig_.triggerName(j) == pathname )
      output=true;
  }
  return output;
}

//--------------------------------------------------------
bool BTagHLTOfflineSource::isHLTPathAccepted(std::string pathName){
  // triggerResults_, triggerNames_ has to be defined first before calling this method
  bool output=false;
  if(&triggerResults_) {
    unsigned index = triggerNames_.triggerIndex(pathName);
    if(index < triggerNames_.size() && triggerResults_->accept(index)) output = true;
  }
  return output;
}

//--------------------------------------------------------
// This returns the position of trigger name defined in summary histograms
double BTagHLTOfflineSource::TriggerPosition(std::string trigName){
  int nbins = rate_All->getTH1()->GetNbinsX();
  double binVal = -100;
  for(int ibin=1; ibin<nbins+1; ibin++)
  {
    const char * binLabel = rate_All->getTH1()->GetXaxis()->GetBinLabel(ibin);
    if(binLabel[0]=='\0')continue;
    //       std::string binLabel_str = string(binLabel);
    //       if(binLabel_str.compare(trigName)!=0)continue;
    if(trigName.compare(binLabel)!=0)continue;

    if(trigName.compare(binLabel)==0){
      binVal = rate_All->getTH1()->GetBinCenter(ibin);
      break;
    }
  }
  return binVal;
}

//--------------------------------------------------------
bool BTagHLTOfflineSource::isTriggerObjectFound(std::string objectName){
  // processname_, triggerObj_ has to be defined before calling this method
  bool output=false;
  edm::InputTag testTag(objectName,"",processname_);
  const int index = triggerObj_->filterIndex(testTag);
  if ( index >= triggerObj_->sizeFilters() ) {    
    edm::LogInfo("BTagHLTOfflineSource") << "no index "<< index << " of that name ";
  } else {       
    const trigger::Keys & k = triggerObj_->filterKeys(index);
    if (k.size()) output=true;
  }
  return output;
}

//--------------------------------------------------------
void BTagHLTOfflineSource::selectMuons(const edm::Handle<reco::MuonCollection> & muonHandle)
{
  // for every event, first clear vector of selected objects
  SelectedMuons->clear();

  if(muonHandle.isValid()) { 

    for( reco::MuonCollection::const_iterator iter = muonHandle->begin(), iend = muonHandle->end(); iter != iend; ++iter )
      {
	
	if( iter->isGlobalMuon())// Global Muon
	  {
	    if(isVBTFMuon(*iter)) SelectedMuons->push_back(*iter);
	  }//Global Muon

      } // end for
  
    edm::Handle<reco::MuonCollection> localSelMuonsHandle(SelectedMuons, muonHandle.provenance());
    SelectedMuonsColl_ = localSelMuonsHandle;

  } // end if


}

//--------------------------------------------------------
bool BTagHLTOfflineSource::isVBTFMuon(const reco::Muon& muon)
{

  bool quality = 1;

  reco::TrackRef gm = muon.globalTrack();
  reco::TrackRef tk = muon.innerTrack();

  // Muon Quality cuts same as b-tag efficiency methods
  // --------------------------------------------------

  double mupt             = muon.pt();
  double mueta            = muon.eta();
  int muonHits            = gm->hitPattern().numberOfValidMuonHits();
  int nMatches            = muon.numberOfMatches();
  int trackerHits         = tk->hitPattern().numberOfValidHits();
  int pixelHits           = tk->hitPattern().numberOfValidPixelHits();
  int outerHits           = tk->trackerExpectedHitsOuter().numberOfHits();
  double tknormalizedChi2 = tk->normalizedChi2(); 
  double gmnormalizedChi2 = gm->normalizedChi2(); 

  // Must have BeamSpot
  if(!beamSpot_.isValid()) return 0;
  double mudZ             = muon.vz() - beamSpot_->z0(); //pv->z();
  
  /*
    #pt > 5. & abs(eta) < 2.4 & isGlobalMuon() & 
    # globalTrack().hitPattern().numberOfValidMuonHits() > 0 
    #& numberOfMatches() > 1 & 
    #innerTrack().numberOfValidHits()> 10 
    #& innerTrack().hitPattern().numberOfValidPixelHits()>1 
    #& innerTrack().trackerExpectedHitsOuter().numberOfHits() <3 
    #& innerTrack().normalizedChi2() < 10 & globalTrack().normalizedChi2() < 10
    #& muon.vz()‐primaryVertex.z() <2 & DR(muon,jet)<0.4
  */

  if (mupt < _mupt)                           {return 0; quality=0;}
  if (fabs(mueta) > _mueta)                   {return 0; quality=0;}
  if (muonHits < _muonHits)                   {return 0; quality=0;}
  if (nMatches < _nMatches)                   {return 0; quality=0;}
  if (trackerHits < _trackerHits)             {return 0; quality=0;}
  if (pixelHits < _pixelHits)                 {return 0; quality=0;}
  if (outerHits > _outerHits)                 {return 0; quality=0;}
  if (tknormalizedChi2 > _tknormalizedChi2)   {return 0; quality=0;}
  if (gmnormalizedChi2 > _gmnormalizedChi2)   {return 0; quality=0;}
  if (mudZ > _mudZ)                           {return 0; quality=0;}

  return true;

}

//--------------------------------------------------------
void BTagHLTOfflineSource::selectJets(const edm::Event& iEvent, const edm::Handle<reco::CaloJetCollection> & jetHandle)
{
  // for every event, first clear vector of selected objects
  SelectedCaloJets->clear();
  
  if(jetHandle.isValid()) { 
    
    //    std::stable_sort( jetHandle->begin(), jetHandle->end(), JetPtSorter() );
    
    for( reco::CaloJetCollection::const_iterator iter = jetHandle->begin(), iend = jetHandle->end(); iter != iend; ++iter )
      {
	jetID->calculate(iEvent, *iter);
	if ( iter->pt() > _jetpt && 
	     (iter->emEnergyFraction()>_fEMF || fabs(iter->eta()) > _jeteta) &&
	     jetID->fHPD() < _fHPD &&
	     iter->n90() >= _n90Hits 
	     )
	  { // apply jet cuts
	    SelectedCaloJets->push_back(*iter);
	  }
      } // end for
  
    edm::Handle<reco::CaloJetCollection> localSelJetsHandle(SelectedCaloJets, jetHandle.provenance());
    SelectedCaloJetsColl_ = localSelJetsHandle;
    
  } // end if
  
}

//--------------------------------------------------------
bool BTagHLTOfflineSource::isMuonJet(const reco::CaloJet& calojet, reco::MuonCollection *SelectedMuons)
{
  
  bool  isMuJet = false;
  
  if (SelectedMuons->size() != 0) 
    {//non empty muon collection
      
      for( reco::MuonCollection::const_iterator mu = SelectedMuons->begin(); mu != SelectedMuons->end(); ++mu )
	{
	  if( (deltaR(calojet.eta(), calojet.phi(), mu->eta(), mu->phi() )) < _mujetdR )
	    isMuJet = true;
	}

    }//non empty muon collection

  return isMuJet;
}

//--------------------------------------------------------

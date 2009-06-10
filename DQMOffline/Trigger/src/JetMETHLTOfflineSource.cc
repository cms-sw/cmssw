#include "DQMOffline/Trigger/interface/JetMETHLTOfflineSource.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Run.h"

#include <boost/algorithm/string.hpp>

JetMETHLTOfflineSource::JetMETHLTOfflineSource(const edm::ParameterSet& iConfig)
{

  dbe_ = edm::Service<DQMStore>().operator->();
  if (!dbe_) {
    edm::LogInfo("JetMETHLTOfflineSource") << "unable to get DQMStore service?";
  }
  if(iConfig.getUntrackedParameter<bool>("DQMStore", false)) {
    dbe_->setVerbose(0);
  }

  debug_ = false;
  verbose_ = false;
 
  //--- trigger objects
  triggerSummaryLabel_ = iConfig.getParameter<edm::InputTag>("triggerSummaryLabel");
  if (debug_) std::cout << triggerSummaryLabel_ << std::endl;
  
  //--- trigger bits
  triggerResultsLabel_ = iConfig.getParameter<edm::InputTag>("triggerResultsLabel");
  if (debug_) std::cout << triggerResultsLabel_ << std::endl;

  //--- trigger path names for getting efficiencies
  //--- single jets
  std::vector<std::string > hltPathsEff;
  hltPathsEff = iConfig.getParameter<std::vector<std::string > >("HLTPathsEffSingleJet");
  for (unsigned int i=0; i<hltPathsEff.size(); i++){
    if (debug_) std::cout << getNumeratorTrigger(hltPathsEff[i]) << std::endl;
    if (debug_) std::cout << getDenominatorTrigger(hltPathsEff[i]) << std::endl;
    HLTPathsEffSingleJet_.push_back(PathInfo(getDenominatorTrigger(hltPathsEff[i]),
					     getNumeratorTrigger(hltPathsEff[i]),
					     getTriggerEffLevel(hltPathsEff[i]),
					     getTriggerThreshold(hltPathsEff[i])));
  }

  //--- dijet average
  if (debug_) std::cout << "dijet ave" << std::endl; 
  hltPathsEff.clear();
  hltPathsEff = iConfig.getParameter<std::vector<std::string > >("HLTPathsEffDiJetAve");
  for (unsigned int i=0; i<hltPathsEff.size(); i++){
    HLTPathsEffDiJetAve_.push_back(PathInfo(getDenominatorTrigger(hltPathsEff[i]),
					    getNumeratorTrigger(hltPathsEff[i]),
					    getTriggerEffLevel(hltPathsEff[i]),
					    getTriggerThreshold(hltPathsEff[i])));
  }

  //--- met
  if (debug_) std::cout << "met" << std::endl; 
  hltPathsEff.clear();
  hltPathsEff = iConfig.getParameter<std::vector<std::string > >("HLTPathsEffMET");
  for (unsigned int i=0; i<hltPathsEff.size(); i++){
    HLTPathsEffMET_.push_back(PathInfo(getDenominatorTrigger(hltPathsEff[i]),
				       getNumeratorTrigger(hltPathsEff[i]),
				       getTriggerEffLevel(hltPathsEff[i]),
				       getTriggerThreshold(hltPathsEff[i])));
  }

  //--- mht
  if (debug_) std::cout << "mht" << std::endl; 
  hltPathsEff.clear();
  hltPathsEff = iConfig.getParameter<std::vector<std::string > >("HLTPathsEffMHT");
  for (unsigned int i=0; i<hltPathsEff.size(); i++){
    HLTPathsEffMHT_.push_back(PathInfo(getDenominatorTrigger(hltPathsEff[i]),
				       getNumeratorTrigger(hltPathsEff[i]),
				       getTriggerEffLevel(hltPathsEff[i]),
				       getTriggerThreshold(hltPathsEff[i])));
  }

  //--- trigger path names for more monitoring histograms
  std::vector<std::string > hltMonPaths;
  hltMonPaths = iConfig.getParameter<std::vector<std::string > >("HLTPathsMonSingleJet");
  for (unsigned int i=0; i<hltMonPaths.size(); i++)
    HLTPathsMonSingleJet_.push_back(PathHLTMonInfo(hltMonPaths[i]));

  hltMonPaths.clear();
  hltMonPaths = iConfig.getParameter<std::vector<std::string > >("HLTPathsMonDiJetAve");
  for (unsigned int i=0; i<hltMonPaths.size(); i++)
    HLTPathsMonDiJetAve_.push_back(PathHLTMonInfo(hltMonPaths[i]));

  hltMonPaths.clear();
  hltMonPaths = iConfig.getParameter<std::vector<std::string > >("HLTPathsMonMET");
  for (unsigned int i=0; i<hltMonPaths.size(); i++)
    HLTPathsMonMET_.push_back(PathHLTMonInfo(hltMonPaths[i]));

  hltMonPaths.clear();
  hltMonPaths = iConfig.getParameter<std::vector<std::string > >("HLTPathsMonMHT");
  for (unsigned int i=0; i<hltMonPaths.size(); i++)
    HLTPathsMonMHT_.push_back(PathHLTMonInfo(hltMonPaths[i]));

  //--- offline calo jets
  caloJetsTag_ = iConfig.getParameter<edm::InputTag>("CaloJetCollectionLabel");
  //iEvent.getByLabel(caloJetsTag_,calojetObj);

  //--- offline calo met
  caloMETTag_ = iConfig.getParameter<edm::InputTag>("CaloMETCollectionLabel");
  //iEvent.getByLabel(caloMETTag_, calometObj);

  processname_ = iConfig.getParameter<std::string>("processname");

  //--- HLT tag
  hltTag_ = iConfig.getParameter<std::string>("hltTag");
  if (debug_) std::cout << hltTag_ << std::endl;

  //--- DQM output folder name
  dirName_=iConfig.getParameter<std::string>("DQMDirName");
  if (debug_) std::cout << dirName_ << std::endl;

  l1ExtraTaus_ = iConfig.getParameter<edm::InputTag>("L1Taus");
  l1ExtraCJets_= iConfig.getParameter<edm::InputTag>("L1CJets");
  l1ExtraFJets_= iConfig.getParameter<edm::InputTag>("L1FJets");

  if (dbe_ != 0 ) {
    dbe_->setCurrentFolder(dirName_);
  }

}


JetMETHLTOfflineSource::~JetMETHLTOfflineSource()
{ 
}

void JetMETHLTOfflineSource::beginJob(const edm::EventSetup& iSetup)
{

  if (dbe_ != 0 ) {
    dbe_->setCurrentFolder(dirName_);
  }

  //the one monitor element the source fills directly
  dqmErrsMonElem_ =dbe_->book1D("dqmErrors","JetMETHLTOfflineSource Errors",101,-0.5,100.5);
  
}


void JetMETHLTOfflineSource::endJob() 
{
  //LogDebug("JetMETHLTOfflineSource") << "ending job";
}


void JetMETHLTOfflineSource::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  //LogDebug("JetMETHLTOfflineSource") << "beginRun, run " << run.id();

  //
  //--- htlConfig_
  if (!hltConfig_.init(processname_)) {
    processname_ = "FU";
    if (!hltConfig_.init(processname_)){
      LogDebug("JetMETHLTOffline") << "HLTConfigProvider failed to initialize.";
    }
  }

  if (verbose_){
    for (unsigned int j=0; j!=hltConfig_.size(); ++j) {
      cout << j << " " << hltConfig_.triggerName(j) << endl;      
    }
  }
  
  //
  //--- obtain the L1 and HLT module names
  HLTPathsEffSingleJet_ = fillL1andHLTModuleNames(HLTPathsEffSingleJet_,"HLTLevel1GTSeed","HLT1CaloJet");
  HLTPathsEffDiJetAve_  = fillL1andHLTModuleNames(HLTPathsEffDiJetAve_, "HLTLevel1GTSeed","HLTDiJetAveFilter");
  HLTPathsEffMET_       = fillL1andHLTModuleNames(HLTPathsEffMET_,      "HLTLevel1GTSeed","HLT1CaloMET");
 
  HLTPathsMonSingleJet_ = fillL1andHLTModuleNames(HLTPathsMonSingleJet_,"HLTLevel1GTSeed","HLT1CaloJet");
  HLTPathsMonDiJetAve_  = fillL1andHLTModuleNames(HLTPathsMonDiJetAve_, "HLTLevel1GTSeed","HLTDiJetAveFilter");
  HLTPathsMonMET_       = fillL1andHLTModuleNames(HLTPathsMonMET_,      "HLTLevel1GTSeed","HLT1CaloMET");

  //--- MonitorElement booking ---
  bookMEforEffSingleJet();
  bookMEforEffDiJetAve();
  bookMEforEffMET();
  bookMEforEffMHT();

  bookMEforMonSingleJet();
  bookMEforMonDiJetAve();
  bookMEforMonMET();
  bookMEforMonMHT();

}

void JetMETHLTOfflineSource::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  //LogDebug("JetMETHLTOfflineSource") << "endRun, run " << run.id();
}


void JetMETHLTOfflineSource::analyze(const edm::Event& iEvent,const edm::EventSetup& iSetup)
{ 

  if (debug_) std::cout << ">>>>>>new event start - " << std::endl;
  
  //const double weight=1.; //we have the ability to weight but its disabled for now
  
  int errCode = 1;
  dqmErrsMonElem_->Fill(errCode);
  
  //---------- triggerResults ----------
  if (debug_) std::cout << ">>>now triggerResults" << std::endl;
  iEvent.getByLabel(triggerResultsLabel_, triggerResults_);
  if(!triggerResults_.isValid()) {
    edm::InputTag triggerResultsLabelFU(triggerResultsLabel_.label(),triggerResultsLabel_.instance(), "FU");
   iEvent.getByLabel(triggerResultsLabelFU,triggerResults_);
  if(!triggerResults_.isValid()) {
    edm::LogInfo("FourVectorHLTOffline") << "TriggerResults not found, "
      "skipping event";
    return;
   }
  }

  int npath;
  if(&triggerResults_) {

    // Check how many HLT triggers are in triggerResults
    npath = triggerResults_->size();
    if (debug_) std::cout << "npath(triggerResults)=" << npath << std::endl;

    triggerNames_.init(*(triggerResults_.product()));

  } else {

    edm::LogInfo("CaloMETHLTOfflineSource") << "TriggerResults::HLT not found, "
      "automatically select events";
    return;

  }

  // did we pass each trigger path?
  if (debug_){
    std::cout << std::endl;
    // tirggerResults
    for(int i = 0; i < npath; ++i) {
      if (triggerNames_.triggerName(i).find("HLT_") != std::string::npos ){
	std::cout << i << " "
		  << triggerNames_.triggerName(i) << " "
		  << triggerResults_->wasrun(i) << " "
		  << triggerResults_->accept(i) << " "
		  << triggerResults_->error(i) << std::endl;
      }
    }
  }

  //---------- triggerSummary ----------
  if (debug_) std::cout << ">>>now triggerSummary" << std::endl;
  iEvent.getByLabel(triggerSummaryLabel_,triggerObj_); 
  if(!triggerObj_.isValid()) {
    edm::InputTag triggerSummaryLabelFU(triggerSummaryLabel_.label(),triggerSummaryLabel_.instance(), "FU");
   iEvent.getByLabel(triggerSummaryLabelFU,triggerObj_);
  if(!triggerObj_.isValid()) {
    edm::LogInfo("FourVectorHLTOffline") << "TriggerEvent not found, "
      "skipping event"; 
    return;
   }
  }

  const trigger::TriggerObjectCollection & toc(triggerObj_->getObjects());

  //--- Show everything
  if (debug_) {
    for ( size_t ia = 0; ia < triggerObj_->sizeFilters(); ++ ia) {
      std::string name = triggerObj_->filterTag(ia).encode();
      std::cout << ia << " " << name << std::endl;
      
      const trigger::Vids & idtype = triggerObj_->filterIds(ia);
      const trigger::Keys & k      = triggerObj_->filterKeys(ia);
      trigger::Vids::const_iterator idtypeiter = idtype.begin();
      for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {
	std::cout << toc[*ki].pt()  << " "
		  << toc[*ki].eta() << " "
		  << toc[*ki].phi() << std::endl;
	++idtypeiter;
      } // loop over different objects
    }   // loop over different paths
  }

  //--- Show one particular path
  if (debug_){
    edm::InputTag testTag("hltDiJetAve30U8E29","",processname_);
    const int index              = triggerObj_->filterIndex(testTag);
    
    if ( index >= triggerObj_->sizeFilters() ) {    
      edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< index << " of that name ";
      
    } else {       
      
      const trigger::Keys & k = triggerObj_->filterKeys(index);
      std::string name = triggerObj_->filterTag(index).encode();
      std::cout << name << std::endl;

      for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {
	std::cout << toc[*ki].pt()  << " "
		  << toc[*ki].eta() << " "
		  << toc[*ki].phi() << std::endl;
      }
      
    } // index
  }

  //--- Show one particular L1s path
  if (debug_){
    edm::InputTag testTag2("hltL1sJet50U","",processname_);
    const int index2              = triggerObj_->filterIndex(testTag2);

    if ( index2 >= triggerObj_->sizeFilters() ) {    
      edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< index2 << " of that name ";
      
    } else {       
      
      const trigger::Keys & k = triggerObj_->filterKeys(index2);
      std::string name = triggerObj_->filterTag(index2).encode();
      std::cout << name << std::endl;

      for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {
	std::cout << toc[*ki].pt()  << " "
		  << toc[*ki].eta() << " "
		  << toc[*ki].phi() << std::endl;
      }
    
    } // index2
  }
 
  //----------- l1extra jet particle collection -----

  edm::Handle<l1extra::L1JetParticleCollection> taus;
  edm::Handle<l1extra::L1JetParticleCollection> cjets;
  edm::Handle<l1extra::L1JetParticleCollection> fjets;

  if (debug_){
  if(iEvent.getByLabel(l1ExtraTaus_,taus))
    for(l1extra::L1JetParticleCollection::const_iterator i = taus->begin();i!=taus->end();++i)
      {
	std::cout << "tau" << i->pt() << std::endl;
      }
 
  if(iEvent.getByLabel(l1ExtraCJets_,cjets))
    for(l1extra::L1JetParticleCollection::const_iterator i = cjets->begin();i!=cjets->end();++i)
      {     
	std::cout << "cjet" << i->pt() << std::endl;
      }
  
  if(iEvent.getByLabel(l1ExtraFJets_,fjets))
    for(l1extra::L1JetParticleCollection::const_iterator i = fjets->begin();i!=fjets->end();++i)
      {     
	std::cout << "fjet" << i->pt() << std::endl;
      }
  }

  //----------- Offline Jets -----
  if (debug_) std::cout << ">>>now offline jets" << std::endl;
  iEvent.getByLabel(caloJetsTag_,calojetColl_);
  int j = 0;
  double LeadingCaloJetPt = 0.;
  if(calojetColl_.isValid()){
    for(CaloJetCollection::const_iterator jet = calojetColl_->begin();
	jet != calojetColl_->end(); ++jet ) {
      if (debug_) cout <<" Jet " << j
		       <<" pt = " << jet->pt()
		       <<" eta = " << jet->eta()
		       <<" phi = " << jet->phi() << endl;
      if (j==0) LeadingCaloJetPt = jet->pt();
      j++;
      
      if (debug_) {
      //test - pt>100 gev case
      if (j==1 && jet->pt()>100.){
	std::cout << ">>>>>>new event start - " << std::endl;
	cout <<" Jet " << j
	     <<" pt = " << jet->pt()
	     <<" eta = " << jet->eta()
	     <<" phi = " << jet->phi() << endl;

	//
	for(CaloJetCollection::const_iterator jet2 = calojetColl_->begin();
	    jet2 != calojetColl_->end(); ++jet2 ) {
	  cout <<" Jet "   << j
	       <<" pt = "  << jet2->pt()
	       <<" eta = " << jet2->eta()
	       <<" phi = " << jet2->phi() << endl;
	}
	
	// tirggerResults
	for(int i = 0; i < npath; ++i) {
	  if (triggerNames_.triggerName(i).find("HLT_Jet") != std::string::npos ){
	    std::cout << i << " "
		      << triggerNames_.triggerName(i) << " "
		      << triggerResults_->wasrun(i) << " "
		      << triggerResults_->accept(i) << " "
		      << triggerResults_->error(i) << std::endl;
	  }
	}
	
	//-----
	// triggerSummary
 	edm::InputTag testTag2("hltL1sJet30U","",processname_);
 	const int index2 = triggerObj_->filterIndex(testTag2);
	//std::cout << index2 << " " << triggerObj_->sizeFilters() << std::endl;

 	if ( index2 >= triggerObj_->sizeFilters() ) {    
 	  edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< index2 << " of that name ";
	  std::cout << "no index "<< index2 << " of that name " << std::endl;
 	} else {       
       
	  const trigger::Keys & k = triggerObj_->filterKeys(index2);
	  std::string name = triggerObj_->filterTag(index2).encode();
	  std::cout << name << k.size() << std::endl;	  
	  if (k.size()){
	    for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {
	      std::cout << toc[*ki].pt()  << " "
			<< toc[*ki].eta() << " "
			<< toc[*ki].phi() << std::endl;
	    }
	  }

	}

	//-----
	edm::InputTag testTag("hlt1jet30U","",processname_);
	const int index              = triggerObj_->filterIndex(testTag);	
	//std::cout << index << " " << triggerObj_->sizeFilters() << std::endl;

	if ( index >= triggerObj_->sizeFilters() ) {    
	  edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< index << " of that name ";
	  
	} else {       
      
	  const trigger::Keys & k = triggerObj_->filterKeys(index);
	  std::string name = triggerObj_->filterTag(index).encode();

	  if (k.size()){
	    std::cout << name << std::endl;
	    for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {
	      std::cout << toc[*ki].pt()  << " "
			<< toc[*ki].eta() << " "
			<< toc[*ki].phi() << std::endl;
	    }
	  }
      
	} // index

	if(iEvent.getByLabel(l1ExtraTaus_,taus))
	  for(l1extra::L1JetParticleCollection::const_iterator i = taus->begin();i!=taus->end();++i)
	    {
	      std::cout << "tau" << i->pt() << " " << i->eta() << " " << i->phi() << std::endl;
	    }
	
	if(iEvent.getByLabel(l1ExtraCJets_,cjets))
	  for(l1extra::L1JetParticleCollection::const_iterator i = cjets->begin();i!=cjets->end();++i)
	    {     
	      std::cout << "cjet" << i->pt() << " " << i->eta() << " " << i->phi() << std::endl;
	    }
  
	if(iEvent.getByLabel(l1ExtraFJets_,fjets))
	  for(l1extra::L1JetParticleCollection::const_iterator i = fjets->begin();i!=fjets->end();++i)
	    {     
	      std::cout << "fjet" << i->pt() << " " << i->eta() << " " << i->phi() << std::endl;
	    }
	
      }
      //test
      }

    }
  }

  //----------- Offline MET -----
  if (debug_) std::cout << ">>>now offline MET" << std::endl;
  iEvent.getByLabel(caloMETTag_, calometColl_);
  if(calometColl_.isValid()){
    const CaloMETCollection *calometcol = calometColl_.product();
    const CaloMET met = calometcol->front();
    if (debug_) cout <<" MET " 
		     <<" pt = "  << met.pt()
		     <<" phi = " << met.phi() << endl;
  }

  //----------- fill MonitorElements -----
  fillMEforEffSingleJet();
  fillMEforEffDiJetAve();
  fillMEforEffMET();
  fillMEforEffMHT();

  fillMEforMonSingleJet();
  fillMEforMonDiJetAve();
  fillMEforMonMET();
  fillMEforMonMHT();

}


//============ filling MonitorElements =================================

void JetMETHLTOfflineSource::fillMEforEffSingleJet(){

  int npath;
  if(&triggerResults_) {
    npath = triggerResults_->size();
  } else {
    return;
  }

  const trigger::TriggerObjectCollection & toc(triggerObj_->getObjects());

  //---------- loop over different pathCollections ----------
  for(PathInfoCollection::iterator v = HLTPathsEffSingleJet_.begin(); 
      v!= HLTPathsEffSingleJet_.end(); ++v )
    {
      
      // did we pass the denomPath?
      bool acceptedDenom = isHLTPathAccepted("HLT_"+v->getDenomPathName());

      if (acceptedDenom){
	    // denomPath passed 
	    
// 	    edm::InputTag l1sTag(v->getPathNameL1s(),"",processname_);
// 	    const int indexl1s = triggerObj_->filterIndex(l1sTag);

 	    edm::InputTag testTag(v->getDenomPathNameHLT(),"",processname_);
 	    const int index = triggerObj_->filterIndex(testTag);    

// 	    std::cout 
// 		      << v->getTrigEffLevel() << " " 
// 		      << v->getPathName() << " "
// 		      << v->getPathNameHLT() << " "
// 		      << v->getPathNameL1s() << std::endl;
      
	    // njets>=1
	    if (calojetColl_.isValid()){
	    if (calojetColl_->size()){	    
	      // leading jet iterator
	      CaloJetCollection::const_iterator jet = calojetColl_->begin();
	      v->getMEDenominatorPt()->Fill(jet->pt());
	      if (isBarrel(jet->eta()))  v->getMEDenominatorPtBarrel()->Fill(jet->pt());
	      if (isEndCap(jet->eta()))  v->getMEDenominatorPtEndCap()->Fill(jet->pt());
	      if (isForward(jet->eta())) v->getMEDenominatorPtForward()->Fill(jet->pt());
	      v->getMEDenominatorEta()->Fill(jet->eta());
	      v->getMEDenominatorPhi()->Fill(jet->phi());
	      v->getMEDenominatorEtaPhi()->Fill(jet->eta(),jet->phi());
	    }}

 	    if ( index >= triggerObj_->sizeFilters() ) {    
 	      edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< index << " of that name ";	      
 	    } else {             
 	      const trigger::Keys & k = triggerObj_->filterKeys(index);
 	      std::string name = triggerObj_->filterTag(index).encode();
 	      if (k.size()){
 		trigger::Keys::const_iterator ki = k.begin();
 		v->getMEDenominatorPtHLT()->Fill(toc[*ki].pt());
 		if (isBarrel(toc[*ki].eta()))  v->getMEDenominatorPtHLTBarrel()->Fill(toc[*ki].pt());
 		if (isEndCap(toc[*ki].eta()))  v->getMEDenominatorPtHLTEndCap()->Fill(toc[*ki].pt());
 		if (isForward(toc[*ki].eta())) v->getMEDenominatorPtHLTForward()->Fill(toc[*ki].pt());
 		v->getMEDenominatorEtaHLT()->Fill(toc[*ki].eta());
 		v->getMEDenominatorPhiHLT()->Fill(toc[*ki].phi());
 		v->getMEDenominatorEtaPhiHLT()->Fill(toc[*ki].eta(),toc[*ki].phi());
 	      }
 	    }	      

	    bool acceptedNumerator=false;
	    if (v->getTrigEffLevel()=="HLT") {
	      // HLT path
	      acceptedNumerator = isHLTPathAccepted("HLT_"+v->getPathName());
	    } else {
	      // L1s (L1seed case)	     
	      acceptedNumerator = isTriggerObjectFound(v->getPathNameL1s());
	    }

	    if (acceptedNumerator){

		  // numeratorPath passed 
		  if (calojetColl_.isValid()){
 		  if (calojetColl_->size()){	    
 		    // leading jet iterator
 		    CaloJetCollection::const_iterator jet = calojetColl_->begin();
  		    v->getMENumeratorPt()->Fill(jet->pt());
  		    if (isBarrel(jet->eta()))  v->getMENumeratorPtBarrel()->Fill(jet->pt());
  		    if (isEndCap(jet->eta()))  v->getMENumeratorPtEndCap()->Fill(jet->pt());
  		    if (isForward(jet->eta())) v->getMENumeratorPtForward()->Fill(jet->pt());
 		    v->getMENumeratorEta()->Fill(jet->eta());
 		    v->getMENumeratorPhi()->Fill(jet->phi());
 		    v->getMENumeratorEtaPhi()->Fill(jet->eta(),jet->phi());
 		  }}

 		  if ( index >= triggerObj_->sizeFilters() ) {    
 		    edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< index << " of that name ";	      
 		  } else {             
 		    const trigger::Keys & k = triggerObj_->filterKeys(index);
 		    std::string name = triggerObj_->filterTag(index).encode();
 		    if (k.size()){
 		      trigger::Keys::const_iterator ki = k.begin();
 		      v->getMENumeratorPtHLT()->Fill(toc[*ki].pt());
 		      if (isBarrel(toc[*ki].eta()))  v->getMENumeratorPtHLTBarrel()->Fill(toc[*ki].pt());
 		      if (isEndCap(toc[*ki].eta()))  v->getMENumeratorPtHLTEndCap()->Fill(toc[*ki].pt());
 		      if (isForward(toc[*ki].eta())) v->getMENumeratorPtHLTForward()->Fill(toc[*ki].pt());
 		      v->getMENumeratorEtaHLT()->Fill(toc[*ki].eta());
 		      v->getMENumeratorPhiHLT()->Fill(toc[*ki].phi());
 		      v->getMENumeratorEtaPhiHLT()->Fill(toc[*ki].eta(),toc[*ki].phi());
 		    }
 		  }	      		  

	    }     // numerator trig accepted?


	    bool acceptedNumeratorEmulate=false;
	    acceptedNumeratorEmulate = isTrigAcceptedEmulatedSingleJet(*v);

	    if (acceptedNumeratorEmulate){

		  // numeratorPath passed 
		  if (calojetColl_.isValid()){
 		  if (calojetColl_->size()){	    
 		    // leading jet iterator
 		    CaloJetCollection::const_iterator jet = calojetColl_->begin();
  		    v->getMEEmulatedNumeratorPt()->Fill(jet->pt());
  		    if (isBarrel(jet->eta()))  v->getMEEmulatedNumeratorPtBarrel()->Fill(jet->pt());
  		    if (isEndCap(jet->eta()))  v->getMEEmulatedNumeratorPtEndCap()->Fill(jet->pt());
  		    if (isForward(jet->eta())) v->getMEEmulatedNumeratorPtForward()->Fill(jet->pt());
 		    v->getMEEmulatedNumeratorEta()->Fill(jet->eta());
 		    v->getMEEmulatedNumeratorPhi()->Fill(jet->phi());
 		    v->getMEEmulatedNumeratorEtaPhi()->Fill(jet->eta(),jet->phi());
 		  }}

 		  if ( index >= triggerObj_->sizeFilters() ) {    
 		    edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< index << " of that name ";	      
 		  } else {             
 		    const trigger::Keys & k = triggerObj_->filterKeys(index);
 		    std::string name = triggerObj_->filterTag(index).encode();
 		    if (k.size()){
 		      trigger::Keys::const_iterator ki = k.begin();
 		      v->getMEEmulatedNumeratorPtHLT()->Fill(toc[*ki].pt());
 		      if (isBarrel(toc[*ki].eta()))  v->getMEEmulatedNumeratorPtHLTBarrel()->Fill(toc[*ki].pt());
 		      if (isEndCap(toc[*ki].eta()))  v->getMEEmulatedNumeratorPtHLTEndCap()->Fill(toc[*ki].pt());
 		      if (isForward(toc[*ki].eta())) v->getMEEmulatedNumeratorPtHLTForward()->Fill(toc[*ki].pt());
 		      v->getMEEmulatedNumeratorEtaHLT()->Fill(toc[*ki].eta());
 		      v->getMEEmulatedNumeratorPhiHLT()->Fill(toc[*ki].phi());
 		      v->getMEEmulatedNumeratorEtaPhiHLT()->Fill(toc[*ki].eta(),toc[*ki].phi());
 		    }
 		  }	      		  

	    }     // numerator trig accepted?

      }           // denominator trig accepted?
    }             // Loop over all path combinations

}

void JetMETHLTOfflineSource::fillMEforEffDiJetAve(){

  int npath;
  if(&triggerResults_) {
    npath = triggerResults_->size();
  } else {
    return;
  }

  const trigger::TriggerObjectCollection & toc(triggerObj_->getObjects());

  //---------- loop over different pathCollections ----------
  for(PathInfoCollection::iterator v = HLTPathsEffDiJetAve_.begin(); 
      v!= HLTPathsEffDiJetAve_.end(); ++v )
    {
      
      // did we pass the denomPath?
      bool acceptedDenom = isHLTPathAccepted("HLT_"+v->getDenomPathName());

      if (acceptedDenom){
	    // denomPath passed 

// 	    edm::InputTag l1sTag(v->getPathNameL1s(),"",processname_);
// 	    const int indexl1s = triggerObj_->filterIndex(l1sTag);

 	    edm::InputTag testTag(v->getPathNameHLT(),"",processname_);
 	    const int index = triggerObj_->filterIndex(testTag);    

	    // njets>=1
	    if (calojetColl_.isValid()){
	    if (calojetColl_->size()>=2){	    
	      // leading two jets iterator
	      CaloJetCollection::const_iterator jet = calojetColl_->begin();
	      CaloJetCollection::const_iterator jet2= calojetColl_->begin(); jet2++;
	      v->getMEDenominatorPt()->Fill( (jet->pt()+jet2->pt())/2. );
	      v->getMEDenominatorEta()->Fill( (jet->eta()+jet2->eta())/2. );
	    }
	    }

 	    if ( index >= triggerObj_->sizeFilters() ) {    
 	      edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< index << " of that name ";	      
 	    } else {             
 	      const trigger::Keys & k = triggerObj_->filterKeys(index);
 	      std::string name = triggerObj_->filterTag(index).encode();
 	      if (k.size()>=2){
 		trigger::Keys::const_iterator ki = k.begin();
 		trigger::Keys::const_iterator ki2= k.begin(); ki2++;
 		v->getMEDenominatorPtHLT()->Fill( (toc[*ki].pt()+toc[*ki2].pt())/2. );
 		v->getMEDenominatorEtaHLT()->Fill( (toc[*ki].eta()+toc[*ki2].eta())/2. );
 	      }
 	    }	      

	    bool acceptedNumerator=false;
	    if (v->getTrigEffLevel()=="HLT") {
	      // HLT path
	      acceptedNumerator = isHLTPathAccepted("HLT_"+v->getPathName());
	    } else {
	      // L1s (L1seed case)	     
	      acceptedNumerator = isTriggerObjectFound(v->getPathNameL1s());
	    }

	    if (acceptedNumerator){

		  // numeratorPath passed 
		  if (calojetColl_.isValid()){
 		  if (calojetColl_->size()>=2){	    
 		    // leading two jets iterator
 		    CaloJetCollection::const_iterator jet = calojetColl_->begin();
		    CaloJetCollection::const_iterator jet2= calojetColl_->begin(); jet2++;
  		    v->getMENumeratorPt()->Fill( (jet->pt()+jet2->pt())/2. );
 		    v->getMENumeratorEta()->Fill( (jet->eta()+jet2->eta())/2. );
 		  }}

 		  if ( index >= triggerObj_->sizeFilters() ) {    
 		    edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< index << " of that name ";	      
 		  } else {             
 		    const trigger::Keys & k = triggerObj_->filterKeys(index);
 		    std::string name = triggerObj_->filterTag(index).encode();
 		    if (k.size()>=2){
 		      trigger::Keys::const_iterator ki = k.begin();
		      trigger::Keys::const_iterator ki2= k.begin(); ki2++;
 		      v->getMENumeratorPtHLT()->Fill( (toc[*ki].pt()+toc[*ki2].pt())/2. );
 		      v->getMENumeratorEtaHLT()->Fill( (toc[*ki].eta()+toc[*ki2].eta())/2. );
 		    }
 		  }	      
		  
		} // numerator trig accepted?

	    bool acceptedNumeratorEmulate=false;
	    acceptedNumeratorEmulate = isTrigAcceptedEmulatedDiJetAve(*v);
	    
	    if (acceptedNumeratorEmulate){

		  // numeratorPath passed 
		  if (calojetColl_.isValid()){
 		  if (calojetColl_->size()>=2){	    
 		    // leading two jets iterator
 		    CaloJetCollection::const_iterator jet = calojetColl_->begin();
		    CaloJetCollection::const_iterator jet2= calojetColl_->begin(); jet2++;
  		    v->getMEEmulatedNumeratorPt()->Fill( (jet->pt()+jet2->pt())/2. );
 		    v->getMEEmulatedNumeratorEta()->Fill( (jet->eta()+jet2->eta())/2. );
 		  }}

 		  if ( index >= triggerObj_->sizeFilters() ) {    
 		    edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< index << " of that name ";	      
 		  } else {             
 		    const trigger::Keys & k = triggerObj_->filterKeys(index);
 		    std::string name = triggerObj_->filterTag(index).encode();
 		    if (k.size()>=2){
 		      trigger::Keys::const_iterator ki = k.begin();
		      trigger::Keys::const_iterator ki2= k.begin(); ki2++;
 		      v->getMEEmulatedNumeratorPtHLT()->Fill( (toc[*ki].pt()+toc[*ki2].pt())/2. );
 		      v->getMEEmulatedNumeratorEtaHLT()->Fill( (toc[*ki].eta()+toc[*ki2].eta())/2. );
 		    }
 		  }	      
		  
		} // numerator trig accepted?

	  }         // denominator trig accepted?
    }               // Loop over all path combinations

}

void JetMETHLTOfflineSource::fillMEforEffMET(){

  int npath;
  if(&triggerResults_) {
    npath = triggerResults_->size();
  } else {
    return;
  }

  const trigger::TriggerObjectCollection & toc(triggerObj_->getObjects());

  //---------- loop over different pathCollections ----------
  for(PathInfoCollection::iterator v = HLTPathsEffMET_.begin(); 
      v!= HLTPathsEffMET_.end(); ++v )
    {
      
      // did we pass the denomPath?
      bool acceptedDenom = isHLTPathAccepted("HLT_"+v->getDenomPathName());

      if (acceptedDenom){
	// denomPath passed

	    edm::InputTag testTag(v->getDenomPathNameHLT(),"",processname_);
	    const int index = triggerObj_->filterIndex(testTag);    

	    // calomet valid?
	    if (calometColl_.isValid()){	    
	      const CaloMETCollection *calometcol = calometColl_.product();
	      const CaloMET met = calometcol->front();
	      v->getMEDenominatorPt()->Fill(met.pt());
	      v->getMEDenominatorPhi()->Fill(met.phi());
	    }

 	    if ( index >= triggerObj_->sizeFilters() ) {    
 	      edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< index << " of that name ";	      
 	    } else {             
 	      const trigger::Keys & k = triggerObj_->filterKeys(index);
 	      std::string name = triggerObj_->filterTag(index).encode();
 	      if (k.size()){
 		trigger::Keys::const_iterator ki = k.begin();
 		v->getMEDenominatorPtHLT()->Fill(toc[*ki].pt());
 		v->getMEDenominatorPhiHLT()->Fill(toc[*ki].phi());
 	      }
 	    }	      

	    bool acceptedNumerator=false;
	    if (v->getTrigEffLevel()=="HLT") {
	      // HLT path
	      acceptedNumerator = isHLTPathAccepted("HLT_"+v->getPathName());
	    } else {
	      // L1s (L1seed case)	     
	      acceptedNumerator = isTriggerObjectFound(v->getPathNameL1s());
	    }

	    if (acceptedNumerator){

		  // numeratorPath passed 
		  // calomet valid?
		  if (calometColl_.isValid()){	    
		    const CaloMETCollection *calometcol = calometColl_.product();
		    const CaloMET met = calometcol->front();
  		    v->getMENumeratorPt()->Fill(met.pt());
 		    v->getMENumeratorPhi()->Fill(met.phi());
 		  }

 		  if ( index >= triggerObj_->sizeFilters() ) {    
 		    edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< index << " of that name ";	      
 		  } else {             
 		    const trigger::Keys & k = triggerObj_->filterKeys(index);
 		    std::string name = triggerObj_->filterTag(index).encode();
 		    if (k.size()){
 		      trigger::Keys::const_iterator ki = k.begin();
 		      v->getMENumeratorPtHLT()->Fill(toc[*ki].pt());
 		      v->getMENumeratorPhiHLT()->Fill(toc[*ki].phi());
 		    }
 		  }	      
		  
	    } // numerator trig accepted?

	    bool acceptedNumeratorEmulate=false;
	    acceptedNumeratorEmulate = isTrigAcceptedEmulatedMET(*v);
	    	    
	    if (acceptedNumeratorEmulate){

		  // numeratorPath passed 
		  // calomet valid?
		  if (calometColl_.isValid()){	    
		    const CaloMETCollection *calometcol = calometColl_.product();
		    const CaloMET met = calometcol->front();
  		    v->getMEEmulatedNumeratorPt()->Fill(met.pt());
 		    v->getMEEmulatedNumeratorPhi()->Fill(met.phi());
 		  }

 		  if ( index >= triggerObj_->sizeFilters() ) {    
 		    edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< index << " of that name ";	      
 		  } else {             
 		    const trigger::Keys & k = triggerObj_->filterKeys(index);
 		    std::string name = triggerObj_->filterTag(index).encode();
 		    if (k.size()){
 		      trigger::Keys::const_iterator ki = k.begin();
 		      v->getMEEmulatedNumeratorPtHLT()->Fill(toc[*ki].pt());
 		      v->getMEEmulatedNumeratorPhiHLT()->Fill(toc[*ki].phi());
 		    }
 		  }	      
		  
	    } // numerator trig accepted?

      }         // denominator trig accepted?
    }               // Loop over all path combinations

}

void JetMETHLTOfflineSource::fillMEforEffMHT(){

}

void JetMETHLTOfflineSource::fillMEforMonSingleJet(){

  int npath;
  if(&triggerResults_) {
    npath = triggerResults_->size();
  } else {
    return;
  }

  const trigger::TriggerObjectCollection & toc(triggerObj_->getObjects());

  //---------- loop over different pathCollections ----------
  for(PathHLTMonInfoCollection::iterator v = HLTPathsMonSingleJet_.begin(); 
      v!= HLTPathsMonSingleJet_.end(); ++v )
    {
      
      // did we pass the denomPath?
      for(int i = 0; i < npath; ++i) {
	if (triggerNames_.triggerName(i).find("HLT_"+v->getPathName()) != std::string::npos 
	    && triggerResults_->accept(i))
	  {
	    // denomPath passed 
	    
	    // numerator L1s passed
	    edm::InputTag l1sTag(v->getPathNameL1s(),"",processname_);
	    //const int indexl1s = triggerObj_->filterIndex(l1sTag);
	    //if ( indexl1s >= triggerObj_->sizeFilters() ) break;

	    // njets>=1
	    if (calojetColl_.isValid()){
	    if (calojetColl_->size()){	    
	      // leading jet iterator
	      CaloJetCollection::const_iterator jet = calojetColl_->begin();
	      v->getMEPt()->Fill(jet->pt());
	      if (isBarrel(jet->eta()))  v->getMEPtBarrel()->Fill(jet->pt());
	      if (isEndCap(jet->eta()))  v->getMEPtEndCap()->Fill(jet->pt());
	      if (isForward(jet->eta())) v->getMEPtForward()->Fill(jet->pt());
	      v->getMEEta()->Fill(jet->eta());
	      v->getMEPhi()->Fill(jet->phi());
	      v->getMEEtaPhi()->Fill(jet->eta(),jet->phi());
	    }}

 	    edm::InputTag testTag(v->getPathNameHLT(),"",processname_);
 	    const int index = triggerObj_->filterIndex(testTag);    
 	    if ( index >= triggerObj_->sizeFilters() ) {    
 	      edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< index << " of that name ";	      
 	    } else {             
 	      const trigger::Keys & k = triggerObj_->filterKeys(index);
 	      std::string name = triggerObj_->filterTag(index).encode();
 	      if (k.size()){
 		trigger::Keys::const_iterator ki = k.begin();
 		v->getMEPtHLT()->Fill(toc[*ki].pt());
 		if (isBarrel(toc[*ki].eta()))  v->getMEPtHLTBarrel()->Fill(toc[*ki].pt());
 		if (isEndCap(toc[*ki].eta()))  v->getMEPtHLTEndCap()->Fill(toc[*ki].pt());
 		if (isForward(toc[*ki].eta())) v->getMEPtHLTForward()->Fill(toc[*ki].pt());
 		v->getMEEtaHLT()->Fill(toc[*ki].eta());
 		v->getMEPhiHLT()->Fill(toc[*ki].phi());
 		v->getMEEtaPhiHLT()->Fill(toc[*ki].eta(),toc[*ki].phi());
 	      }
 	    }	      

 	    edm::InputTag testTag2(v->getPathNameL1s(),"",processname_);
 	    const int index2 = triggerObj_->filterIndex(testTag2);    
 	    if ( index2 >= triggerObj_->sizeFilters() ) {    
 	      edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< index2 << " of that name ";	      
 	    } else {             
 	      const trigger::Keys & k = triggerObj_->filterKeys(index2);
 	      std::string name = triggerObj_->filterTag(index2).encode();
 	      if (k.size()){
 		trigger::Keys::const_iterator ki = k.begin();
 		v->getMEPtL1s()->Fill(toc[*ki].pt());
 		if (isBarrel(toc[*ki].eta()))  v->getMEPtL1sBarrel()->Fill(toc[*ki].pt());
 		if (isEndCap(toc[*ki].eta()))  v->getMEPtL1sEndCap()->Fill(toc[*ki].pt());
 		if (isForward(toc[*ki].eta())) v->getMEPtL1sForward()->Fill(toc[*ki].pt());
 		v->getMEEtaL1s()->Fill(toc[*ki].eta());
 		v->getMEPhiL1s()->Fill(toc[*ki].phi());
 		v->getMEEtaPhiL1s()->Fill(toc[*ki].eta(),toc[*ki].phi());
 	      }
 	    }	      

	  }         // denominator trig accepted?
      }             // 1st loop for numerator trig path
    }               // Loop over all path combinations

}

void JetMETHLTOfflineSource::fillMEforMonDiJetAve(){

  int npath;
  if(&triggerResults_) {
    npath = triggerResults_->size();
  } else {
    return;
  }

  const trigger::TriggerObjectCollection & toc(triggerObj_->getObjects());

  //---------- loop over different pathCollections ----------
  for(PathHLTMonInfoCollection::iterator v = HLTPathsMonDiJetAve_.begin(); 
      v!= HLTPathsMonDiJetAve_.end(); ++v )
    {
      
      // did we pass the denomPath?
      for(int i = 0; i < npath; ++i) {
	if (triggerNames_.triggerName(i).find("HLT_"+v->getPathName()) != std::string::npos 
	    && triggerResults_->accept(i))
	  {
	    // denomPath passed 

	    // numerator L1s passed
	    edm::InputTag l1sTag(v->getPathNameL1s(),"",processname_);
	    const int indexl1s = triggerObj_->filterIndex(l1sTag);
	    if ( indexl1s >= triggerObj_->sizeFilters() ) break;

	    // njets>=1
	    if (calojetColl_.isValid()){
	    if (calojetColl_->size()>=2){	    
	      // leading two jets iterator
	      CaloJetCollection::const_iterator jet = calojetColl_->begin();
	      CaloJetCollection::const_iterator jet2= calojetColl_->begin(); jet2++;
	      v->getMEPt()->Fill( (jet->pt()+jet2->pt())/2. );
	      v->getMEEta()->Fill( (jet->eta()+jet2->eta())/2. );
	    }
	    }

 	    edm::InputTag testTag(v->getPathNameHLT(),"",processname_);
 	    const int index = triggerObj_->filterIndex(testTag);    
 	    if ( index >= triggerObj_->sizeFilters() ) {    
 	      edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< index << " of that name ";	      
 	    } else {             
 	      const trigger::Keys & k = triggerObj_->filterKeys(index);
 	      std::string name = triggerObj_->filterTag(index).encode();
 	      if (k.size()>=2){
 		trigger::Keys::const_iterator ki = k.begin();
 		trigger::Keys::const_iterator ki2= k.begin(); ki2++;
 		v->getMEPtHLT()->Fill( (toc[*ki].pt()+toc[*ki2].pt())/2. );
 		v->getMEEtaHLT()->Fill( (toc[*ki].eta()+toc[*ki2].eta())/2. );
 	      }
 	    }	      

 	    edm::InputTag testTag2(v->getPathNameL1s(),"",processname_);
 	    const int index2 = triggerObj_->filterIndex(testTag2);    
 	    if ( index2 >= triggerObj_->sizeFilters() ) {    
 	      edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< index2 << " of that name ";	      
 	    } else {             
 	      const trigger::Keys & k = triggerObj_->filterKeys(index2);
 	      std::string name = triggerObj_->filterTag(index).encode();
 	      if (k.size()>=2){
 		trigger::Keys::const_iterator ki = k.begin();
 		trigger::Keys::const_iterator ki2= k.begin(); ki2++;
 		v->getMEPtL1s()->Fill( (toc[*ki].pt()+toc[*ki2].pt())/2. );
 		v->getMEEtaL1s()->Fill( (toc[*ki].eta()+toc[*ki2].eta())/2. );
 	      }
 	    }	      

	  }         // denominator trig accepted?
      }             // 1st loop for numerator trig path
    }               // Loop over all path combinations

}

void JetMETHLTOfflineSource::fillMEforMonMET(){

  int npath;
  if(&triggerResults_) {
    npath = triggerResults_->size();
  } else {
    return;
  }

  const trigger::TriggerObjectCollection & toc(triggerObj_->getObjects());

  //---------- loop over different pathCollections ----------
  for(PathHLTMonInfoCollection::iterator v = HLTPathsMonMET_.begin(); 
      v!= HLTPathsMonMET_.end(); ++v )
    {
      
      // did we pass the denomPath?
      for(int i = 0; i < npath; ++i) {
	if (triggerNames_.triggerName(i).find("HLT_"+v->getPathName()) != std::string::npos 
	    && triggerResults_->accept(i))
	  {
	    // denomPath passed 
	    
	    // numerator L1s passed
	    edm::InputTag l1sTag(v->getPathNameL1s(),"",processname_);
	    //const int indexl1s = triggerObj_->filterIndex(l1sTag);
	    //if ( indexl1s >= triggerObj_->sizeFilters() ) break;

	    // calomet valid?
	    if (calometColl_.isValid()){	    
	      const CaloMETCollection *calometcol = calometColl_.product();
	      const CaloMET met = calometcol->front();
	      v->getMEPt()->Fill(met.pt());
	      v->getMEPhi()->Fill(met.phi());
	    }

 	    edm::InputTag testTag(v->getPathNameHLT(),"",processname_);
 	    const int index = triggerObj_->filterIndex(testTag);    
 	    if ( index >= triggerObj_->sizeFilters() ) {    
 	      edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< index << " of that name ";	      
 	    } else {             
 	      const trigger::Keys & k = triggerObj_->filterKeys(index);
 	      std::string name = triggerObj_->filterTag(index).encode();
 	      if (k.size()){
 		trigger::Keys::const_iterator ki = k.begin();
 		v->getMEPtHLT()->Fill(toc[*ki].pt());
 		v->getMEPhiHLT()->Fill(toc[*ki].phi());
 	      }
 	    }	      

 	    edm::InputTag testTag2(v->getPathNameL1s(),"",processname_);
 	    const int index2 = triggerObj_->filterIndex(testTag2);    
 	    if ( index2 >= triggerObj_->sizeFilters() ) {    
 	      edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< index2 << " of that name ";	      
 	    } else {             
 	      const trigger::Keys & k = triggerObj_->filterKeys(index2);
 	      std::string name = triggerObj_->filterTag(index2).encode();
 	      if (k.size()){
 		trigger::Keys::const_iterator ki = k.begin();
 		v->getMEPtL1s()->Fill(toc[*ki].pt());
 		v->getMEPhiL1s()->Fill(toc[*ki].phi());
 	      }
 	    }	      

	  }         // denominator trig accepted?
      }             // 1st loop for numerator trig path
    }               // Loop over all path combinations

}

void JetMETHLTOfflineSource::fillMEforMonMHT(){

}

//============ booking MonitorElements =================================

void JetMETHLTOfflineSource::bookMEforEffSingleJet(){

  MonitorElement *dummy;
  dummy =  dbe_->bookFloat("dummy");

  double PI = 3.14159;

  //---------- ----------
  std::string dirname = dirName_ + "/EffSingleJet";

  for(PathInfoCollection::iterator v = HLTPathsEffSingleJet_.begin(); 
      v!= HLTPathsEffSingleJet_.end(); ++v )
    {      
      std::string subdirname = dirname + "/Eff_" + v->getPathName() + "_wrt_" + v->getDenomPathName(); 
      if (v->getTrigEffLevel()=="L1s") subdirname = dirname + "/EffL1s_" + v->getPathName() + "_wrt_" + v->getDenomPathName();
      dbe_->setCurrentFolder(subdirname);

      MonitorElement *NumeratorPt     
	= dbe_->book1D("NumeratorPt",    "LeadJetPt_"+v->getPathNameAndLevel()+"_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *NumeratorPtBarrel     
	= dbe_->book1D("NumeratorPtBarrel", "LeadJetPtBarrel_"+v->getPathNameAndLevel()+"_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *NumeratorPtEndCap     
	= dbe_->book1D("NumeratorPtEndCap", "LeadJetPtEndCap_"+v->getPathNameAndLevel()+"_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *NumeratorPtForward     
	= dbe_->book1D("NumeratorPtForward","LeadJetPtForward_"+v->getPathNameAndLevel()+"_"+v->getDenomPathName(),   40, 0.,200.);
      MonitorElement *NumeratorEta    
	= dbe_->book1D("NumeratorEta",   "LeadJetEta_"+v->getPathNameAndLevel()+"_"+v->getDenomPathName(),   50,-5.,5.);
      MonitorElement *NumeratorPhi    
	= dbe_->book1D("NumeratorPhi",   "LeadJetPhi_"+v->getPathNameAndLevel()+"_"+v->getDenomPathName(),   24,-PI,PI);
      MonitorElement *NumeratorEtaPhi 
	= dbe_->book2D("NumeratorEtaPhi","LeadJetEtaPhi_"+v->getPathNameAndLevel()+"_"+v->getDenomPathName(),50,-5.,5.,24,-PI,PI);

      MonitorElement *EmulatedNumeratorPt     
	= dbe_->book1D("EmulatedNumeratorPt",    "LeadJetPt_"+v->getPathNameAndLevel()+"_Emulated_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *EmulatedNumeratorPtBarrel     
	= dbe_->book1D("EmulatedNumeratorPtBarrel", "LeadJetPtBarrel_"+v->getPathNameAndLevel()+"_Emulated_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *EmulatedNumeratorPtEndCap     
	= dbe_->book1D("EmulatedNumeratorPtEndCap", "LeadJetPtEndCap_"+v->getPathNameAndLevel()+"_Emulated_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *EmulatedNumeratorPtForward     
	= dbe_->book1D("EmulatedNumeratorPtForward","LeadJetPtForward_"+v->getPathNameAndLevel()+"_Emulated_"+v->getDenomPathName(),   40, 0.,200.);
      MonitorElement *EmulatedNumeratorEta    
	= dbe_->book1D("EmulatedNumeratorEta",   "LeadJetEta_"+v->getPathNameAndLevel()+"_Emulated_"+v->getDenomPathName(),   50,-5.,5.);
      MonitorElement *EmulatedNumeratorPhi    
	= dbe_->book1D("EmulatedNumeratorPhi",   "LeadJetPhi_"+v->getPathNameAndLevel()+"_Emulated_"+v->getDenomPathName(),   24,-PI,PI);
      MonitorElement *EmulatedNumeratorEtaPhi 
	= dbe_->book2D("EmulatedNumeratorEtaPhi","LeadJetEtaPhi_"+v->getPathNameAndLevel()+"_Emulated_"+v->getDenomPathName(),50,-5.,5.,24,-PI,PI);

      MonitorElement *DenominatorPt     
	= dbe_->book1D("DenominatorPt",    "LeadJetPt_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *DenominatorPtBarrel     
	= dbe_->book1D("DenominatorPtBarrel", "LeadJetPtBarrel_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *DenominatorPtEndCap     
	= dbe_->book1D("DenominatorPtEndCap", "LeadJetPtEndCap_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *DenominatorPtForward     
	= dbe_->book1D("DenominatorPtForward","LeadJetPtForward_"+v->getDenomPathName(),   40, 0.,200.);
      MonitorElement *DenominatorEta    
	= dbe_->book1D("DenominatorEta",   "LeadJetEta_"+v->getDenomPathName(),   50,-5.,5.);
      MonitorElement *DenominatorPhi    
	= dbe_->book1D("DenominatorPhi",   "LeadJetPhi_"+v->getDenomPathName(),   24,-PI,PI);
      MonitorElement *DenominatorEtaPhi 
	= dbe_->book2D("DenominatorEtaPhi","LeadJetEtaPhi_"+v->getDenomPathName(),50,-5.,5.,24,-PI,PI);

      MonitorElement *NumeratorPtHLT     
	= dbe_->book1D("NumeratorPtHLT",    "LeadJetPtHLT_"+v->getPathNameAndLevel()+"_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *NumeratorPtHLTBarrel     
	= dbe_->book1D("NumeratorPtHLTBarrel", "LeadJetPtHLTBarrel_"+v->getPathNameAndLevel()+"_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *NumeratorPtHLTEndCap     
	= dbe_->book1D("NumeratorPtHLTEndCap", "LeadJetPtHLTEndCap_"+v->getPathNameAndLevel()+"_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *NumeratorPtHLTForward     
	= dbe_->book1D("NumeratorPtHLTForward","LeadJetPtHLTForward_"+v->getPathNameAndLevel()+"_"+v->getDenomPathName(),   40, 0.,200.);
      MonitorElement *NumeratorEtaHLT    
	= dbe_->book1D("NumeratorEtaHLT",   "LeadJetEtaHLT_"+v->getPathNameAndLevel()+"_"+v->getDenomPathName(),   50,-5.,5.);
      MonitorElement *NumeratorPhiHLT    
	= dbe_->book1D("NumeratorPhiHLT",   "LeadJetPhiHLT_"+v->getPathNameAndLevel()+"_"+v->getDenomPathName(),   24,-PI,PI);
      MonitorElement *NumeratorEtaPhiHLT 
	= dbe_->book2D("NumeratorEtaPhiHLT","LeadJetEtaPhiHLT_"+v->getPathNameAndLevel()+"_"+v->getDenomPathName(),50,-5.,5.,24,-PI,PI);

      MonitorElement *EmulatedNumeratorPtHLT     
	= dbe_->book1D("EmulatedNumeratorPtHLT",    "LeadJetPtHLT_"+v->getPathNameAndLevel()+"_Emulated_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *EmulatedNumeratorPtHLTBarrel     
	= dbe_->book1D("EmulatedNumeratorPtHLTBarrel", "LeadJetPtHLTBarrel_"+v->getPathNameAndLevel()+"_Emulated_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *EmulatedNumeratorPtHLTEndCap     
	= dbe_->book1D("EmulatedNumeratorPtHLTEndCap", "LeadJetPtHLTEndCap_"+v->getPathNameAndLevel()+"_Emulated_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *EmulatedNumeratorPtHLTForward     
	= dbe_->book1D("EmulatedNumeratorPtHLTForward","LeadJetPtHLTForward_"+v->getPathNameAndLevel()+"_Emulated_"+v->getDenomPathName(),   40, 0.,200.);
      MonitorElement *EmulatedNumeratorEtaHLT    
	= dbe_->book1D("EmulatedNumeratorEtaHLT",   "LeadJetEtaHLT_"+v->getPathNameAndLevel()+"_Emulated_"+v->getDenomPathName(),   50,-5.,5.);
      MonitorElement *EmulatedNumeratorPhiHLT    
	= dbe_->book1D("EmulatedNumeratorPhiHLT",   "LeadJetPhiHLT_"+v->getPathNameAndLevel()+"_Emuated_"+v->getDenomPathName(),   24,-PI,PI);
      MonitorElement *EmulatedNumeratorEtaPhiHLT 
	= dbe_->book2D("EmulatedNumeratorEtaPhiHLT","LeadJetEtaPhiHLT_"+v->getPathNameAndLevel()+"_Emulated_"+v->getDenomPathName(),50,-5.,5.,24,-PI,PI);

      MonitorElement *DenominatorPtHLT     
	= dbe_->book1D("DenominatorPtHLT",    "LeadJetPtHLT_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *DenominatorPtHLTBarrel     
	= dbe_->book1D("DenominatorPtHLTBarrel", "LeadJetPtHLTBarrel_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *DenominatorPtHLTEndCap     
	= dbe_->book1D("DenominatorPtHLTEndCap", "LeadJetPtHLTEndCap_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *DenominatorPtHLTForward     
	= dbe_->book1D("DenominatorPtHLTForward","LeadJetPtHLTForward_"+v->getDenomPathName(),   40, 0.,200.);
      MonitorElement *DenominatorEtaHLT    
	= dbe_->book1D("DenominatorEtaHLT",   "LeadJetEtaHLT_"+v->getDenomPathName(),   50,-5.,5.);
      MonitorElement *DenominatorPhiHLT    
	= dbe_->book1D("DenominatorPhiHLT",   "LeadJetPhiHLT_"+v->getDenomPathName(),   24,-PI,PI);
      MonitorElement *DenominatorEtaPhiHLT 
	= dbe_->book2D("DenominatorEtaPhiHLT","LeadJetEtaPhiHLT_"+v->getDenomPathName(),50,-5.,5.,24,-PI,PI);

      v->setHistos( NumeratorPt,   NumeratorPtBarrel,   NumeratorPtEndCap,   NumeratorPtForward,   
		    NumeratorEta,   NumeratorPhi,   NumeratorEtaPhi,
                    EmulatedNumeratorPt,   EmulatedNumeratorPtBarrel,   EmulatedNumeratorPtEndCap,   EmulatedNumeratorPtForward,   
		    EmulatedNumeratorEta,   EmulatedNumeratorPhi,   EmulatedNumeratorEtaPhi, 
		    DenominatorPt, DenominatorPtBarrel, DenominatorPtEndCap, DenominatorPtForward, 
		    DenominatorEta, DenominatorPhi, DenominatorEtaPhi,
                    NumeratorPtHLT,   NumeratorPtHLTBarrel,   NumeratorPtHLTEndCap,   NumeratorPtHLTForward,   
		    NumeratorEtaHLT,   NumeratorPhiHLT,   NumeratorEtaPhiHLT, 
                    EmulatedNumeratorPtHLT,   EmulatedNumeratorPtHLTBarrel,   EmulatedNumeratorPtHLTEndCap,   EmulatedNumeratorPtHLTForward,   
		    EmulatedNumeratorEtaHLT,   EmulatedNumeratorPhiHLT,   EmulatedNumeratorEtaPhiHLT, 
		    DenominatorPtHLT, DenominatorPtHLTBarrel, DenominatorPtHLTEndCap, DenominatorPtHLTForward, 
		    DenominatorEtaHLT, DenominatorPhiHLT, DenominatorEtaPhiHLT);

    }
  
}

void JetMETHLTOfflineSource::bookMEforEffDiJetAve(){

  MonitorElement *dummy;
  dummy =  dbe_->bookFloat("dummy");

  //double PI = 3.14159;

  //---------- ----------
  std::string dirname = dirName_ + "/EffDiJetAve";

  for(PathInfoCollection::iterator v = HLTPathsEffDiJetAve_.begin(); 
      v!= HLTPathsEffDiJetAve_.end(); ++v )
    {      
      std::string subdirname = dirname + "/Eff_" + v->getPathName() + "_wrt_" + v->getDenomPathName(); 
      if (v->getTrigEffLevel()=="L1s") subdirname = dirname + "/EffL1s_" + v->getPathName() + "_wrt_" + v->getDenomPathName();
      dbe_->setCurrentFolder(subdirname);

      MonitorElement *NumeratorPtAve     
	= dbe_->book1D("NumeratorPtAve",    "DiJetAvePt_"+v->getPathNameAndLevel()+"_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *NumeratorEtaAve    
	= dbe_->book1D("NumeratorEtaAve",   "DiJetAveEta_"+v->getPathNameAndLevel()+"_"+v->getDenomPathName(),   50,-5.,5.);

      MonitorElement *EmulatedNumeratorPtAve     
	= dbe_->book1D("EmulatedNumeratorPtAve",    "DiJetAvePt_"+v->getPathNameAndLevel()+"_Emulated_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *EmulatedNumeratorEtaAve    
	= dbe_->book1D("EmulatedNumeratorEtaAve",   "DiJetAveEta_"+v->getPathNameAndLevel()+"_Emulated_"+v->getDenomPathName(),   50,-5.,5.);

      MonitorElement *DenominatorPtAve     
	= dbe_->book1D("DenominatorPtAve",    "DiJetAvePt_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *DenominatorEtaAve    
	= dbe_->book1D("DenominatorEtaAve",   "DiJetAveEta_"+v->getDenomPathName(),   50,-5.,5.);

      MonitorElement *NumeratorPtAveHLT     
	= dbe_->book1D("NumeratorPtAveHLT",    "DiJetAvePtHLT_"+v->getPathNameAndLevel()+"_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *NumeratorEtaAveHLT    
	= dbe_->book1D("NumeratorEtaAveHLT",   "DiJetAveEtaHLT_"+v->getPathNameAndLevel()+"_"+v->getDenomPathName(),   50,-5.,5.);

      MonitorElement *EmulatedNumeratorPtAveHLT     
	= dbe_->book1D("EmulatedNumeratorPtAveHLT",    "DiJetAvePtHLT_"+v->getPathNameAndLevel()+"_Emulated_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *EmulatedNumeratorEtaAveHLT    
	= dbe_->book1D("EmulatedNumeratorEtaAveHLT",   "DiJetAveEtaHLT_"+v->getPathNameAndLevel()+"_Emulated_"+v->getDenomPathName(),   50,-5.,5.);

      MonitorElement *DenominatorPtAveHLT     
	= dbe_->book1D("DenominatorPtAveHLT",    "DiJetAvePtHLT_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *DenominatorEtaAveHLT    
	= dbe_->book1D("DenominatorEtaAveHLT",   "DiJetAveEtaHLT_"+v->getDenomPathName(),   50,-5.,5.);

      v->setHistos( NumeratorPtAve,       dummy, dummy, dummy,
		    NumeratorEtaAve,      dummy, dummy,
                    EmulatedNumeratorPtAve,       dummy, dummy, dummy,
		    EmulatedNumeratorEtaAve,      dummy, dummy,
		    DenominatorPtAve,     dummy, dummy, dummy,
		    DenominatorEtaAve,    dummy, dummy,
                    NumeratorPtAveHLT,    dummy, dummy, dummy,
		    NumeratorEtaAveHLT,   dummy, dummy,
                    EmulatedNumeratorPtAveHLT,    dummy, dummy, dummy,
		    EmulatedNumeratorEtaAveHLT,   dummy, dummy,
		    DenominatorPtAveHLT,  dummy, dummy, dummy,
		    DenominatorEtaAveHLT, dummy, dummy);

    }
  
}

void JetMETHLTOfflineSource::bookMEforEffMET(){

  MonitorElement *dummy;
  dummy =  dbe_->bookFloat("dummy");

  double PI = 3.14159;

  //---------- ----------
  std::string dirname = dirName_ + "/EffMET";

  for(PathInfoCollection::iterator v = HLTPathsEffMET_.begin(); 
      v!= HLTPathsEffMET_.end(); ++v )
    {      
      std::string subdirname = dirname + "/Eff_" + v->getPathName() + "_wrt_" + v->getDenomPathName(); 
      if (v->getTrigEffLevel()=="L1s") subdirname = dirname + "/EffL1s_" + v->getPathName() + "_wrt_" + v->getDenomPathName();
      dbe_->setCurrentFolder(subdirname);

      MonitorElement *NumeratorEt     
	= dbe_->book1D("NumeratorEt",    "MET_"+v->getPathNameAndLevel()+"_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *NumeratorPhi    
	= dbe_->book1D("NumeratorPhi",   "METPhi_"+v->getPathNameAndLevel()+"_"+v->getDenomPathName(),   24,-PI,PI);

      MonitorElement *EmulatedNumeratorEt     
	= dbe_->book1D("EmulatedNumeratorEt",    "MET_"+v->getPathNameAndLevel()+"_Emulated_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *EmulatedNumeratorPhi    
	= dbe_->book1D("EmulatedNumeratorPhi",   "METPhi_"+v->getPathNameAndLevel()+"_Emulated_"+v->getDenomPathName(),   24,-PI,PI);

      MonitorElement *DenominatorEt     
	= dbe_->book1D("DenominatorEt",    "MET_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *DenominatorPhi    
	= dbe_->book1D("DenominatorPhi",   "METPhi_"+v->getDenomPathName(),   24,-PI,PI);

      MonitorElement *NumeratorEtHLT     
	= dbe_->book1D("NumeratorEtHLT",    "METHLT_"+v->getPathNameAndLevel()+"_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *NumeratorPhiHLT    
	= dbe_->book1D("NumeratorPhiHLT",   "METPhiHLT_"+v->getPathNameAndLevel()+"_"+v->getDenomPathName(),   24,-PI,PI);

      MonitorElement *EmulatedNumeratorEtHLT     
	= dbe_->book1D("EmulatedNumeratorEtHLT",    "METHLT_"+v->getPathNameAndLevel()+"_Emulated_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *EmulatedNumeratorPhiHLT    
	= dbe_->book1D("EmulatedNumeratorPhiHLT",   "METPhiHLT_"+v->getPathNameAndLevel()+"_Emulated_"+v->getDenomPathName(),   24,-PI,PI);

      MonitorElement *DenominatorEtHLT     
	= dbe_->book1D("DenominatorEtHLT",    "METHLT_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *DenominatorPhiHLT    
	= dbe_->book1D("DenominatorPhiHLT",   "METPhiHLT_"+v->getDenomPathName(),   24,-PI,PI);

      v->setHistos( NumeratorEt,   dummy, dummy, dummy, dummy,   NumeratorPhi,   dummy, 
                    EmulatedNumeratorEt,   dummy, dummy, dummy, dummy,   EmulatedNumeratorPhi,   dummy, 
		    DenominatorEt, dummy, dummy, dummy, dummy,   DenominatorPhi, dummy,
                    NumeratorEtHLT,dummy, dummy, dummy, dummy,   NumeratorPhiHLT,dummy,
                    EmulatedNumeratorEtHLT,dummy, dummy, dummy, dummy,   EmulatedNumeratorPhiHLT,dummy,
		    DenominatorEtHLT, dummy, dummy, dummy, dummy,DenominatorPhiHLT,dummy);

    }
  
}

void JetMETHLTOfflineSource::bookMEforEffMHT(){
}


void JetMETHLTOfflineSource::bookMEforMonSingleJet(){

  MonitorElement *dummy;
  dummy =  dbe_->bookFloat("dummy");

  double PI = 3.14159;

  //---------- ----------
  std::string dirname = dirName_ + "/MonitorSingleJet/";

  for(PathHLTMonInfoCollection::iterator v = HLTPathsMonSingleJet_.begin(); 
      v!= HLTPathsMonSingleJet_.end(); ++v )
    {      
      std::string subdirname = dirname + v->getPathName(); 
      dbe_->setCurrentFolder(subdirname);

      MonitorElement *Pt     
	= dbe_->book1D("Pt",    "LeadJetPt",    40, 0.,200.);
      MonitorElement *PtBarrel     
	= dbe_->book1D("PtBarrel", "LeadJetPtBarrel",    40, 0.,200.);
      MonitorElement *PtEndCap     
	= dbe_->book1D("PtEndCap", "LeadJetPtEndCap",    40, 0.,200.);
      MonitorElement *PtForward     
	= dbe_->book1D("PtForward","LeadJetPtForward",   40, 0.,200.);
      MonitorElement *Eta    
	= dbe_->book1D("Eta",   "LeadJetEta",   50,-5.,5.);
      MonitorElement *Phi    
	= dbe_->book1D("Phi",   "LeadJetPhi",   24,-PI,PI);
      MonitorElement *EtaPhi 
	= dbe_->book2D("EtaPhi","LeadJetEtaPhi",50,-5.,5.,24,-PI,PI);

      MonitorElement *PtHLT     
	= dbe_->book1D("PtHLT",    "LeadJetPtHLT",    40, 0.,200.);
      MonitorElement *PtHLTBarrel     
	= dbe_->book1D("PtHLTBarrel", "LeadJetPtHLTBarrel",    40, 0.,200.);
      MonitorElement *PtHLTEndCap     
	= dbe_->book1D("PtHLTEndCap", "LeadJetPtHLTEndCap",    40, 0.,200.);
      MonitorElement *PtHLTForward     
	= dbe_->book1D("PtHLTForward","LeadJetPtHLTForward",   40, 0.,200.);
      MonitorElement *EtaHLT    
	= dbe_->book1D("EtaHLT",   "LeadJetEtaHLT",   50,-5.,5.);
      MonitorElement *PhiHLT    
	= dbe_->book1D("PhiHLT",   "LeadJetPhiHLT",   24,-PI,PI);
      MonitorElement *EtaPhiHLT 
	= dbe_->book2D("EtaPhiHLT","LeadJetEtaPhiHLT",50,-5.,5.,24,-PI,PI);

      MonitorElement *PtL1s     
	= dbe_->book1D("PtL1s",    "LeadJetPtL1s",    40, 0.,200.);
      MonitorElement *PtL1sBarrel     
	= dbe_->book1D("PtL1sBarrel", "LeadJetPtL1sBarrel",    40, 0.,200.);
      MonitorElement *PtL1sEndCap     
	= dbe_->book1D("PtL1sEndCap", "LeadJetPtL1sEndCap",    40, 0.,200.);
      MonitorElement *PtL1sForward     
	= dbe_->book1D("PtL1sForward","LeadJetPtL1sForward",   40, 0.,200.);
      MonitorElement *EtaL1s    
	= dbe_->book1D("EtaL1s",   "LeadJetEtaL1s",   50,-5.,5.);
      MonitorElement *PhiL1s    
	= dbe_->book1D("PhiL1s",   "LeadJetPhiL1s",   24,-PI,PI);
      MonitorElement *EtaPhiL1s 
	= dbe_->book2D("EtaPhiL1s","LeadJetEtaPhiL1s",50,-5.,5.,24,-PI,PI);

      v->setHistos( Pt,   PtBarrel,   PtEndCap,   PtForward,   
		    Eta,   Phi,   EtaPhi, 
                    PtHLT,   PtHLTBarrel,   PtHLTEndCap,   PtHLTForward,   
		    EtaHLT,   PhiHLT,   EtaPhiHLT,
                    PtL1s,   PtL1sBarrel,   PtL1sEndCap,   PtL1sForward,   
		    EtaL1s,   PhiL1s,   EtaPhiL1s);

    }
  
}

void JetMETHLTOfflineSource::bookMEforMonDiJetAve(){

  MonitorElement *dummy;
  dummy =  dbe_->bookFloat("dummy");

  //double PI = 3.14159;

  //---------- ----------
  std::string dirname = dirName_ + "/MonitorDiJetAve/";

  for(PathHLTMonInfoCollection::iterator v = HLTPathsMonDiJetAve_.begin(); 
      v!= HLTPathsMonDiJetAve_.end(); ++v )
    {      
      std::string subdirname = dirname + v->getPathName(); 
      dbe_->setCurrentFolder(subdirname);

      MonitorElement *PtAve     
	= dbe_->book1D("PtAve",    "DiJetAvePt",    40, 0.,200.);
      MonitorElement *EtaAve    
	= dbe_->book1D("EtaAve",   "DiJetAveEta",   50,-5.,5.);

      MonitorElement *PtAveHLT     
	= dbe_->book1D("PtAveHLT",    "DiJetAvePtHLT",    40, 0.,200.);
      MonitorElement *EtaAveHLT    
	= dbe_->book1D("EtaAveHLT",   "DiJetAveEtaHLT",   50,-5.,5.);

      MonitorElement *PtAveL1s     
	= dbe_->book1D("PtAveL1s",    "DiJetAvePtL1s",    40, 0.,200.);
      MonitorElement *EtaAveL1s    
	= dbe_->book1D("EtaAveL1s",   "DiJetAveEtaL1s",   50,-5.,5.);

      v->setHistos( PtAve,       dummy, dummy, dummy,
		    EtaAve,      dummy, dummy,
                    PtAveHLT,    dummy, dummy, dummy,
		    EtaAveHLT,   dummy, dummy,
                    PtAveL1s,    dummy, dummy, dummy,
		    EtaAveL1s,   dummy, dummy);

    }
  
}

void JetMETHLTOfflineSource::bookMEforMonMET(){

  MonitorElement *dummy;
  dummy =  dbe_->bookFloat("dummy");

  double PI = 3.14159;

  //---------- ----------
  std::string dirname = dirName_ + "/MonitorMET/";

  for(PathHLTMonInfoCollection::iterator v = HLTPathsMonMET_.begin(); 
      v!= HLTPathsMonMET_.end(); ++v )
    {      
      std::string subdirname = dirname + v->getPathName(); 
      dbe_->setCurrentFolder(subdirname);

      MonitorElement *Et     
	= dbe_->book1D("Et",    "MET",    40, 0.,200.);
      MonitorElement *Phi    
	= dbe_->book1D("Phi",   "METPhi",   24,-PI,PI);

      MonitorElement *EtHLT     
	= dbe_->book1D("EtHLT",    "METHLT",    40, 0.,200.);
      MonitorElement *PhiHLT    
	= dbe_->book1D("PhiHLT",   "METPhiHLT",   24,-PI,PI);

      MonitorElement *EtL1s     
	= dbe_->book1D("EtL1s",    "METL1s",    40, 0.,200.);
      MonitorElement *PhiL1s    
	= dbe_->book1D("PhiL1s",   "METPhiL1s",   24,-PI,PI);

      v->setHistos( Et,   dummy, dummy, dummy, dummy,   Phi,   dummy, 
                    EtHLT,dummy, dummy, dummy, dummy,   PhiHLT,dummy,
                    EtL1s,dummy, dummy, dummy, dummy,   PhiL1s,dummy);

    }
  
}

void JetMETHLTOfflineSource::bookMEforMonMHT(){
}

//============ Auxiliary =================================

std::string JetMETHLTOfflineSource::getNumeratorTrigger(const std::string& name)
{
//   std::string output = name;
//   //std::cout << name.length()  << std::endl;
//   //std::cout << name.find(":") << std::endl;
//   unsigned int position = name.find(":");
//   output = name.substr(0,position);
//   return output;

  std::string output = "none";
  std::vector<std::string> splitStrings;
  boost::split(splitStrings,name,boost::is_any_of(":"));
  if (splitStrings.size()<1) {
  } else {
    output = splitStrings[0];
  }
  return output;
}


std::string JetMETHLTOfflineSource::getDenominatorTrigger(const std::string& name)
{
//   std::string output = name;
//   //std::cout << name.length()  << std::endl;
//   //std::cout << name.find(":") << std::endl;
//   unsigned int position = name.find(":");
//   output = name.substr(position+1,name.length());
//   return output;

  std::string output = "none";
  std::vector<std::string> splitStrings;
  boost::split(splitStrings,name,boost::is_any_of(":"));
  if (splitStrings.size()<2) {
  } else {
    output = splitStrings[1];
  }
  return output;
}


std::string JetMETHLTOfflineSource::getTriggerEffLevel(const std::string& name)
{
  std::string output = "HLT";
  std::vector<std::string> splitStrings;
  boost::split(splitStrings,name,boost::is_any_of(":"));
  if (splitStrings.size()<3) {
  } else {
    output = splitStrings[2];
  }
  return output;
}


double JetMETHLTOfflineSource::getTriggerThreshold(const std::string& name)
{
  double output = 0.;
  std::vector<std::string> splitStrings;
  boost::split(splitStrings,name,boost::is_any_of(":"));
  if (splitStrings.size()<4) {
  } else {
    output = atof(splitStrings[3].c_str());
  }
  return output;
}


bool JetMETHLTOfflineSource::isBarrel(double eta){
  bool output = false;
  if (fabs(eta)<=1.3) output=true;
  return output;
}


bool JetMETHLTOfflineSource::isEndCap(double eta){
  bool output = false;
  if (fabs(eta)<=3.0 && fabs(eta)>1.3) output=true;
  return output;
}


bool JetMETHLTOfflineSource::isForward(double eta){
  bool output = false;
  if (fabs(eta)>3.0) output=true;
  return output;
}


bool JetMETHLTOfflineSource::validPathHLT(std::string pathname){
  // hltConfig_ has to be defined first before calling this method
  bool output=false;
  for (unsigned int j=0; j!=hltConfig_.size(); ++j) {
    if (hltConfig_.triggerName(j) == pathname )
      output=true;
  }
  return output; 
}


bool JetMETHLTOfflineSource::isHLTPathAccepted(std::string pathName){
  // triggerResults_, triggerNames_ has to be defined first before calling this method
  bool output=false;
  if(&triggerResults_) {
    int npath = triggerResults_->size();
    for(int i = 0; i < npath; ++i) {
      if (triggerNames_.triggerName(i).find(pathName) != std::string::npos 
	  && triggerResults_->accept(i))
	{ output = true; break; }
    }  
  }
  return output;
}


bool JetMETHLTOfflineSource::isTriggerObjectFound(std::string objectName){
  // processname_, triggerObj_ has to be defined before calling this method
  bool output=false;
  edm::InputTag testTag(objectName,"",processname_);
  const int index = triggerObj_->filterIndex(testTag);    
  if ( index >= triggerObj_->sizeFilters() ) {    
    edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< index << " of that name ";
  } else {       
    const trigger::Keys & k = triggerObj_->filterKeys(index);
    if (k.size()) output=true;
  }
  return output;
}


bool JetMETHLTOfflineSource::isTrigAcceptedEmulatedSingleJet(PathInfo v){
  // processname_, triggerObj_ has to be defined before calling this method
  bool output=false;
  std::string objectName = v.getPathNameHLT();
  if (v.getTrigEffLevel()=="L1s") objectName = v.getPathNameL1s();

  const trigger::TriggerObjectCollection & toc(triggerObj_->getObjects());

  edm::InputTag testTag(objectName,"",processname_);
  const int index = triggerObj_->filterIndex(testTag);    
  if ( index >= triggerObj_->sizeFilters() ) {    
    edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< index << " of that name ";
  } else {       
    const trigger::Keys & k = triggerObj_->filterKeys(index);
    if (k.size()){
      for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {
// 	std::cout << toc[*ki].pt()  << " "
// 		  << toc[*ki].eta() << " "
// 		  << toc[*ki].phi() << std::endl;
	if (toc[*ki].pt() > v.getTrigThreshold()) output = true;
	break;
      }      
    }
  }
  return output;
}


bool JetMETHLTOfflineSource::isTrigAcceptedEmulatedDiJetAve(PathInfo v){
  // processname_, triggerObj_ has to be defined before calling this method
  bool output=false;
  std::string objectName = v.getPathNameHLT();
  if (v.getTrigEffLevel()=="L1s") objectName = v.getPathNameL1s();

  const trigger::TriggerObjectCollection & toc(triggerObj_->getObjects());

  edm::InputTag testTag(objectName,"",processname_);
  const int index = triggerObj_->filterIndex(testTag);    
  if ( index >= triggerObj_->sizeFilters() ) {    
    edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< index << " of that name ";
  } else {       
    const trigger::Keys & k = triggerObj_->filterKeys(index);
    if (v.getTrigEffLevel()=="HLT"){
    if (k.size()>2){
      trigger::Keys::const_iterator ki = k.begin();
      trigger::Keys::const_iterator kj = k.begin(); kj++;
      if ((toc[*ki].pt()+toc[*kj].pt())/2 > v.getTrigThreshold()) output = true;
    } else {
      if (k.size()){
      for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {
	if (toc[*ki].pt() > v.getTrigThreshold()) output = true;
	break;
      }      
      }
    }
    }
  }
  return output;
}


bool JetMETHLTOfflineSource::isTrigAcceptedEmulatedMET(PathInfo v){
  // processname_, triggerObj_ has to be defined before calling this method
  bool output=false;
  std::string objectName = v.getPathNameHLT();
  if (v.getTrigEffLevel()=="L1s") objectName = v.getPathNameL1s();

  const trigger::TriggerObjectCollection & toc(triggerObj_->getObjects());

  edm::InputTag testTag(objectName,"",processname_);
  const int index = triggerObj_->filterIndex(testTag);    
  if ( index >= triggerObj_->sizeFilters() ) {    
    edm::LogInfo("JetMETHLTOfflineSource") << "no index "<< index << " of that name ";
  } else {       
    const trigger::Keys & k = triggerObj_->filterKeys(index);
    if (k.size()){
      for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {
// 	std::cout << toc[*ki].pt()  << " "
// 		  << toc[*ki].eta() << " "
// 		  << toc[*ki].phi() << std::endl;
	if (toc[*ki].pt() > v.getTrigThreshold()) output = true;
	break;
      }      
    }
  }
  return output;
}


// Set L1 and HLT module names for each PathInfo
JetMETHLTOfflineSource::PathInfoCollection JetMETHLTOfflineSource::fillL1andHLTModuleNames(PathInfoCollection hltPaths, 
						     std::string L1ModuleName, std::string HLTModuleName){
  // hltConfig_ has to be defined first before calling this method

  for(PathInfoCollection::iterator v = hltPaths.begin(); v!= hltPaths.end(); ) {
    // Check if these paths exist in menu. If not, erase it.
    if (!validPathHLT("HLT_"+v->getPathName()) || !validPathHLT("HLT_"+v->getDenomPathName())){
      v = hltPaths.erase(v);
      continue;
    }

    // list module labels for numpath
    std::vector<std::string> numpathmodules = hltConfig_.moduleLabels("HLT_"+v->getPathName());
    for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin(); 
	numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
      // find L1 global seed for numpath,
      if (hltConfig_.moduleType(*numpathmodule) == L1ModuleName)  v->setPathNameL1s(*numpathmodule);
      if (hltConfig_.moduleType(*numpathmodule) == HLTModuleName) v->setPathNameHLT(*numpathmodule);
      //       std::cout << "testest " << v->getPathName() << "\t" 
      //        		<< *numpathmodule << "\t" 
      // 		<< hltConfig_.moduleType(*numpathmodule) 
      // 		<< L1ModuleName << " "
      // 		<< HLTModuleName << " "
      // 		<< v->getPathNameL1s() << " "
      // 		<< v->getPathNameHLT() << " "
      // 		<< std::endl;
    } // loop over module names

    // list module labels for denompath
    std::vector<std::string> denompathmodules = hltConfig_.moduleLabels("HLT_"+v->getDenomPathName());
    for(std::vector<std::string>::iterator denompathmodule = denompathmodules.begin(); 
	denompathmodule!= denompathmodules.end(); ++denompathmodule ) {
      //       std::cout << "testest " << v->getPathName() << "\t" 
      // 		<< *denompathmodule << "\t" 
      // 		<< hltConfig_.moduleType(*denompathmodule) << std::endl;
      // find L1 global seed for denompath,
      if (hltConfig_.moduleType(*denompathmodule) == L1ModuleName)  v->setDenomPathNameL1s(*denompathmodule);
      if (hltConfig_.moduleType(*denompathmodule) == HLTModuleName) v->setDenomPathNameHLT(*denompathmodule);
    } // loop over module names

    ++v;
  }   // loop over path collections

  return hltPaths;

}


// Set L1 and HLT module names for each PathInfo
JetMETHLTOfflineSource::PathHLTMonInfoCollection JetMETHLTOfflineSource::fillL1andHLTModuleNames(PathHLTMonInfoCollection hltPaths, 
						     std::string L1ModuleName, std::string HLTModuleName){
  // hltConfig_ has to be defined first before calling this method

  for(PathHLTMonInfoCollection::iterator v = hltPaths.begin(); v!= hltPaths.end(); ) {
    // Check if these paths exist in menu. If not, erase it.
    if (!validPathHLT("HLT_"+v->getPathName())){
      v = hltPaths.erase(v);
      continue;
    } 

    // list module labels for numpath
    std::vector<std::string> numpathmodules = hltConfig_.moduleLabels("HLT_"+v->getPathName());
    for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin(); 
	numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
      //std::cout << "testest " << v->getPathName() << "\t" 
      // 		<< *numpathmodule << "\t" 
      // 		<< hltConfig_.moduleType(*numpathmodule) << std::endl;
      // find L1 global seed for numpath,
      if (hltConfig_.moduleType(*numpathmodule) == L1ModuleName)  v->setPathNameL1s(*numpathmodule);
      if (hltConfig_.moduleType(*numpathmodule) == HLTModuleName) v->setPathNameHLT(*numpathmodule);
    } // loop over module names

    ++v;
  }   // loop over path collections

  return hltPaths;

}

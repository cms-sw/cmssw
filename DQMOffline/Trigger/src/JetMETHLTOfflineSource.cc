#include "DQMOffline/Trigger/interface/JetMETHLTOfflineSource.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Run.h"

#include <boost/algorithm/string.hpp>

// using namespace egHLT;

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
					     getNumeratorTrigger(hltPathsEff[i])));
  }

  //--- dijet average
  if (debug_) std::cout << "dijet ave" << std::endl; 
  hltPathsEff.clear();
  hltPathsEff = iConfig.getParameter<std::vector<std::string > >("HLTPathsEffDiJetAve");
  for (unsigned int i=0; i<hltPathsEff.size(); i++){
    if (debug_) std::cout << getNumeratorTrigger(hltPathsEff[i]) << std::endl;
    if (debug_) std::cout << getDenominatorTrigger(hltPathsEff[i]) << std::endl;
    HLTPathsEffDiJetAve_.push_back(PathInfo(getDenominatorTrigger(hltPathsEff[i]),
					     getNumeratorTrigger(hltPathsEff[i])));
  }

  //--- met
  if (debug_) std::cout << "met" << std::endl; 
  hltPathsEff.clear();
  hltPathsEff = iConfig.getParameter<std::vector<std::string > >("HLTPathsEffMET");
  for (unsigned int i=0; i<hltPathsEff.size(); i++){
    if (debug_) std::cout << getNumeratorTrigger(hltPathsEff[i]) << std::endl;
    if (debug_) std::cout << getDenominatorTrigger(hltPathsEff[i]) << std::endl;
    HLTPathsEffMET_.push_back(PathInfo(getDenominatorTrigger(hltPathsEff[i]),
					     getNumeratorTrigger(hltPathsEff[i])));
  }

  //--- mht
  if (debug_) std::cout << "mht" << std::endl; 
  hltPathsEff.clear();
  hltPathsEff = iConfig.getParameter<std::vector<std::string > >("HLTPathsEffMHT");
  for (unsigned int i=0; i<hltPathsEff.size(); i++){
    if (debug_) std::cout << getNumeratorTrigger(hltPathsEff[i]) << std::endl;
    if (debug_) std::cout << getDenominatorTrigger(hltPathsEff[i]) << std::endl;
    HLTPathsEffMHT_.push_back(PathInfo(getDenominatorTrigger(hltPathsEff[i]),
					     getNumeratorTrigger(hltPathsEff[i])));
  }

  //--- trigger path names for more monitoring histograms
  std::vector<std::string > hltMonPaths;
  hltMonPaths = iConfig.getParameter<std::vector<std::string > >("HLTPathsMonSingleJet");
  for (unsigned int i=0; i<hltMonPaths.size(); i++)
    HLTPathsMonSingleJet_.push_back(PathHLTMonInfo(hltMonPaths[i]));

  hltMonPaths = iConfig.getParameter<std::vector<std::string > >("HLTPathsMonDiJetAve");
  for (unsigned int i=0; i<hltMonPaths.size(); i++)
    HLTPathsMonDiJetAve_.push_back(PathHLTMonInfo(hltMonPaths[i]));

  hltMonPaths = iConfig.getParameter<std::vector<std::string > >("HLTPathsMonMET");
  for (unsigned int i=0; i<hltMonPaths.size(); i++)
    HLTPathsMonMET_.push_back(PathHLTMonInfo(hltMonPaths[i]));

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

//   std::vector<edm::ParameterSet> paths = 
//     iConfig.getParameter<std::vector<edm::ParameterSet> >("paths");
//   for(std::vector<edm::ParameterSet>::iterator 
//         pathconf = paths.begin() ; pathconf != paths.end(); 
//       pathconf++) {
//     std::pair<std::string, std::string> custompathnamepair;
//     custompathnamepair.first =pathconf->getParameter<std::string>("pathname"); 
//     custompathnamepair.second = pathconf->getParameter<std::string>("denompathname");   
//     custompathnamepairs_.push_back(custompathnamepair);
//     // customdenompathnames_.push_back(pathconf->getParameter<std::string>("denompathname"));  
//     // custompathnames_.push_back(pathconf->getParameter<std::string>("pathname"));  
//   }

  //--- HLT tag
  hltTag_ = iConfig.getParameter<std::string>("hltTag");
  if (debug_) std::cout << hltTag_ << std::endl;

  //--- DQM output folder name
  dirName_=iConfig.getParameter<std::string>("DQMDirName");
  if (debug_) std::cout << dirName_ << std::endl;

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
    // check if trigger name in (new) config
    cout << "Available TriggerNames are: " << endl;
    hltConfig_.dump("Triggers");
  }

  if (verbose_){
    const unsigned int n(hltConfig_.size());
    for (unsigned int j=0; j!=n; ++j) {
      std::string pathname = hltConfig_.triggerName(j);  
      cout << j << " " << hltConfig_.triggerName(j) << endl;      
    }
  }
  
  //
  //--- obtain the L1 and HLT module names
  for(PathInfoCollection::iterator v = HLTPathsEffSingleJet_.begin(); 
      v!= HLTPathsEffSingleJet_.end(); ++v ) {
    // find L1 global seed for numpath,
    // list module labels for numpath
    //KH needs to put a protectoon so that even if this path doesn't exist in hltConfig_ it doesn't crash
    std::vector<std::string> numpathmodules = hltConfig_.moduleLabels("HLT_"+v->getPathName());
    for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin();
	numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
      std::cout << "testest " << v->getPathName() << "\t" 
      		<< *numpathmodule << "\t" 
      		<< hltConfig_.moduleType(*numpathmodule) << std::endl;
      if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed")
	v->setPathNameL1s(*numpathmodule);
      if (hltConfig_.moduleType(*numpathmodule) == "HLT1CaloJet")
	v->setPathNameL1s(*numpathmodule);
    } // loop over module names
  }   // loop over path collections

  for(PathInfoCollection::iterator v = HLTPathsEffDiJetAve_.begin(); 
      v!= HLTPathsEffDiJetAve_.end(); ++v ) {
    // find L1 global seed for numpath,
    // list module labels for numpath
    //KH need to put a protectoon so that even if this path doesn't exist in hltConfig_ it doesn't crash
    std::vector<std::string> numpathmodules = hltConfig_.moduleLabels("HLT_"+v->getPathName());
    for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin();
	numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
      if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed")
	v->setPathNameL1s(*numpathmodule);
      if (hltConfig_.moduleType(*numpathmodule) == "HLTDiJetAveFilter")
	v->setPathNameL1s(*numpathmodule);
    } // loop over module names
  }   // loop over path collections

  for(PathInfoCollection::iterator v = HLTPathsEffMET_.begin(); 
      v!= HLTPathsEffMET_.end(); ++v ) {
    // find L1 global seed for numpath,
    // list module labels for numpath
    //KH need to put a protectoon so that even if this path doesn't exist in hltConfig_ it doesn't crash
    std::vector<std::string> numpathmodules = hltConfig_.moduleLabels("HLT_"+v->getPathName());
    for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin();
	numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
      if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed")
	v->setPathNameL1s(*numpathmodule);
      if (hltConfig_.moduleType(*numpathmodule) == "HLT1CaloMET")
	v->setPathNameL1s(*numpathmodule);
    } // loop over module names
  }   // loop over path collections


  for(PathHLTMonInfoCollection::iterator v = HLTPathsMonSingleJet_.begin(); 
      v!= HLTPathsMonSingleJet_.end(); ++v ) {
    // find L1 global seed for numpath,
    // list module labels for numpath
    //KH needs to put a protectoon so that even if this path doesn't exist in hltConfig_ it doesn't crash
    std::vector<std::string> numpathmodules = hltConfig_.moduleLabels("HLT_"+v->getPathName());
    for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin();
	numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
      if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed")
	v->setPathNameL1s(*numpathmodule);
      if (hltConfig_.moduleType(*numpathmodule) == "HLT1CaloJet")
	v->setPathNameL1s(*numpathmodule);
    } // loop over module names
  }   // loop over path collections

  for(PathHLTMonInfoCollection::iterator v = HLTPathsMonDiJetAve_.begin(); 
      v!= HLTPathsMonDiJetAve_.end(); ++v ) {
    // find L1 global seed for numpath,
    // list module labels for numpath
    //KH need to put a protectoon so that even if this path doesn't exist in hltConfig_ it doesn't crash
    std::vector<std::string> numpathmodules = hltConfig_.moduleLabels("HLT_"+v->getPathName());
    for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin();
	numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
      if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed")
	v->setPathNameL1s(*numpathmodule);
      if (hltConfig_.moduleType(*numpathmodule) == "HLTDiJetAveFilter")
	v->setPathNameL1s(*numpathmodule);
    } // loop over module names
  }   // loop over path collections

  for(PathHLTMonInfoCollection::iterator v = HLTPathsMonMET_.begin(); 
      v!= HLTPathsMonMET_.end(); ++v ) {
    // find L1 global seed for numpath,
    // list module labels for numpath
    //KH need to put a protectoon so that even if this path doesn't exist in hltConfig_ it doesn't crash
    std::vector<std::string> numpathmodules = hltConfig_.moduleLabels("HLT_"+v->getPathName());
    for(std::vector<std::string>::iterator numpathmodule = numpathmodules.begin();
	numpathmodule!= numpathmodules.end(); ++numpathmodule ) {
      if (hltConfig_.moduleType(*numpathmodule) == "HLTLevel1GTSeed")
	v->setPathNameL1s(*numpathmodule);
      if (hltConfig_.moduleType(*numpathmodule) == "HLT1CaloMET")
	v->setPathNameL1s(*numpathmodule);
    } // loop over module names
  }   // loop over path collections

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

  // did we pass the denomPath?
  if (debug_){
    for(int i = 0; i < npath; ++i) {
      if (triggerNames_.triggerName(i).find("HLT_Jet") != std::string::npos && triggerResults_->accept(i))
	{
	  std::cout << i << " " << triggerNames_.triggerName(i) << " ";
	}
    }
    std::cout << std::endl;
  }

  // 6 HLT_L1Jet6U
  // 7 HLT_Jet15U
  // 8 HLT_Jet30U
  // 9 HLT_Jet50U
  // 10 HLT_FwdJet20U
  // 11 HLT_DiJetAve15U_8E29
  // 12 HLT_DiJetAve30U_8E29
  // 13 HLT_QuadJet15U
  // 14 HLT_L1MET20
  // 15 HLT_MET35
  // 16 HLT_MET100
  // 17 HLT_L1MuOpen
  // 18 HLT_L1Mu
  // 19 HLT_L1Mu20
  // 20 HLT_L2Mu9
  // 21 HLT_L2Mu11
  // 22 HLT_IsoMu3
  // 23 HLT_Mu3
  // 24 HLT_Mu5
  // 25 HLT_Mu9

  //---------- triggerSummary ----------
  if (debug_) std::cout << ">>>now triggerSummary" << std::endl;
  iEvent.getByLabel(triggerSummaryLabel_,triggerObj_);
  if(!triggerObj_.isValid()) {
    edm::LogInfo("JetMETHLTOfflineSource") << "TriggerSummary not found, "
      "skipping event";
    return;
  }
  const trigger::TriggerObjectCollection & toc(triggerObj_->getObjects());

  //--- Show everything
  if (debug_) {
    for ( size_t ia = 0; ia < triggerObj_->sizeFilters(); ++ ia) {
      std::string name = triggerObj_->filterTag(ia).encode();
      std::cout << name << std::endl;
      
      const trigger::Keys & k      = triggerObj_->filterKeys(ia);
      for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {
	std::cout << toc[*ki].pt()  << " "
		  << toc[*ki].eta() << " "
		  << toc[*ki].phi() << std::endl;
      } // loop over different objects
    }   // loop over different paths
  }

  //
  //hlt1jet15U::HLT
  //hlt1jet30U::HLT
  //hltDiJetAve15U8E29::HLT
  //hltDiJetAve30U8E29::HLT
  //
  //level-1 seed
  //hltL1sJet15U::HLT
  //hltL1sJet30U::HLT
  //hltL1sJet50U::HLT
  //hltL1sL1Jet6U::HLT ?
  //hltL1sL1MET20::HLT ?
  //hltL1sMET100::HLT
  //hltL1sMET35::HLT

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
    edm::InputTag testTag2("hltL1sJet30U","",processname_);
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

// 302    // work with l1 information
// 303    if (&l1extjetc) {
// 304      nl1extjetc = ((l1extra::L1JetParticleCollection&)*l1extjetc).size();
// 305      if (debug_) printf("[Info] There are %i L1 jets.\n",nl1extjetc);
// 306      l1extra::L1JetParticleCollection myl1jetsc;
// 307      //     myl1jetsc= (l1extra::L1JetParticleCollection&)*l1extjetc;
// 308 // //      std::sort(myl1jetsc.begin(),myl1jetsc.end(),EtGreater());
// 309      int il1exjt = 0;
// 310      //     for (l1extra::L1JetParticleCollection::const_iterator jtItr = myl1jetsc.begin(); jtItr != myl1jetsc.end(); ++jtItr) {
// 311      if (nl1extjetc > 0) {
// 312        for (l1extra::L1JetParticleCollection::const_iterator jtItr = ((l1extra::L1JetParticleCollection&)*l1extjetc).begin(); jtItr != ((l1extra::L1JetParticleCollection&)*l1extjetc).end(); ++jtItr) {
// 313          l1extjtcet[il1exjt] = jtItr->et();
// 314          l1extjtce[il1exjt] = jtItr->energy();
// 315          l1extjtceta[il1exjt] = jtItr->eta();
// 316          l1extjtcphi[il1exjt] = jtItr->phi();
// 317 
// 318          if (debug_){
// 319            printf("[INFO] L1 Jet(%i) Et = %f\n",il1exjt,l1extjtcet[il1exjt]);
// 320            printf("[INFO] L1 Jet(%i) E  = %f\n",il1exjt,l1extjtce[il1exjt]);
// 321            printf("[INFO] L1 Jet(%i) Eta= %f\n",il1exjt,l1extjtceta[il1exjt]);
// 322            printf("[INFO] L1 Jet(%i) Phi= %f\n",il1exjt,l1extjtcphi[il1exjt]);
// 323          }
// 324          il1exjt++;
// 325        }
// 326      }
// 327    }
// 328    else {
// 329      nl1extjetc = 0;
// 330      if (debug_) std::cout << "[ERROR] -- No L1 Central JET object" << std::endl;
// 331    }


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
      for(int i = 0; i < npath; ++i) {
	if (triggerNames_.triggerName(i).find("HLT_"+v->getDenomPathName()) != std::string::npos 
	    && triggerResults_->accept(i))
	  {
	    // denomPath passed 
	    
	    // numerator L1s passed
	    edm::InputTag l1sTag(v->getPathNameL1s(),"",processname_);
	    const int indexl1s = triggerObj_->filterIndex(l1sTag);
	    if ( indexl1s >= triggerObj_->sizeFilters() ) break;

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

 	    edm::InputTag testTag(v->getPathNameHLT(),"",processname_);
 	    const int index = triggerObj_->filterIndex(testTag);    
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

	    for(int j = 0; j < npath; ++j) {
	      if (triggerNames_.triggerName(j).find("HLT_"+v->getPathName()) != std::string::npos 
		  && triggerResults_->accept(j))
		{		    
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

 		  edm::InputTag testTag(v->getPathNameHLT(),"",processname_);
 		  const int index = triggerObj_->filterIndex(testTag);    
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
		  
		  break;
		} // numerator trig accepted?
	    }     // 2nd loop for numerator trig path
	    break;
	  }         // denominator trig accepted?
      }             // 1st loop for numerator trig path
    }               // Loop over all path combinations

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
      for(int i = 0; i < npath; ++i) {
	if (triggerNames_.triggerName(i).find("HLT_"+v->getDenomPathName()) != std::string::npos 
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
	      v->getMEDenominatorPt()->Fill( (jet->pt()+jet2->pt())/2. );
	      v->getMEDenominatorEta()->Fill( (jet->eta()+jet2->eta())/2. );
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
 		v->getMEDenominatorPtHLT()->Fill( (toc[*ki].pt()+toc[*ki2].pt())/2. );
 		v->getMEDenominatorEtaHLT()->Fill( (toc[*ki].eta()+toc[*ki2].eta())/2. );
 	      }
 	    }	      

	    bool accepted=false;
	    for(int j = 0; j < npath; ++j) {
	      if (triggerNames_.triggerName(j).find("HLT_"+v->getPathName()) != std::string::npos 
		  && triggerResults_->accept(j))
		{		    
		  accepted=true;
		  // numeratorPath passed 

		  if (calojetColl_.isValid()){
 		  if (calojetColl_->size()>=2){	    
 		    // leading two jets iterator
 		    CaloJetCollection::const_iterator jet = calojetColl_->begin();
		    CaloJetCollection::const_iterator jet2= calojetColl_->begin(); jet2++;
  		    v->getMENumeratorPt()->Fill( (jet->pt()+jet2->pt())/2. );
 		    v->getMENumeratorEta()->Fill( (jet->eta()+jet2->eta())/2. );
 		  }}

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
 		      v->getMENumeratorPtHLT()->Fill( (toc[*ki].pt()+toc[*ki2].pt())/2. );
 		      v->getMENumeratorEtaHLT()->Fill( (toc[*ki].eta()+toc[*ki2].eta())/2. );
 		    }
 		  }	      
		  
		  break;
		} // numerator trig accepted?
	    }     // 2nd loop for numerator trig path
	    //if (!accepted) std::cout << "not accepted" << std::endl;
	    break;
	  }         // denominator trig accepted?
      }             // 1st loop for numerator trig path
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
      for(int i = 0; i < npath; ++i) {
	if (triggerNames_.triggerName(i).find("HLT_"+v->getPathName()) != std::string::npos 
	    && triggerResults_->accept(i))
	  {
	    // Path passed 
	    
	    // numerator L1s passed
	    edm::InputTag l1sTag(v->getPathNameL1s(),"",processname_);
	    const int indexl1s = triggerObj_->filterIndex(l1sTag);
	    if ( indexl1s >= triggerObj_->sizeFilters() ) break;

	    // calomet valid?
	    if (calometColl_.isValid()){	    
	      const CaloMETCollection *calometcol = calometColl_.product();
	      const CaloMET met = calometcol->front();
	      v->getMEDenominatorPt()->Fill(met.pt());
	      v->getMEDenominatorPhi()->Fill(met.phi());
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
 		v->getMEDenominatorPtHLT()->Fill(toc[*ki].pt());
 		v->getMEDenominatorPhiHLT()->Fill(toc[*ki].phi());
 	      }
 	    }	      

	    for(int j = 0; j < npath; ++j) {
	      if (triggerNames_.triggerName(j).find("HLT_"+v->getPathName()) != std::string::npos 
		  && triggerResults_->accept(j))
		{		    
		  // numeratorPath passed 
		  // calomet valid?
		  if (calometColl_.isValid()){	    
		    const CaloMETCollection *calometcol = calometColl_.product();
		    const CaloMET met = calometcol->front();
  		    v->getMENumeratorPt()->Fill(met.pt());
 		    v->getMENumeratorPhi()->Fill(met.phi());
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
 		      v->getMENumeratorPtHLT()->Fill(toc[*ki].pt());
 		      v->getMENumeratorPhiHLT()->Fill(toc[*ki].phi());
 		    }
 		  }	      
		  
		  break;
		} // numerator trig accepted?
	    }     // 2nd loop for numerator trig path
	    break;
	  }         // denominator trig accepted?
      }             // 1st loop for numerator trig path
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
	    const int indexl1s = triggerObj_->filterIndex(l1sTag);
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
	    const int indexl1s = triggerObj_->filterIndex(l1sTag);
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
      dbe_->setCurrentFolder(subdirname);

      MonitorElement *NumeratorPt     
	= dbe_->book1D("NumeratorPt",    "LeadJetPt_"+v->getPathName()+"_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *NumeratorPtBarrel     
	= dbe_->book1D("NumeratorPtBarrel", "LeadJetPtBarrel_"+v->getPathName()+"_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *NumeratorPtEndCap     
	= dbe_->book1D("NumeratorPtEndCap", "LeadJetPtEndCap_"+v->getPathName()+"_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *NumeratorPtForward     
	= dbe_->book1D("NumeratorPtForward","LeadJetPtForward_"+v->getPathName()+"_"+v->getDenomPathName(),   40, 0.,200.);
      MonitorElement *NumeratorEta    
	= dbe_->book1D("NumeratorEta",   "LeadJetEta_"+v->getPathName()+"_"+v->getDenomPathName(),   50,-5.,5.);
      MonitorElement *NumeratorPhi    
	= dbe_->book1D("NumeratorPhi",   "LeadJetPhi_"+v->getPathName()+"_"+v->getDenomPathName(),   24,-PI,PI);
      MonitorElement *NumeratorEtaPhi 
	= dbe_->book2D("NumeratorEtaPhi","LeadJetEtaPhi_"+v->getPathName()+"_"+v->getDenomPathName(),50,-5.,5.,24,-PI,PI);

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
	= dbe_->book1D("NumeratorPtHLT",    "LeadJetPtHLT_"+v->getPathName()+"_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *NumeratorPtHLTBarrel     
	= dbe_->book1D("NumeratorPtHLTBarrel", "LeadJetPtHLTBarrel_"+v->getPathName()+"_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *NumeratorPtHLTEndCap     
	= dbe_->book1D("NumeratorPtHLTEndCap", "LeadJetPtHLTEndCap_"+v->getPathName()+"_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *NumeratorPtHLTForward     
	= dbe_->book1D("NumeratorPtHLTForward","LeadJetPtHLTForward_"+v->getPathName()+"_"+v->getDenomPathName(),   40, 0.,200.);
      MonitorElement *NumeratorEtaHLT    
	= dbe_->book1D("NumeratorEtaHLT",   "LeadJetEtaHLT_"+v->getPathName()+"_"+v->getDenomPathName(),   50,-5.,5.);
      MonitorElement *NumeratorPhiHLT    
	= dbe_->book1D("NumeratorPhiHLT",   "LeadJetPhiHLT_"+v->getPathName()+"_"+v->getDenomPathName(),   24,-PI,PI);
      MonitorElement *NumeratorEtaPhiHLT 
	= dbe_->book2D("NumeratorEtaPhiHLT","LeadJetEtaPhiHLT_"+v->getPathName()+"_"+v->getDenomPathName(),50,-5.,5.,24,-PI,PI);

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
		    DenominatorPt, DenominatorPtBarrel, DenominatorPtEndCap, DenominatorPtForward, 
		    DenominatorEta, DenominatorPhi, DenominatorEtaPhi,
                    NumeratorPtHLT,   NumeratorPtHLTBarrel,   NumeratorPtHLTEndCap,   NumeratorPtHLTForward,   
		    NumeratorEtaHLT,   NumeratorPhiHLT,   NumeratorEtaPhiHLT, 
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
      dbe_->setCurrentFolder(subdirname);

      MonitorElement *NumeratorPtAve     
	= dbe_->book1D("NumeratorPtAve",    "DiJetAvePt_"+v->getPathName()+"_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *NumeratorEtaAve    
	= dbe_->book1D("NumeratorEtaAve",   "DiJetAveEta_"+v->getPathName()+"_"+v->getDenomPathName(),   50,-5.,5.);

      MonitorElement *DenominatorPtAve     
	= dbe_->book1D("DenominatorPtAve",    "DiJetAvePt_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *DenominatorEtaAve    
	= dbe_->book1D("DenominatorEtaAve",   "DiJetAveEta_"+v->getDenomPathName(),   50,-5.,5.);

      MonitorElement *NumeratorPtAveHLT     
	= dbe_->book1D("NumeratorPtAveHLT",    "DiJetAvePtHLT_"+v->getPathName()+"_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *NumeratorEtaAveHLT    
	= dbe_->book1D("NumeratorEtaAveHLT",   "DiJetAveEtaHLT_"+v->getPathName()+"_"+v->getDenomPathName(),   50,-5.,5.);

      MonitorElement *DenominatorPtAveHLT     
	= dbe_->book1D("DenominatorPtAveHLT",    "DiJetAvePtHLT_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *DenominatorEtaAveHLT    
	= dbe_->book1D("DenominatorEtaAveHLT",   "DiJetAveEtaHLT_"+v->getDenomPathName(),   50,-5.,5.);

      v->setHistos( NumeratorPtAve,       dummy, dummy, dummy,
		    NumeratorEtaAve,      dummy, dummy,
		    DenominatorPtAve,     dummy, dummy, dummy,
		    DenominatorEtaAve,    dummy, dummy,
                    NumeratorPtAveHLT,    dummy, dummy, dummy,
		    NumeratorEtaAveHLT,   dummy, dummy,
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
      dbe_->setCurrentFolder(subdirname);

      MonitorElement *NumeratorEt     
	= dbe_->book1D("NumeratorEt",    "MET_"+v->getPathName()+"_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *NumeratorPhi    
	= dbe_->book1D("NumeratorPhi",   "METPhi_"+v->getPathName()+"_"+v->getDenomPathName(),   24,-PI,PI);

      MonitorElement *DenominatorEt     
	= dbe_->book1D("DenominatorEt",    "MET_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *DenominatorPhi    
	= dbe_->book1D("DenominatorPhi",   "METPhi_"+v->getDenomPathName(),   24,-PI,PI);

      MonitorElement *NumeratorEtHLT     
	= dbe_->book1D("NumeratorEtHLT",    "METHLT_"+v->getPathName()+"_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *NumeratorPhiHLT    
	= dbe_->book1D("NumeratorPhiHLT",   "METPhiHLT_"+v->getPathName()+"_"+v->getDenomPathName(),   24,-PI,PI);

      MonitorElement *DenominatorEtHLT     
	= dbe_->book1D("DenominatorEtHLT",    "METHLT_"+v->getDenomPathName(),    40, 0.,200.);
      MonitorElement *DenominatorPhiHLT    
	= dbe_->book1D("DenominatorPhiHLT",   "METPhiHLT_"+v->getDenomPathName(),   24,-PI,PI);

      v->setHistos( NumeratorEt,   dummy, dummy, dummy, dummy,   NumeratorPhi,   dummy, 
		    DenominatorEt, dummy, dummy, dummy, dummy,   DenominatorPhi, dummy,
                    NumeratorEtHLT,dummy, dummy, dummy, dummy,   NumeratorPhiHLT,dummy,
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

      v->setHistos( Pt,   PtBarrel,   PtEndCap,   PtForward,   
		    Eta,   Phi,   EtaPhi, 
                    PtHLT,   PtHLTBarrel,   PtHLTEndCap,   PtHLTForward,   
		    EtaHLT,   PhiHLT,   EtaPhiHLT);

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

      v->setHistos( PtAve,       dummy, dummy, dummy,
		    EtaAve,      dummy, dummy,
                    PtAveHLT,    dummy, dummy, dummy,
		    EtaAveHLT,   dummy, dummy);

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

      v->setHistos( Et,   dummy, dummy, dummy, dummy,   Phi,   dummy, 
                    EtHLT,dummy, dummy, dummy, dummy,   PhiHLT,dummy);

    }
  
}

void JetMETHLTOfflineSource::bookMEforMonMHT(){
}

//============ Auxiliary =================================

//KH needs to put a protectoon so that the output won't contain forbidden chars like ":"
std::string JetMETHLTOfflineSource::getNumeratorTrigger(const std::string& name)
{
  std::string output = name;
  //std::cout << name.length()  << std::endl;
  //std::cout << name.find(":") << std::endl;
  unsigned int position = name.find(":");
  output = name.substr(0,position);
  return output;
}


//KH needs to put a protectoon so that the output won't contain forbidden chars like ":"
std::string JetMETHLTOfflineSource::getDenominatorTrigger(const std::string& name)
{
  std::string output = name;
  //std::cout << name.length()  << std::endl;
  //std::cout << name.find(":") << std::endl;
  unsigned int position = name.find(":");
  output = name.substr(position+1,name.length());
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

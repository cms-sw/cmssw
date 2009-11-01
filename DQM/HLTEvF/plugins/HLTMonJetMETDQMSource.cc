
//Source code for HLT JetMET DQ monitoring.

#include "DQM/HLTEvF/interface/HLTMonJetMETDQMSource.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"


#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"

using namespace edm;
using namespace reco;
  
HLTMonJetMETDQMSource::HLTMonJetMETDQMSource(const edm::ParameterSet& iConfig)
{
  if(verbose_) cout  << "constructor...." ;

  logFile_.open("HLTMonJetMETDQMSource.log");

  dbe = NULL;
  if (iConfig.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe = Service < DQMStore > ().operator->();
    dbe->setVerbose(0);
  }
  
  
  debug_=false;
  verbose_=false;
  outputFile_ =
   iConfig.getUntrackedParameter < std::string > ("outputFile", "");
  if (outputFile_.size() != 0) {
    LogInfo("HLTMonJetMETDQMSource") << "L1T Monitoring histograms will be saved to " 
				     << outputFile_ ;
  }
  else {
    outputFile_ = "L1TDQM.root";
  }
  
  bool disable =
    iConfig.getUntrackedParameter < bool > ("disableROOToutput", false);
  if (disable) {
    outputFile_ = "";
  }
  //dirname_="HLT/HLTMonJetMETDQMSource"+iConfig.getParameter<std::string>("@module_label");
  dirname_="HLT/HLTMonJetMET/HLTMonJetMETDQMSource";
  if(verbose_) cout  << dirname_ << " is the dqm source dir." << endl;
  if (dbe != NULL) {
    dbe->setCurrentFolder(dirname_);
  }
  
  //plotting paramters
  thePtMin = iConfig.getUntrackedParameter<double>("PtMin",0.);
  thePtMax = iConfig.getUntrackedParameter<double>("PtMax",1000.);
  theNbins = iConfig.getUntrackedParameter<unsigned int>("Nbins",1000);
  theNbinseta = iConfig.getUntrackedParameter<unsigned int>("Nbins",100);
  
  //info for each filter-step
  reqNum = iConfig.getParameter<unsigned int>("reqNum");
  std::vector<edm::ParameterSet> filters = iConfig.getParameter<std::vector<edm::ParameterSet> >("filters");
  // get parameter info for each trigger path from the config file
  for(std::vector<edm::ParameterSet>::iterator filterconf = filters.begin() ; filterconf != filters.end() ; filterconf++){
    theHLTCollectionLabels.push_back(filterconf->getParameter<edm::InputTag>("HLTCollectionLabels"));
    theHLTOutputTypes.push_back(filterconf->getParameter<unsigned int>("theHLTOutputTypes"));
    
    subdir_.push_back(filterconf->getParameter<std::string>("theSubDir"));
    std::vector<double> bounds = filterconf->getParameter<std::vector<double> >("PlotBounds");
    assert(bounds.size() == 2);
    plotBounds.push_back(std::pair<double,double>(bounds[0],bounds[1]));
  }
  
}

HLTMonJetMETDQMSource::~HLTMonJetMETDQMSource()
{
  
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}

void HLTMonJetMETDQMSource::beginJob(const edm::EventSetup&)
//HLTMonJetMETDQMSource::beginJob(const edm::EventSetup&)
{
  nev_ = 0;
  DQMStore *dbe = 0;
  dbe = Service < DQMStore > ().operator->();
  
  if (!dbe) {
    edm::LogInfo("HLTJetMETDQMSource") << "unable to get DQMStore service?";
  }
  
  if (dbe) {
    dbe->setCurrentFolder(dirname_);
    std::string histoname="total rate";
    
    MonitorElement *dummy;
    MonitorElement* tmphisto;
    dummy =  dbe->bookFloat("dummy");
    if(verbose_) cout  << dirname_ << endl;
    
    
    total = dbe->book1D(histoname.c_str(),histoname.c_str(),theHLTCollectionLabels.size()+1,0,theHLTCollectionLabels.size()+1);
    total->setBinLabel(theHLTCollectionLabels.size()+1,"Total",1);
    for (unsigned int u=0; u<theHLTCollectionLabels.size(); u++){total->setBinLabel(u+1,theHLTCollectionLabels[u].label().c_str());}
    
    for(unsigned int i = 0; i< theHLTCollectionLabels.size() ; i++){
      
      dirname_="HLT/HLTMonJetMET/HLTMonJetMETDQMSource/" + subdir_[i];
      dbe->setCurrentFolder(dirname_);
      histoname = theHLTCollectionLabels[i].label()+"et";
      //ability to customize histograms
      
      thePtMaxTemp = thePtMax;
      thePtMinTemp = thePtMin;
      //}
      // Formatting of various plots.
      histoTitle = theHLTCollectionLabels[i].label() + " Et";
      tmphisto =  dbe->book1D(histoname.c_str(),histoTitle.c_str(),theNbins,thePtMinTemp,thePtMaxTemp);
      tmphisto->setAxisTitle("Number of Events", 2);
      tmphisto->setAxisTitle("p_{T}", 1);
      ethist.push_back(tmphisto);
      
      if(verbose_) cout  << histoname <<", " ;
      
      
      histoname = theHLTCollectionLabels[i].label()+"phi";
      histoTitle = theHLTCollectionLabels[i].label() + " #phi";
      tmphisto =  dbe->book1D(histoname.c_str(),histoTitle.c_str(),theNbinseta,-3.14,3.14);
      tmphisto->setAxisTitle("Number of Events", 2);
      tmphisto->setAxisTitle("#phi", 1);
      phihist.push_back(tmphisto);  
      
      if(verbose_) cout  << histoname <<", " ;
      
      if (!(subdir_[i].find("MET") != std::string::npos ))
	{
	  
	  histoname = theHLTCollectionLabels[i].label()+"eta";
	  histoTitle = theHLTCollectionLabels[i].label() + " #eta";
	  tmphisto =  dbe->book1D(histoname.c_str(),histoTitle.c_str(),theNbinseta,-2.7,2.7);
	  tmphisto->setAxisTitle("Number of Events", 2);
	  tmphisto->setAxisTitle("#eta", 1);
	  etahist.push_back(tmphisto);          
	  
	  if(verbose_) cout  << histoname <<", " ;
	  
	  histoname = theHLTCollectionLabels[i].label()+"eta_phi";
	  histoTitle = theHLTCollectionLabels[i].label() + " #eta vs. #phi";
	  tmphisto =  dbe->book2D(histoname.c_str(),histoTitle.c_str(),theNbinseta,-2.7,2.7, theNbinseta,
				  -3.14, 3.14);
	  tmphisto->setAxisTitle("#phi", 2);
	  tmphisto->setAxisTitle("#eta", 1);
	  eta_phihist.push_back(tmphisto);	
	  
	  if(verbose_) cout  << histoname <<", " ;
	}
    }//done booking all histograms
    
    
    
  }//if dbe
  
  if(verbose_) cout  << " Done with begin job tasks!! " << endl;
  
}

void HLTMonJetMETDQMSource::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  //LogDebug("JetMETHLTOfflineSource") << "beginRun, run " << run.id();
  processname_="HLT";
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
      cout << j << " th trigger path is: " << hltConfig_.triggerName(j) << endl;      
    }
  }
  
}

template <class T> void HLTMonJetMETDQMSource::fillHistos(edm::Handle<trigger::TriggerEventWithRefs>& triggerObj,const edm::Event& iEvent  ,unsigned int n){
  
  
 std::vector<edm::Ref<T> > particlecands;
 // To keep track of what particlecands have passed a filter in TriggerEventWithRefs
 // it adds the name to a list of filter names. To check whether a filter got passed, 
 // one just looks at its index  in the list...if it is not there it (for some reason) 
 // returns the size of the list.
 if (!( triggerObj->filterIndex(theHLTCollectionLabels[n])>=triggerObj->size() ))
   { // only process if availabel  
     
     if(verbose_) cout  << "triggerObj->filterIndex(theHLTCollectionLabels[n]) " << triggerObj->filterIndex(theHLTCollectionLabels[n]) << endl;
     // retrieve saved filter objects
     triggerObj->getObjects(triggerObj->filterIndex(theHLTCollectionLabels[n]),theHLTOutputTypes[n],particlecands);
     
     //fill filter objects into histos
     if (particlecands.size()!=0){
       if(particlecands.size() >= reqNum ) 
	 total->Fill(n+0.5);
       
       if(verbose_) cout  << " theHLTCollectionLabels[n].label() " << theHLTCollectionLabels[n].label() << endl;
       
       if (particlecands[0].isAvailable()){
	 if(verbose_) cout  << " particlecands.size()" << particlecands.size() << endl;
	 ethist[n]->Fill(particlecands[0]->et() );
	   phihist[n]->Fill(particlecands[0]->phi() );
	   
	   
	   if (!(subdir_[n].find("MET") != std::string::npos ))
	     {
	       if(verbose_) cout  << "found Jet object so filling the eta and eta-phi plots as well " << endl;
	       eta_phihist[n]->Fill(particlecands[0]->eta(), particlecands[0]->phi() );
	       etahist[n]->Fill(particlecands[0]->eta() );
	     }
	   
       }// leading canditate in the event 
     }
   }
}

void HLTMonJetMETDQMSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace trigger;
  nev_++;
  if(verbose_) {  
    cout  << " Analysing event nummber =   "  << nev_ << endl;
    cout << "HLTMonJetMETDQMSource: analyze...." << endl;
  }
  
  edm::Handle<trigger::TriggerEventWithRefs> triggerObj;
  iEvent.getByLabel("hltTriggerSummaryRAW",triggerObj);   //Gets the data product which holds
  if(!triggerObj.isValid()) {                             //all of the information. 
    edm::LogWarning("HLTMonJetMETDQMSource") << "RAW-type HLT results not found, skipping event";
    if(verbose_) cout << "RAW-type HLT results not found, skipping event" << endl;
    return;
  }
  
  // total event number
  total->Fill(theHLTCollectionLabels.size()+0.5);
  
  if(verbose_) cout  << "theHLTCollectionLabels.size()" << theHLTCollectionLabels.size() << endl;
  
  if (verbose_){
    int filter_size = triggerObj->size();
    printf("[Filters]: triggerObj->size() = %i\n",filter_size);
    for (int i=0; i<filter_size; i++){
      edm::InputTag MyInputTag = triggerObj->filterTag(i);
      std::cout << "JoCa: filter" << i << " name: " << MyInputTag.label() << std::endl;
    }
  }
  
  for(unsigned int n=0; n < theHLTCollectionLabels.size() ; n++) { //loop over filter modules
    switch(theHLTOutputTypes[n]){
    case 84: // 
      fillHistos<l1extra::L1JetParticleCollection>(triggerObj,iEvent,n);break;
    case 85: // 
      fillHistos<l1extra::L1JetParticleCollection>(triggerObj,iEvent,n);break;
    case 86: // 
      fillHistos<l1extra::L1JetParticleCollection>(triggerObj,iEvent,n);break;
    case 87: // 
     fillHistos<l1extra::L1EtMissParticleCollection>(triggerObj,iEvent,n);break;
    case 91: //photon 
      fillHistos<reco::RecoEcalCandidateCollection>(triggerObj,iEvent,n);break;
    case 95: //Jet
      fillHistos<reco::CaloJetCollection>(triggerObj,iEvent,n);break;
    case 97: //MET
      fillHistos<reco::CaloMETCollection>(triggerObj,iEvent,n);break;
    case 98: //HT
      fillHistos<reco::METCollection>(triggerObj,iEvent,n);break;
    case 100: // TriggerCluster
      fillHistos<reco::RecoEcalCandidateCollection>(triggerObj,iEvent,n);break;
    default: throw(cms::Exception("Release Validation Error")<< "HLT output type not implemented: theHLTOutputTypes[n]" );
    }
  }
}


void HLTMonJetMETDQMSource::bookJetMET()
{
  
  
}

// ------------ method called once each job just after ending the event loop  ------------
void HLTMonJetMETDQMSource::endJob() {
  
  LogInfo("HLTMonJetMETDQMSource") << "analyzed " << nev_ << " events";
  
  if (outputFile_.size() != 0 && dbe)
    dbe->save(outputFile_);
  
  return;
}

//DEFINE_FWK_MODULE(HLTMonJetMETDQMSource);

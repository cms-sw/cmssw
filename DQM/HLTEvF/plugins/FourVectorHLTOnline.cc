// $Id: FourVectorHLTOnline.cc,v 1.12 2008/10/02 18:43:42 berryhil Exp $
// See header file for information. 
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/HLTEvF/interface/FourVectorHLTOnline.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"

#include "PhysicsTools/Utilities/interface/deltaR.h"

#include "DQMServices/Core/interface/MonitorElement.h"

using namespace edm;

FourVectorHLTOnline::FourVectorHLTOnline(const edm::ParameterSet& iConfig):
  resetMe_(true),  currentRun_(-99)
{
  LogDebug("FourVectorHLTOnline") << "constructor...." ;

  dbe_ = Service < DQMStore > ().operator->();
  if ( ! dbe_ ) {
    LogInfo("FourVectorHLTOnline") << "unabel to get DQMStore service?";
  }
  if (iConfig.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe_->setVerbose(0);
  }
  
  
  dirname_="HLTOnline/FourVectorHLTOnline" + 
    iConfig.getParameter<std::string>("@module_label");
  
  if (dbe_ != 0 ) {
    dbe_->setCurrentFolder(dirname_);
  }
  
  
  // plotting paramters
  ptMin_ = iConfig.getUntrackedParameter<double>("ptMin",0.);
  ptMax_ = iConfig.getUntrackedParameter<double>("ptMax",1000.);
  nBins_ = iConfig.getUntrackedParameter<unsigned int>("Nbins",40);
  
  plotAll_ = iConfig.getUntrackedParameter<bool>("plotAll", false);

  if (!plotAll_)
 {
  // this is the list of paths to look at.
  std::vector<edm::ParameterSet> paths = 
    iConfig.getParameter<std::vector<edm::ParameterSet> >("paths");
  for(std::vector<edm::ParameterSet>::iterator 
	pathconf = paths.begin() ; pathconf != paths.end(); 
      pathconf++) {
    std::string pathname = pathconf->getParameter<std::string>("pathname");  
    std::string filtername = pathconf->getParameter<std::string>("filtername");
    int objectType = pathconf->getParameter<unsigned int>("type");
    float ptMin = pathconf->getUntrackedParameter<double>("ptMin");
    float ptMax = pathconf->getUntrackedParameter<double>("ptMax");
    hltPaths_.push_back(PathInfo(pathname, filtername, objectType, ptMin, ptMax));
  }

  if (hltPaths_.size() > 0)
    {
      // book a histogram of scalers
     scalersSelect = dbe_->book1D("selectedScalers","Selected Scalers", hltPaths_.size(), 0.0, (double)hltPaths_.size());
    }

 }
  triggerSummaryLabel_ = 
    iConfig.getParameter<edm::InputTag>("triggerSummaryLabel");
  triggerResultsLabel_ = 
    iConfig.getParameter<edm::InputTag>("triggerResultsLabel");
 
  
}


FourVectorHLTOnline::~FourVectorHLTOnline()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
FourVectorHLTOnline::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace trigger;
  using namespace l1extra;
  ++nev_;
  LogDebug("FourVectorHLTOnline")<< "FourVectorHLTOnline: analyze...." ;
  
  edm::Handle<TriggerResults> triggerResults;
  iEvent.getByLabel(triggerResultsLabel_,triggerResults);
  if(!triggerResults.isValid()) { 
    edm::LogInfo("FourVectorHLTOnline") << "TriggerResults not found, "
      "skipping event"; 
    return;
  }
    
  edm::Handle<TriggerEvent> triggerObj;
  iEvent.getByLabel(triggerSummaryLabel_,triggerObj); 
  if(!triggerObj.isValid()) { 
    edm::LogInfo("FourVectorHLTOnline") << "Summary HLT objects not found, "
      "skipping event"; 
    return;
  }
  
  const trigger::TriggerObjectCollection & toc(triggerObj->getObjects());

    for(PathInfoCollection::iterator v = hltPaths_.begin();
	v!= hltPaths_.end(); ++v ) 
{ 
  // fill scaler histograms
      edm::InputTag filterTag = v->getTag();
      if (plotAll_)
	{
	// loop through indices and see if the filter is on the list of filters used by this path
      
    if (v->getLabel() == "dummy"){
        const std::vector<std::string> filterLabels = hltConfig_.moduleLabels(v->getPath());
	//loop over labels
        for (std::vector<std::string>::const_iterator labelIter= filterLabels.begin(); labelIter!=filterLabels.end(); labelIter++)          
	 {
	   //cout << v->getPath() << "\t" << *labelIter << endl;
           // last match wins...
	   edm::InputTag testTag(*labelIter,"","HLT");
	   //           cout << v->getPath() << "\t" << testTag.label() << "\t" << testTag.process() << endl;
           int testindex = triggerObj->filterIndex(testTag);
           if ( !(testindex >= triggerObj->sizeFilters()) ) {
	     //cout << "found one! " << v->getPath() << "\t" << testTag.label() << endl; 
            filterTag = testTag; v->setLabel(*labelIter);}
	 }
         }
	}

      const int index = triggerObj->filterIndex(filterTag);
      if ( index >= triggerObj->sizeFilters() ) {
	//        cout << "WTF no index "<< index << " of that name "
	//	     << filterTag << endl;
	continue; // not in this event
      }
      LogDebug("FourVectorHLTOnline") << "filling ... " ;
      const trigger::Keys & k = triggerObj->filterKeys(index);
      const trigger::Vids & idtype = triggerObj->filterIds(index);
      // assume for now the first object type is the same as all objects in the collection
      //    cout << filterTag << "\t" << idtype.size() << "\t" << k.size() << endl;
      int triggertype = 0;     
      if (idtype.size() > 0) triggertype = *idtype.begin();
      //     cout << "path " << v->getPath() << " trigger type "<<triggertype << endl;
      if (k.size() > 0) v->getNOnHisto()->Fill(k.size());
      for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {
	v->getEtOnHisto()->Fill(toc[*ki].pt());
	v->getEtaOnHisto()->Fill(toc[*ki].eta());
	v->getPhiOnHisto()->Fill(toc[*ki].phi());
	v->getEtaVsPhiOnHisto()->Fill(toc[*ki].eta(), toc[*ki].phi());
	//	  cout << "pdgId "<<toc[*ki].id() << endl;
      // for muon triggers, loop over and fill offline 4-vectors
      if (triggertype == trigger::TriggerMuon || triggertype == trigger::TriggerL1Mu)
	{
         edm::Handle<l1extra::L1MuonParticleCollection> l1MuonHandle;
         iEvent.getByType(l1MuonHandle);

         if(!l1MuonHandle.isValid()) { 
            edm::LogInfo("FourVectorHLTOnline") << "l1MuonHandle not found, "
            "skipping event"; 
            return;
         }
         const l1extra::L1MuonParticleCollection l1MuonCollection = *(l1MuonHandle.product());

         for (l1extra::L1MuonParticleCollection::const_iterator l1MuonIter=l1MuonCollection.begin(); l1MuonIter!=l1MuonCollection.end(); l1MuonIter++)
         {
	   if (reco::deltaR((*l1MuonIter).eta(),(*l1MuonIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
	  v->getEtL1Histo()->Fill((*l1MuonIter).pt());
	  v->getEtaL1Histo()->Fill((*l1MuonIter).eta());
	  v->getPhiL1Histo()->Fill((*l1MuonIter).phi());
	  v->getEtaVsPhiL1Histo()->Fill((*l1MuonIter).eta(),(*l1MuonIter).phi());
	   }
         }
	}

      // for electron triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerElectron)
	{
         std::vector<edm::Handle<l1extra::L1EmParticleCollection> > l1ElectronHandleList;
         iEvent.getManyByType(l1ElectronHandleList);        
         std::vector<edm::Handle<l1extra::L1EmParticleCollection> >::iterator l1ElectronHandle;

         for (l1ElectronHandle=l1ElectronHandleList.begin(); l1ElectronHandle!=l1ElectronHandleList.end(); l1ElectronHandle++) {

         const L1EmParticleCollection l1ElectronCollection = *(l1ElectronHandle->product());
	   for (L1EmParticleCollection::const_iterator l1ElectronIter=l1ElectronCollection.begin(); l1ElectronIter!=l1ElectronCollection.end(); l1ElectronIter++){
	   if (reco::deltaR((*l1ElectronIter).eta(),(*l1ElectronIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
     	  v->getEtL1Histo()->Fill((*l1ElectronIter).pt());
     	  v->getEtaL1Histo()->Fill((*l1ElectronIter).eta());
          v->getPhiL1Histo()->Fill((*l1ElectronIter).phi());
     	  v->getEtaVsPhiL1Histo()->Fill((*l1ElectronIter).eta(),(*l1ElectronIter).phi());
	   }
	   }
         }
	}


      // for tau triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerTau)
	{

         std::vector<edm::Handle<l1extra::L1JetParticleCollection> > l1TauHandleList;
         iEvent.getManyByType(l1TauHandleList);        
         std::vector<edm::Handle<l1extra::L1JetParticleCollection> >::iterator l1TauHandle;

         for (l1TauHandle=l1TauHandleList.begin(); l1TauHandle!=l1TauHandleList.end(); l1TauHandle++) {
	   if (!l1TauHandle->isValid())
	     {
            edm::LogInfo("FourVectorHLTOnline") << "photonHandle not found, "
            "skipping event"; 
            return;
             } 
         const L1JetParticleCollection l1TauCollection = *(l1TauHandle->product());
	   for (L1JetParticleCollection::const_iterator l1TauIter=l1TauCollection.begin(); l1TauIter!=l1TauCollection.end(); l1TauIter++){
	   if (reco::deltaR((*l1TauIter).eta(),(*l1TauIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
     	  v->getEtL1Histo()->Fill((*l1TauIter).pt());
     	  v->getEtaL1Histo()->Fill((*l1TauIter).eta());
          v->getPhiL1Histo()->Fill((*l1TauIter).phi());
     	  v->getEtaVsPhiL1Histo()->Fill((*l1TauIter).eta(),(*l1TauIter).phi());
	   }
	   }
         }
	}


      // for jet triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerJet)
	{
         std::vector<edm::Handle<l1extra::L1JetParticleCollection> > l1JetHandleList;
         iEvent.getManyByType(l1JetHandleList);        
         std::vector<edm::Handle<l1extra::L1JetParticleCollection> >::iterator l1JetHandle;

         for (l1JetHandle=l1JetHandleList.begin(); l1JetHandle!=l1JetHandleList.end(); l1JetHandle++) {
	   if (!l1JetHandle->isValid())
	     {
            edm::LogInfo("FourVectorHLTOnline") << "photonHandle not found, "
            "skipping event"; 
            return;
             } 
         const L1JetParticleCollection l1JetCollection = *(l1JetHandle->product());
	   for (L1JetParticleCollection::const_iterator l1JetIter=l1JetCollection.begin(); l1JetIter!=l1JetCollection.end(); l1JetIter++){
	   if (reco::deltaR((*l1JetIter).eta(),(*l1JetIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
     	  v->getEtL1Histo()->Fill((*l1JetIter).pt());
     	  v->getEtaL1Histo()->Fill((*l1JetIter).eta());
          v->getPhiL1Histo()->Fill((*l1JetIter).phi());
     	  v->getEtaVsPhiL1Histo()->Fill((*l1JetIter).eta(),(*l1JetIter).phi());
	   }
	   }
         }
	}

      // for bjet triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerBJet)
	{
	}
      // for met triggers, loop over and fill offline 4-vectors
      else if (triggertype == trigger::TriggerMET)
	{
         Handle< L1EtMissParticleCollection > l1MetHandle ;
         iEvent.getByType(l1MetHandle) ;

         if(!l1MetHandle.isValid()) { 
            edm::LogInfo("FourVectorHLTOnline") << "l1MetHandle not found, "
            "skipping event"; 
            return;
         }
         const l1extra::L1EtMissParticleCollection l1MetCollection = *(l1MetHandle.product());

         for (l1extra::L1EtMissParticleCollection::const_iterator l1MetIter=l1MetCollection.begin(); l1MetIter!=l1MetCollection.end(); l1MetIter++)
         {
	   if (reco::deltaR((*l1MetIter).eta(),(*l1MetIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
	  v->getEtL1Histo()->Fill((*l1MetIter).pt());
	  v->getEtaL1Histo()->Fill((*l1MetIter).eta());
	  v->getPhiL1Histo()->Fill((*l1MetIter).phi());
	  v->getEtaVsPhiL1Histo()->Fill((*l1MetIter).eta(),(*l1MetIter).phi());
	   }
         }

	}


      // for photon triggers, loop over and fill offline and L1 4-vectors
      else if (triggertype == trigger::TriggerPhoton)
	{
         std::vector<edm::Handle<l1extra::L1EmParticleCollection> > l1PhotonHandleList;
         iEvent.getManyByType(l1PhotonHandleList);        
         std::vector<edm::Handle<l1extra::L1EmParticleCollection> >::iterator l1PhotonHandle;

         for (l1PhotonHandle=l1PhotonHandleList.begin(); l1PhotonHandle!=l1PhotonHandleList.end(); l1PhotonHandle++) {
	   if (!l1PhotonHandle->isValid())
	     {
            edm::LogInfo("FourVectorHLTOnline") << "photonHandle not found, "
            "skipping event"; 
            return;
             } 
         const L1EmParticleCollection l1PhotonCollection = *(l1PhotonHandle->product());
	   for (L1EmParticleCollection::const_iterator l1PhotonIter=l1PhotonCollection.begin(); l1PhotonIter!=l1PhotonCollection.end(); l1PhotonIter++){
	   if (reco::deltaR((*l1PhotonIter).eta(),(*l1PhotonIter).phi(),toc[*ki].eta(),toc[*ki].phi()) < 0.3){
     	  v->getEtL1Histo()->Fill((*l1PhotonIter).pt());
     	  v->getEtaL1Histo()->Fill((*l1PhotonIter).eta());
          v->getPhiL1Histo()->Fill((*l1PhotonIter).phi());
     	  v->getEtaVsPhiL1Histo()->Fill((*l1PhotonIter).eta(),(*l1PhotonIter).phi());
	   }
	   

	 }
       }
     }

    }
  }
}



// -- method called once each job just before starting event loop  --------
void 
FourVectorHLTOnline::beginJob(const edm::EventSetup&)
{
  nev_ = 0;
  DQMStore *dbe = 0;
  dbe = Service<DQMStore>().operator->();
  
  if (dbe) {
    dbe->setCurrentFolder(dirname_);
    dbe->rmdir(dirname_);
  }
  
  
  if (dbe) {
    dbe->setCurrentFolder(dirname_);
    }  
}

// - method called once each job just after ending the event loop  ------------
void 
FourVectorHLTOnline::endJob() 
{
   LogInfo("FourVectorHLTOnline") << "analyzed " << nev_ << " events";
   return;
}


// BeginRun
void FourVectorHLTOnline::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  LogDebug("FourVectorHLTOnline") << "beginRun, run " << run.id();
// HLT config does not change within runs!
  std::string process_ = "HLT";
  if (!hltConfig_.init("HLT")) {
  LogDebug("FourVectorHLTOnline") << "HLTConfigProvider failed to initialize.";
    // check if trigger name in (new) config
    //	cout << "Available TriggerNames are: " << endl;
	//	hltConfig_.dump("Triggers");
      }



  if (1)
 {
  DQMStore *dbe = 0;
  dbe = Service<DQMStore>().operator->();
  
  if (dbe) {
    dbe->setCurrentFolder(dirname_);
  }

    const unsigned int n(hltConfig_.size());
    for (unsigned int i=0; i!=n; ++i) {
      // cout << hltConfig_.triggerName(i) << endl;
    
    std::string pathname = hltConfig_.triggerName(i);  
    std::string filtername("dummy");
    int objectType = 0;
    float ptMin = 0.0;
    float ptMax = 100.0;
    if (pathname.find("HLT_") != std::string::npos && plotAll_)
    hltPaths_.push_back(PathInfo(pathname, filtername, objectType, ptMin, ptMax));
    }
    // now set up all of the histos for each path
    for(PathInfoCollection::iterator v = hltPaths_.begin();
	  v!= hltPaths_.end(); ++v ) {
    	MonitorElement *NOn, *etOn, *etaOn, *phiOn, *etavsphiOn=0;
	MonitorElement *etL1, *etaL1, *phiL1, *etavsphiL1=0;
	std::string labelname("dummy");
        labelname = v->getPath();
	std::string histoname(labelname+"_NOn");
	std::string title(labelname+" N online");
	NOn =  dbe->book1D(histoname.c_str(),
			  title.c_str(),10,
			  0.5,
			  10.5);
      
	histoname = labelname+"_etOn";
	title = labelname+" E_t online";
	etOn =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_etL1";
	title = labelname+" E_t L1";
	etL1 =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_, 
                           v->getPtMin(),
			   v->getPtMax());

	histoname = labelname+"_etaOn";
	title = labelname+" #eta online";
	etaOn =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_,-2.7,2.7);

	histoname = labelname+"_etaL1";
	title = labelname+" #eta L1";
	etaL1 =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_,-2.7,2.7);

	histoname = labelname+"_phiOn";
	title = labelname+" #phi online";
	phiOn =  dbe->book1D(histoname.c_str(),
			   histoname.c_str(),nBins_,-3.14,3.14);

	histoname = labelname+"_phiL1";
	title = labelname+" #phi L1";
	phiL1 =  dbe->book1D(histoname.c_str(),
			   histoname.c_str(),nBins_,-3.14,3.14);
 
	histoname = labelname+"_etaphiOn";
	title = labelname+" #eta vs #phi online";
	etavsphiOn =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins_,-2.7,2.7,
				nBins_,-3.14, 3.14);

	histoname = labelname+"_etaphiL1";
	title = labelname+" #eta vs #phi L1";
	etavsphiL1 =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins_,-2.7,2.7,
				nBins_,-3.14, 3.14);

	v->setHistos( NOn, etOn, etaOn, phiOn, etavsphiOn, etL1, etaL1, phiL1, etavsphiL1);


    }
 }
 return;



}

/// EndRun
void FourVectorHLTOnline::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  LogDebug("FourVectorHLTOnline") << "endRun, run " << run.id();
}

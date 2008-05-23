// $Id$
// See header file for information. 
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/HLTEvF/interface/FourVectorHLT.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "DQMServices/Core/interface/MonitorElement.h"

using namespace edm;

FourVectorHLT::FourVectorHLT(const edm::ParameterSet& iConfig):
  resetMe_(true),  currentRun_(-99)
{
  
  LogDebug("FourVectorHLT") << "constructor...." ;
  
  
  dbe_ = NULL;
  if (iConfig.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe_ = Service < DQMStore > ().operator->();
    dbe_->setVerbose(0);
  }
  
  
  dirname_="HLT/FourVectorHLT" + 
    iConfig.getParameter<std::string>("@module_label");
  
  if (dbe_ != NULL) {
    dbe_->setCurrentFolder(dirname_);
  }
  
  
  // plotting paramters
  ptMin_ = iConfig.getUntrackedParameter<double>("ptMin",0.);
  ptMax_ = iConfig.getUntrackedParameter<double>("ptMax",1000.);
  nBins_ = iConfig.getUntrackedParameter<unsigned int>("Nbins",40);
  
  plotAll_ = iConfig.getUntrackedParameter<bool>("plotAll", false);
  
  // this is the list of paths to look at.
  std::vector<edm::ParameterSet> filters = 
    iConfig.getParameter<std::vector<edm::ParameterSet> >("filters");
  for(std::vector<edm::ParameterSet>::iterator 
	filterconf = filters.begin() ; filterconf != filters.end(); 
      filterconf++) {
    std::string me  = filterconf->getParameter<std::string>("name");
    int objectType = filterconf->getParameter<unsigned int>("type");
    float ptMin = filterconf->getUntrackedParameter<double>("ptMin");
    float ptMax = filterconf->getUntrackedParameter<double>("ptMax");
    hltPaths_.push_back(PathInfo(me, objectType, ptMin, ptMax));
  }
  
  if ( hltPaths_.size() && plotAll_) {
    // these two ought to be mutually exclusive....
    LogWarning("FourVectorHLT") << "Using both plotAll and a list. "
      "list will be ignored." ;
  }
  triggerSummaryLabel_ = 
    iConfig.getParameter<edm::InputTag>("triggerSummaryLabel");

  // for use when we plot all.
  triggerResultLabel_ = 
    iConfig.getParameter<edm::InputTag>("triggerResultLabel");
  
  
}


FourVectorHLT::~FourVectorHLT()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
FourVectorHLT::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace trigger;
  ++nev_;
  LogDebug("FourVectorHLT")<< "FourVectorHLT: analyze...." ;
  
  edm::Handle<TriggerEvent> triggerObj;
  iEvent.getByLabel(triggerSummaryLabel_,triggerObj); 
  if(!triggerObj.isValid()) { 
    edm::LogWarning("FourVectorHLT") << "Summary HLT objects not found, "
      "skipping event"; 
    return;
  }
  
  if ( plotAll_ && resetMe_ ) {
    LogDebug("FourVectorHLT") << "resetting ... " ;
    hltPaths_.clear(); // do I need to get rid of the old histos?
    DQMStore *dbe(Service<DQMStore>().operator->());
    
    edm::Handle<TriggerResults> hltResults;
    bool b = iEvent.getByLabel(triggerResultLabel_, hltResults);
    if ( !b ) {
      edm::LogWarning("FourVectorHLT") << " getByLabel"
				       << " for TriggerResults failed"
				       << " with label "
				       << triggerResultLabel_;
      return;
    }
    std::cout << "pw --------------------------------" << std::endl;
    std::cout << *hltResults << std::endl;
    std::cout << "pw --------------------------------" << std::endl;
    // save path names in DQM-accessible format
    TriggerNames names(*hltResults);
    std::cout << "number of names: " << names.triggerNames().size()
	      << std::endl;
    int oo(0);
    for ( TriggerNames::Strings::const_iterator 
	    j = names.triggerNames().begin();
	  j !=names.triggerNames().end(); ++j ) {
      //--------------
      std::cout << "resetMe: path " << oo++ << ": " << *j << std::endl;
      //--------------
      MonitorElement *et(0), *eta(0), *phi(0), *etavsphi(0);
      std::string histoname((*j)+"_et");
      std::string title((*j)+" E_t");
      et =  dbe->book1D(histoname.c_str(),
			title.c_str(),nBins_, 0, 100);
      
      histoname = (*j)+"_eta";
      title = (*j)+" #eta";
      eta =  dbe->book1D(histoname.c_str(),
			 title.c_str(),nBins_,-2.7,2.7);
      
      histoname = (*j)+"_phi";
      title = (*j)+" #phi";
      phi =  dbe->book1D(histoname.c_str(),
			 histoname.c_str(),nBins_,-3.14,3.14);
      
      
      histoname = (*j)+"_etaphi";
      title = (*j)+" #eta vs #phi";
      etavsphi =  dbe->book2D(histoname.c_str(),
			      title.c_str(),
			      nBins_,-2.7,2.7,
			      nBins_,-3.14, 3.14);
      
      // no idea how to get the bin boundries in this mode
      PathInfo e(*j,0, et, eta, phi, etavsphi, 0,100);
      hltPaths_.push_back(e);  // I don't ever use these....
    } // resetme
    
    
    std::cout << "hltPaths_ is now " << hltPaths_.size() << std::endl;
    resetMe_ = false;
  }
  

  const trigger::TriggerObjectCollection & toc(triggerObj->getObjects());
  for(PathInfoCollection::iterator v = hltPaths_.begin();
      v!= hltPaths_.end(); ++v ) {
    const int index = triggerObj->filterIndex(v->getName());
    if ( index >= triggerObj->sizeFilters() ) {
      continue; // not in this event
    }
    LogDebug("FourVectorHLT") << "filling ... " ;
    const trigger::Keys & k = triggerObj->filterKeys(index);
    for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {
      v->getEtHisto()->Fill(toc[*ki].pt());
      v->getEtaHisto()->Fill(toc[*ki].eta());
      v->getPhiHisto()->Fill(toc[*ki].phi());
      v->getEtaVsPhiHisto()->Fill(toc[*ki].eta(), toc[*ki].phi());
    }  
  }
}


// -- method called once each job just before starting event loop  --------
void 
FourVectorHLT::beginJob(const edm::EventSetup&)
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

    if ( ! plotAll_ ) {
      for(PathInfoCollection::iterator v = hltPaths_.begin();
	  v!= hltPaths_.end(); ++v ) {
	MonitorElement *et, *eta, *phi, *etavsphi=0;
	std::string histoname(v->getName()+"_et");
	std::string title(v->getName()+" E_t");
	et =  dbe->book1D(histoname.c_str(),
			  title.c_str(),nBins_,
			  v->getPtMin(),
			  v->getPtMax());
      
	histoname = v->getName()+"_eta";
	title = v->getName()+" #eta";
	eta =  dbe->book1D(histoname.c_str(),
			   title.c_str(),nBins_,-2.7,2.7);

	histoname = v->getName()+"_phi";
	title = v->getName()+" #phi";
	phi =  dbe->book1D(histoname.c_str(),
			   histoname.c_str(),nBins_,-3.14,3.14);
 

	histoname = v->getName()+"_etaphi";
	title = v->getName()+" #eta vs #phi";
	etavsphi =  dbe->book2D(histoname.c_str(),
				title.c_str(),
				nBins_,-2.7,2.7,
				nBins_,-3.14, 3.14);
      
	v->setHistos( et, eta, phi, etavsphi);
      } 
    } // ! plotAll_
  }
}

// - method called once each job just after ending the event loop  ------------
void 
FourVectorHLT::endJob() 
{
   LogInfo("FourVectorHLT") << "analyzed " << nev_ << " events";
   return;
}


// BeginRun
void FourVectorHLT::beginRun(const edm::Run& run, const edm::EventSetup& c)
{

  LogDebug("FourVectorHLT") << "beginRun, run " << run.id();

  if ( currentRun_ != int(run.id().run()) ) {
    resetMe_ = true;
    currentRun_ = run.id().run();
  }
}

/// EndRun
void FourVectorHLT::endRun(const edm::Run& run, const edm::EventSetup& c)
{
  LogDebug("FourVectorHLT") << "endRun, run " << run.id();
  if ( currentRun_ != int(run.id().run()) ) {
    resetMe_ = true;
    currentRun_ = run.id().run();
  }
}

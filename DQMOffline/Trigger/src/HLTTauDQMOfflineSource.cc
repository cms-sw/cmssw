#include "DQMOffline/Trigger/interface/HLTTauDQMOfflineSource.h"
#include "Math/GenVector/VectorUtil.h"
using namespace std;
using namespace edm;
using namespace trigger;


//
// constructors and destructor
//
HLTTauDQMOfflineSource::HLTTauDQMOfflineSource( const edm::ParameterSet& ps ) :counterEvt_(0)
{

  //Get  Monitoring Parameters
  mainFolder_             = ps.getParameter < std::string > ("DQMFolder");
  monitorName_            = ps.getParameter<string>("monitorName");
  outputFile_             = ps.getParameter < std::string > ("outputFile");
  prescaleEvt_            = ps.getParameter<int>("prescaleEvt");
  disable_                = ps.getParameter < bool > ("disableROOToutput");
  verbose_                = ps.getParameter < bool > ("verbose");
  nTriggeredTaus_         = ps.getParameter < unsigned > ("NTriggeredTaus");
  nTriggeredLeptons_      = ps.getParameter < unsigned > ("NTriggeredLeptons");
  leptonPdgID_            = ps.getParameter < int > ("LeptonPDGId");
  tauPdgID_               = ps.getParameter < int > ("TauPDGId");
  triggerEvent_           = ps.getParameter < edm::InputTag > ("TriggerEvent");
  mainPath_               = ps.getParameter<edm::InputTag>("MainFilter");
  l1BackupPath_           = ps.getParameter<edm::InputTag>("L1BackupFilter");
  l2BackupPath_           = ps.getParameter<edm::InputTag>("L2BackupFilter");
  l25BackupPath_          = ps.getParameter<edm::InputTag>("L25BackupFilter");
  l3BackupPath_           = ps.getParameter<edm::InputTag>("L3BackupFilter");
  refTauObjects_          = ps.getParameter<edm::InputTag>("refTauObjects");
  refLeptonObjects_       = ps.getParameter<edm::InputTag>("refLeptonObjects");
  corrDeltaR_             = ps.getParameter<double>("matchingDeltaR");
  EtMin_                  = ps.getParameter < double > ("HistEtMin");
  EtMax_                  = ps.getParameter < double > ("HistEtMax");
  NEtBins_                = ps.getParameter < int > ("HistNEtBins");
  NEtaBins_               = ps.getParameter < int > ("HistNEtaBins");


  dbe_ = Service < DQMStore > ().operator->();
  dbe_->setVerbose(0);
 
   if (disable_) {
     outputFile_ = "";
   }


}


HLTTauDQMOfflineSource::~HLTTauDQMOfflineSource()
{
   
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}


//--------------------------------------------------------
void 
HLTTauDQMOfflineSource::beginJob(const EventSetup& context)
{
   if (dbe_) 
     {
           dbe_->setCurrentFolder(mainFolder_);
	 
	   EventsPassed_ = (dbe_->book1D((monitorName_+"_Bits").c_str(),(monitorName_+"EvtsPassed").c_str(),11,0,11));
	   EventsPassed_->setAxisTitle("#tau Trigger Paths");
	   EventsPassed_->setBinLabel(1,"");
	   EventsPassed_->setBinLabel(2,"Main Path");
	   EventsPassed_->setBinLabel(3,"");
	   EventsPassed_->setBinLabel(4,"noL1 Path");
	   EventsPassed_->setBinLabel(5,"");
	   EventsPassed_->setBinLabel(6,"no L2 Path");
	   EventsPassed_->setBinLabel(7,"");
	   EventsPassed_->setBinLabel(8,"no L25 Path");
	   EventsPassed_->setBinLabel(9,"");
	   EventsPassed_->setBinLabel(10,"no L3 Path");
	   EventsPassed_->setBinLabel(11,"");

	   EventsRef_ = (dbe_->book1D((monitorName_+"_Ref").c_str(),(monitorName_+"EvtsRef").c_str(),11,0,11));
	   EventsRef_->setAxisTitle("#tau Trigger Paths");
	   EventsRef_->setBinLabel(1,"");
	   EventsRef_->setBinLabel(2,"Main Path");
	   EventsRef_->setBinLabel(3,"");
	   EventsRef_->setBinLabel(4,"noL1 Path");
	   EventsRef_->setBinLabel(5,"");
	   EventsRef_->setBinLabel(6,"no L2 Path");
	   EventsRef_->setBinLabel(7,"");
	   EventsRef_->setBinLabel(8,"no L25 Path");
	   EventsRef_->setBinLabel(9,"");
	   EventsRef_->setBinLabel(10,"no L3 Path");
	   EventsRef_->setBinLabel(11,"");
	
	   EventsPassedMatched_ = (dbe_->book1D((monitorName_+"_BitsMatched").c_str(),(monitorName_+"EvtsPassedAndMatched").c_str(),11,0,11));
	   EventsPassedMatched_->setAxisTitle("#tau Trigger Paths");
	   EventsPassedMatched_->setBinLabel(1,"");
	   EventsPassedMatched_->setBinLabel(2,"Main Path");
	   EventsPassedMatched_->setBinLabel(3,"");
	   EventsPassedMatched_->setBinLabel(4,"Main AND L1 Path");
	   EventsPassedMatched_->setBinLabel(5,"");
	   EventsPassedMatched_->setBinLabel(6,"Main AND L2 Path");
	   EventsPassedMatched_->setBinLabel(7,"");
	   EventsPassedMatched_->setBinLabel(8,"Main AND L25 Path");
	   EventsPassedMatched_->setBinLabel(9,"");
	   EventsPassedMatched_->setBinLabel(10,"Main AND L3 Path");
	   EventsPassedMatched_->setBinLabel(11,"");

	   EventsPassedNotMatched_ = (dbe_->book1D((monitorName_+"_BitsNotMatched").c_str(),(monitorName_+"EvtsPassedAndNotMatched").c_str(),11,0,11));
	   EventsPassedNotMatched_->setAxisTitle("#tau Trigger Paths");
	   EventsPassedNotMatched_->setBinLabel(1,"");
	   EventsPassedNotMatched_->setBinLabel(2,"Main Path");
	   EventsPassedNotMatched_->setBinLabel(3,"");
	   EventsPassedNotMatched_->setBinLabel(4,"Main AND L1 Path");
	   EventsPassedNotMatched_->setBinLabel(5,"");
	   EventsPassedNotMatched_->setBinLabel(6,"Main AND L2 Path");
	   EventsPassedNotMatched_->setBinLabel(7,"");
	   EventsPassedNotMatched_->setBinLabel(8,"Main AND L25 Path");
	   EventsPassedNotMatched_->setBinLabel(9,"");
	   EventsPassedNotMatched_->setBinLabel(10,"Main AND L3 Path");
	   EventsPassedNotMatched_->setBinLabel(11,"");

     }	   

	

}

//--------------------------------------------------------
void HLTTauDQMOfflineSource::beginRun(const edm::Run& r, const EventSetup& context) {

}

//--------------------------------------------------------
void HLTTauDQMOfflineSource::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
				      const EventSetup& context) {
  
}

// ----------------------------------------------------------
void 
HLTTauDQMOfflineSource::analyze(const Event& iEvent, const EventSetup& iSetup )
{  
  //Apply the prescaler
  if(counterEvt_ > prescaleEvt_)
    {

      //Do Reference Trigger Analysis
      bool tau_ok = false;
      bool lepton_ok = false;

      if(nTriggeredTaus_ ==0)
	tau_ok=true;
      if(nTriggeredLeptons_ ==0)
	lepton_ok=true;

      //Check Taus
      edm::Handle<LVColl> refTauObjects;
      if(iEvent.getByLabel(refTauObjects_,refTauObjects))
      {
	if(refTauObjects->size() >= nTriggeredTaus_)
	  tau_ok = true;
      }

      //CheckLeptons
      edm::Handle<LVColl> refLeptonObjects;
      if(iEvent.getByLabel(refLeptonObjects_,refLeptonObjects))
      {
	if(refLeptonObjects->size() >= nTriggeredLeptons_)
	  lepton_ok = true;
      }

      if(tau_ok&&lepton_ok)
	{
	  EventsRef_->Fill(1.5);
	  EventsRef_->Fill(3.5);
	  EventsRef_->Fill(5.5);
	  EventsRef_->Fill(7.5);
	  EventsRef_->Fill(9.5);

	}



      //Do Backup trigger Analysis
      //Check if passed Main Filter
      bool passedMain = false;
      bool passedMainMatched = false;
      bool passedMainNotMatched = false;


      edm::Handle<TriggerEvent> tev;
      if(iEvent.getByLabel(triggerEvent_,tev))
	if(tev.isValid())
	  {
	    if(tev->filterIndex(mainPath_)!=tev->sizeFilters())
	      {
		passedMain = true;
		EventsPassed_->Fill(1.5);
		//Get The HLTTau Collection
		LVColl HLTTaus = importFilterColl(mainPath_,tauPdgID_,iEvent);
		//get the HLTLepton Collection
		LVColl HLTLeptons = importFilterColl(mainPath_,leptonPdgID_,iEvent);
		
		//Loop On the Taus and match them
		size_t nMatchedTaus = 0;

		for(size_t i = 0;i<HLTTaus.size();++i)
		  {
		    if(refTauObjects->size()>0)
		    if(match(HLTTaus[i],*refTauObjects,corrDeltaR_))
		      {
			nMatchedTaus++;
		      }
		  }
		
		size_t nMatchedLeptons = 0;
		for(size_t i = 0;i<HLTLeptons.size();++i)
		  {
		    if(refLeptonObjects->size()>0)
		    if(match(HLTLeptons[i],*refLeptonObjects,corrDeltaR_))
		      {
			nMatchedLeptons++;
		      }
		  }
	
		if(nMatchedTaus>=nTriggeredTaus_ && nMatchedLeptons>=nTriggeredLeptons_)
		  {
		    passedMainMatched=true;
		    EventsPassedMatched_->Fill(1.5);
		  }
		else
		  {
		    passedMainNotMatched=true;
		    EventsPassedNotMatched_->Fill(1.5);
		  }
			
	      }
	    if(tev->filterIndex(l1BackupPath_)!=tev->sizeFilters())
	      {
		passedMain = true;
		EventsPassed_->Fill(3.5);
		//Get The HLTTau Collection
		LVColl HLTTaus = importFilterColl(l1BackupPath_,tauPdgID_,iEvent);
		//get the HLTLepton Collection
		LVColl HLTLeptons = importFilterColl(l1BackupPath_,leptonPdgID_,iEvent);
		
		//Loop On the Taus and match them
		size_t nMatchedTaus = 0;
		for(size_t i = 0;i<HLTTaus.size();++i)
		  {
		    if(refTauObjects->size()>0)
		    if(match(HLTTaus[i],*refTauObjects,corrDeltaR_))
		      {
			nMatchedTaus++;
		      }
		  }
		
		size_t nMatchedLeptons = 0;
		for(size_t i = 0;i<HLTLeptons.size();++i)
		  {
		    if(refLeptonObjects->size()>0)
		    if(match(HLTLeptons[i],*refLeptonObjects,corrDeltaR_))
		      {
			nMatchedLeptons++;
		      }
		  }
	
		if(nMatchedTaus>=nTriggeredTaus_ && nMatchedLeptons>=nTriggeredLeptons_ &&passedMain)
		  {
		    EventsPassedMatched_->Fill(3.5);
		  }
		else if(nMatchedTaus>=nTriggeredTaus_ && nMatchedLeptons>=nTriggeredLeptons_ &&!passedMain)
		  {
		    EventsPassedNotMatched_->Fill(3.5);
		  }

	      }
	    if(tev->filterIndex(l2BackupPath_)!=tev->sizeFilters())
	      {
		passedMain = true;
		EventsPassed_->Fill(5.5);
		//Get The HLTTau Collection
		LVColl HLTTaus = importFilterColl(l2BackupPath_,tauPdgID_,iEvent);
		//get the HLTLepton Collection
		LVColl HLTLeptons = importFilterColl(l2BackupPath_,leptonPdgID_,iEvent);
		
		//Loop On the Taus and match them
		size_t nMatchedTaus = 0;
		for(size_t i = 0;i<HLTTaus.size();++i)
		  {
		    if(refTauObjects->size()>0)
		    if(match(HLTTaus[i],*refTauObjects,corrDeltaR_))
		      {
			nMatchedTaus++;
		      }
		  }
		
		size_t nMatchedLeptons = 0;
		for(size_t i = 0;i<HLTLeptons.size();++i)
		  {
		    if(refLeptonObjects->size()>0)
		    if(match(HLTLeptons[i],*refLeptonObjects,corrDeltaR_))
		      {
			nMatchedLeptons++;
		      }
		  }
	
		if(nMatchedTaus>=nTriggeredTaus_ && nMatchedLeptons>=nTriggeredLeptons_ &&passedMain)
		  {
		    EventsPassedMatched_->Fill(5.5);
		  }
		else if(nMatchedTaus>=nTriggeredTaus_ && nMatchedLeptons>=nTriggeredLeptons_ &&!passedMain)
		  {
		    EventsPassedNotMatched_->Fill(5.5);
		  }
	      }
	    if(tev->filterIndex(l25BackupPath_)!=tev->sizeFilters())
	      {
		passedMain = true;
		EventsPassed_->Fill(7.5);
		//Get The HLTTau Collection
		LVColl HLTTaus = importFilterColl(l25BackupPath_,tauPdgID_,iEvent);
		//get the HLTLepton Collection
		LVColl HLTLeptons = importFilterColl(l25BackupPath_,leptonPdgID_,iEvent);
		
		//Loop On the Taus and match them
		size_t nMatchedTaus = 0;
		for(size_t i = 0;i<HLTTaus.size();++i)
		  {
		    if(refTauObjects->size()>0)
		    if(match(HLTTaus[i],*refTauObjects,corrDeltaR_))
		      {
			nMatchedTaus++;
		      }
		  }
		
		size_t nMatchedLeptons = 0;
		for(size_t i = 0;i<HLTLeptons.size();++i)
		  {
		    if(refLeptonObjects->size()>0)
		    if(match(HLTLeptons[i],*refLeptonObjects,corrDeltaR_))
		      {
			nMatchedLeptons++;
		      }
		  }
	
		if(nMatchedTaus>=nTriggeredTaus_ && nMatchedLeptons>=nTriggeredLeptons_ &&passedMain)
		  {
		    EventsPassedMatched_->Fill(7.5);
		  }
		else if(nMatchedTaus>=nTriggeredTaus_ && nMatchedLeptons>=nTriggeredLeptons_ &&!passedMain)
		  {
		    EventsPassedNotMatched_->Fill(7.5);
		  }
	      }
	    if(tev->filterIndex(l3BackupPath_)!=tev->sizeFilters())
	      { 
		passedMain = true;
		EventsPassed_->Fill(10);
		//Get The HLTTau Collection
		LVColl HLTTaus = importFilterColl(l3BackupPath_,tauPdgID_,iEvent);
		//get the HLTLepton Collection
		LVColl HLTLeptons = importFilterColl(l3BackupPath_,leptonPdgID_,iEvent);
		
		//Loop On the Taus and match them
		size_t nMatchedTaus = 0;
		for(size_t i = 0;i<HLTTaus.size();++i)
		  {
		    if(refTauObjects->size()>0)
		    if(match(HLTTaus[i],*refTauObjects,corrDeltaR_))
		      {
			nMatchedTaus++;
		      }
		  }
		
		size_t nMatchedLeptons = 0;
		for(size_t i = 0;i<HLTLeptons.size();++i)
		  {
		    if(match(HLTLeptons[i],*refLeptonObjects,corrDeltaR_))
		      {
			nMatchedLeptons++;
		      }
		  }
	
		if(nMatchedTaus>=nTriggeredTaus_ && nMatchedLeptons>=nTriggeredLeptons_ &&passedMain)
		  {
		    EventsPassedMatched_->Fill(9.5);
		  }
		else if(nMatchedTaus>=nTriggeredTaus_ && nMatchedLeptons>=nTriggeredLeptons_ &&!passedMain)
		  {
		    EventsPassedNotMatched_->Fill(9.5);
		  }

		}
	  }
      
      counterEvt_ = 0;
    }
  else
      counterEvt_++;

}




//--------------------------------------------------------
void 
HLTTauDQMOfflineSource::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
				    const EventSetup& context) {
}
//--------------------------------------------------------
void 
HLTTauDQMOfflineSource::endRun(const Run& r, const EventSetup& context){
}
//--------------------------------------------------------
void 
HLTTauDQMOfflineSource::endJob(){

   if (outputFile_.size() != 0 && dbe_)
   dbe_->save(outputFile_);
 
  return;
}




bool 
HLTTauDQMOfflineSource::match(const LV& cand,const  LVColl&  electrons,double deltaR)
{
    bool matched=false;
    for(size_t i = 0;i<electrons.size();++i)
      {
	double delta = ROOT::Math::VectorUtil::DeltaR(cand.Vect(),electrons[i].Vect());
	if((delta<deltaR))
	  {
	    matched=true;
	  }
      }

    return matched;

}

LVColl 
HLTTauDQMOfflineSource::importFilterColl(edm::InputTag& filter,int pdgID,const Event& iEvent)
{
    //Create output Collection
    LVColl out;

    //Look at all Different triggers
    Handle<TriggerEvent> handle;
    if(iEvent.getByLabel(triggerEvent_,handle))
      {

	//get All the final trigger objects
	const TriggerObjectCollection& TOC(handle->getObjects());
      
	//filter index
	size_t index = handle->filterIndex(filter);
	if(index!=handle->sizeFilters())
	  {
	    const Keys& KEYS = handle->filterKeys(index);
	    for(size_t i = 0;i<KEYS.size();++i)
	      {
		const TriggerObject& TO(TOC[KEYS[i]]);
		LV a(TO.px(),TO.py(),TO.pz(),TO.energy());
		if(abs(TO.id()) == pdgID)
		  out.push_back(a);
	      }
	  }
      }
    return out;
}

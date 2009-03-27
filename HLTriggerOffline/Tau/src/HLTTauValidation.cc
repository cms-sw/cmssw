#include "HLTriggerOffline/Tau/interface/HLTTauValidation.h"
#include "Math/GenVector/VectorUtil.h"

HLTTauValidation::HLTTauValidation(const edm::ParameterSet& ps) : 
  triggerEventObject_(ps.getUntrackedParameter<edm::InputTag>("triggerEventObject")),
  refCollection_(ps.getUntrackedParameter<edm::InputTag>("refTauCollection")),
  refLeptonCollection_(ps.getUntrackedParameter<edm::InputTag>("refLeptonCollection")),
  triggerTag_(ps.getUntrackedParameter<std::string>("DQMFolder","DoubleTau")),
  filter_(ps.getUntrackedParameter<std::vector<edm::InputTag> >("Filter")),
  TauType_(ps.getUntrackedParameter<std::vector<int> >("TauType")),
  LeptonType_(ps.getUntrackedParameter<std::vector<int> >("LeptonType")),
  nTriggeredTaus_(ps.getUntrackedParameter<std::vector<unsigned> >("NTriggeredTaus")),
  nTriggeredLeptons_(ps.getUntrackedParameter<std::vector<unsigned> >("NTriggeredLeptons")),
  doRefAnalysis_(ps.getUntrackedParameter<bool>("DoReferenceAnalysis",false)),
  matchDeltaR_(ps.getUntrackedParameter<std::vector<double> >("MatchDeltaR"))
{
  //initialize 

  //  for(size_t k=0;k<filter_.size();++k)
  //   NEventsPassed.push_back(0);

  //  for(size_t k=0;k<=filter_.size();++k)
  //    NEventsPassedMatched.push_back(0);

  //Declare DQM Store
  DQMStore* store = &*edm::Service<DQMStore>();

  if(store)
    {
      //Create the histograms
      store->setCurrentFolder(triggerTag_);
      accepted_events = store->book1D("Triggers","Accepted Events per path",filter_.size(),0,filter_.size());
      for(size_t k=0;k<filter_.size();++k)
	accepted_events->setBinLabel(k+1,filter_[k].label(),1);

      if(doRefAnalysis_)
	{
	  accepted_events_matched = store->book1D("MatchedTriggers","Accepted +Matched Events per path",filter_.size()+1,0,filter_.size()+1);
	  accepted_events_matched->setBinLabel(1,"RefEvents",1);
	  for(size_t k=0;k<filter_.size();++k)
	    accepted_events_matched->setBinLabel(k+2,filter_[k].label(),1);
	}



    }

}

HLTTauValidation::~HLTTauValidation()
{
}

//
// member functions
//

void
HLTTauValidation::endJob()
{
  //Fill histograms
  //  for(size_t i=0;i<filter_.size();++i)
  //     accepted_events->setBinContent(i+1,NEventsPassed[i]);


  //For the matched events the zeroth bin is used to put the reference events
  // for(size_t i=0;i<filter_.size();++i)
  //    accepted_events_matched->setBinContent(i+1,NEventsPassedMatched[i+1]);
}


void
HLTTauValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace l1extra;
  using namespace trigger;

  //Look at the reference collection (i.e MC)
  Handle<LVColl> refC;
  Handle<LVColl> refCL;

  bool tau_ok=true;
  bool lepton_ok=true;

  bool isGoodReferenceEvent=false;

  if(doRefAnalysis_)
    {
      //Tau reference
      if(iEvent.getByLabel(refCollection_,refC))
	if(refC->size()<nTriggeredTaus_[0])
	  {
	    tau_ok = false;
	  }
	else
	  {
	    double et=0.;
	    double eta=0.;
	    
	    for(size_t j = 0;j<refC->size();++j)
	      {
		if((*refC)[j].Et()>et)
		  {
		    et=(*refC)[j].Et();
		    eta = (*refC)[j].Eta();
		  }
		
	      }
	 
	}
  
      //lepton reference
      if(iEvent.getByLabel(refLeptonCollection_,refCL))
	if(refCL->size()<nTriggeredLeptons_[0])
	  {
	    lepton_ok = false;
	  }
    
      
      if(lepton_ok&&tau_ok)
	{
	  accepted_events_matched->Fill(0.5);
	  isGoodReferenceEvent=true;
	}
    }


  //get The triggerEvent


  Handle<TriggerEventWithRefs> trigEv;
  if(iEvent.getByLabel(triggerEventObject_,trigEv))
    {

      if (trigEv.isValid()) 
	{

	  //Loop through the filters
	  for(size_t i=0;i<filter_.size();++i)
	    {
	
	      size_t ID =trigEv->filterIndex(filter_[i]);
	      if(ID!=trigEv->size())
		{
		      LVColl leptons = getFilterCollection(ID,LeptonType_[i],*trigEv);
		      LVColl taus = getFilterCollection(ID,TauType_[i],*trigEv);

		      //Fired
		      if(leptons.size()>=nTriggeredLeptons_[i+1]&&taus.size()>=nTriggeredTaus_[i+1])
			{
			  accepted_events->Fill(i+0.5);
			  //Now do the matching only though if we have a good reference event
			  if(doRefAnalysis_)
			  if(isGoodReferenceEvent)
			    {
			      
			      size_t nT=0;
				for(size_t j=0;j<taus.size();++j)
				  {
				    if(match(taus[j],*refC,matchDeltaR_[i]))
				      nT++;

				  }
			      size_t nL=0;
				for(size_t j=0;j<leptons.size();++j)
				  {
				    if(match(leptons[j],*refCL,matchDeltaR_[i]))
				      nL++;

				  }
				if(nT>=nTriggeredTaus_[i+1]&&nL>=nTriggeredLeptons_[i+1])
				  accepted_events_matched->Fill(i+1.5);
			}
		    }
		}
	    }
	}
    }
}





bool 
HLTTauValidation::match(const LV& jet,const LVColl& McInfo,double dr)
{
 
  bool matched=false;

  if(McInfo.size()>0)
    for(std::vector<LV>::const_iterator it = McInfo.begin();it!=McInfo.end();++it)
      {
	double delta = ROOT::Math::VectorUtil::DeltaR(jet,*it);
	if(delta<dr)
	  {
	    matched=true;
	  }
      }

  return matched;
}



LVColl HLTTauValidation::getFilterCollection(size_t filterID,int id,const trigger::TriggerEventWithRefs& trigEv)
{
  using namespace trigger;

  LVColl out;


	  if(id==trigger::TriggerL1IsoEG ||trigger::TriggerL1NoIsoEG) 
	    {
	      VRl1em obj;
	      trigEv.getObjects(filterID,id,obj);
	      for(size_t i=0;i<obj.size();++i)
		    out.push_back(obj[i]->p4());
	    }

	  if(id==trigger::TriggerL1Mu) 
	    {
	      VRl1muon obj;
	      trigEv.getObjects(filterID,id,obj);
	      for(size_t i=0;i<obj.size();++i)
		    out.push_back(obj[i]->p4());
	    }


	  if(id==trigger::TriggerMuon) 
	    {
	      VRmuon obj;
	      trigEv.getObjects(filterID,id,obj);
	      for(size_t i=0;i<obj.size();++i)
		    out.push_back(obj[i]->p4());
	    }

	  if(id==trigger::TriggerElectron) 
	    {
	      VRelectron obj;
	      trigEv.getObjects(filterID,id,obj);
	      for(size_t i=0;i<obj.size();++i)
		    out.push_back(obj[i]->p4());
	    }

	  if(id==trigger::TriggerL1TauJet) 
	    {
	      VRl1jet obj;
	      trigEv.getObjects(filterID,id,obj);
	      for(size_t i=0;i<obj.size();++i)
		    out.push_back(obj[i]->p4());
	    }

	  if(id==trigger::TriggerTau) 
	    {
	      VRjet obj;
	      trigEv.getObjects(filterID,id,obj);
	      for(size_t i=0;i<obj.size();++i)
		    out.push_back(obj[i]->p4());
	    }

	  return out;
}

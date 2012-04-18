
#include "DQM/HLTEvF/interface/HLTTauDQMPathPlotter.h"
#include "Math/GenVector/VectorUtil.h"

HLTTauDQMPathPlotter::HLTTauDQMPathPlotter(const edm::ParameterSet& ps,bool ref) : 
  triggerEventObject_(ps.getUntrackedParameter<edm::InputTag>("triggerEventObject")),
  triggerTag_(ps.getUntrackedParameter<std::string>("DQMFolder","DoubleTau")),
  filter_(ps.getUntrackedParameter<std::vector<edm::InputTag> >("Filter")),
  TauType_(ps.getUntrackedParameter<std::vector<int> >("TauType")),
  LeptonType_(ps.getUntrackedParameter<std::vector<int> >("LeptonType")),
  nTriggeredTaus_(ps.getUntrackedParameter<std::vector<unsigned> >("NTriggeredTaus")),
  nTriggeredLeptons_(ps.getUntrackedParameter<std::vector<unsigned> >("NTriggeredLeptons")),
  doRefAnalysis_(ref),
  matchDeltaR_(ps.getUntrackedParameter<std::vector<double> >("MatchDeltaR")),
  refTauPt_(ps.getUntrackedParameter<double>("refTauPt",20)),
  refLeptonPt_(ps.getUntrackedParameter<double>("refLeptonPt",15))
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
      accepted_events = store->book1D("TriggerBits","Accepted Events per path",filter_.size(),0,filter_.size());
      for(size_t k=0;k<filter_.size();++k)
	accepted_events->setBinLabel(k+1,filter_[k].label(),1);

      if(doRefAnalysis_)
	{
	  accepted_events_matched = store->book1D("MatchedTriggerBits","Accepted +Matched Events per path",filter_.size()+1,0,filter_.size()+1);
	  accepted_events_matched->setBinLabel(1,"RefEvents",1);
	  for(size_t k=0;k<filter_.size();++k)
	    accepted_events_matched->setBinLabel(k+2,filter_[k].label(),1);
	}



    }

}

HLTTauDQMPathPlotter::~HLTTauDQMPathPlotter()
{
}

//
// member functions
//

void
HLTTauDQMPathPlotter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup,const std::vector<LVColl>& refC )
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace l1extra;
  using namespace trigger;

  bool tau_ok=true;
  bool lepton_ok=true;

  bool isGoodReferenceEvent=false;

  if(doRefAnalysis_)
    {
      //Tau reference
      if(refC.size()>0)
	{
	  if(refC[0].size()<nTriggeredTaus_[0])
	    {
	      tau_ok = false;
	    }
	  else
	    {
	      unsigned int highPtTaus=0;
	      for(size_t j = 0;j<refC[0].size();++j)
		{
		  if((refC[0])[j].Et()>refTauPt_)
		    highPtTaus++;
		}
	      if(highPtTaus<nTriggeredTaus_[0])
		{
		  tau_ok = false;
		}
	 

	    }
	}
    //lepton reference
    unsigned int highPtElectrons=0;
    unsigned int highPtMuons=0;
    if(refC.size()>1){	  	
	  	for(size_t j = 0;j<refC[1].size();++j)
	    	{
	      	if((refC[1])[j].Et()>refLeptonPt_)	      
			highPtElectrons++;
	    	}
	}
	if(refC.size()>2){
	  	for(size_t j = 0;j<refC[2].size();++j)
	    	{
	      	if((refC[2])[j].Et()>refLeptonPt_)	      
				highPtMuons++;
	    	}
	}
	if(highPtElectrons<nTriggeredLeptons_[0]&&LeptonType_[1]==82)
	{
	      lepton_ok = false;
	}
	if(highPtMuons<nTriggeredLeptons_[0]&&LeptonType_[1]==83)
	{
	    lepton_ok = false;
	}
	if(lepton_ok&&tau_ok)
	  {
	    accepted_events_matched->Fill(0.5);
	    isGoodReferenceEvent=true;
	  }


}


  Handle<TriggerEventWithRefs> trigEv;
  bool   gotTEV=iEvent.getByLabel(triggerEventObject_,trigEv) &&trigEv.isValid();

  if(gotTEV)
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
		      if(leptons.size()>=nTriggeredLeptons_[i+1] && taus.size()>=nTriggeredTaus_[i+1])
			{
			  accepted_events->Fill(i+0.5);
			  //Now do the matching only though if we have a good reference event
			  if(doRefAnalysis_)
			  if(isGoodReferenceEvent)
			    {

			      size_t nT=0;
				for(size_t j=0;j<taus.size();++j)
				  {
				    if(match(taus[j],refC[0],matchDeltaR_[i]))
				      nT++;

				  }
			      size_t nL=0;
				for(size_t j=0;j<leptons.size();++j)
				  {
			  		if(refC[1].size()>0){
			  			
				    	if(match(leptons[j],refC[1],matchDeltaR_[i]))
				      	nL++;
				    }
    
				    if(refC[2].size()>0){
				    	if(match(leptons[j],refC[2],matchDeltaR_[i]))
				      	nL++;
				    }

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
HLTTauDQMPathPlotter::match(const LV& jet,const LVColl& McInfo,double dr)
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



LVColl HLTTauDQMPathPlotter::getFilterCollection(size_t filterID,int id,const trigger::TriggerEventWithRefs& trigEv)
{
  using namespace trigger;

  LVColl out;


	  if(id==trigger::TriggerL1IsoEG ||trigger::TriggerL1NoIsoEG) 
	    {
	      VRl1em obj;
	      trigEv.getObjects(filterID,id,obj);
	      for(size_t i=0;i<obj.size();++i)
		if(obj.at(i).isAvailable())
		    out.push_back(obj[i]->p4());
	    }

	  if(id==trigger::TriggerL1Mu) 
	    {
	      VRl1muon obj;
	      trigEv.getObjects(filterID,id,obj);
	      for(size_t i=0;i<obj.size();++i)
		if(obj.at(i).isAvailable())

		    out.push_back(obj[i]->p4());
	    }


	  if(id==trigger::TriggerMuon) 
	    {
	      VRmuon obj;
	      trigEv.getObjects(filterID,id,obj);
	      for(size_t i=0;i<obj.size();++i)
		if(obj.at(i).isAvailable())

		    out.push_back(obj[i]->p4());
	    }

	  if(id==trigger::TriggerElectron) 
	    {
	      VRelectron obj;
	      trigEv.getObjects(filterID,id,obj);
	      for(size_t i=0;i<obj.size();++i)
		if(obj.at(i).isAvailable())
		    out.push_back(obj[i]->p4());
	    }

	  if(id==trigger::TriggerL1TauJet) 
	    {
	      VRl1jet obj;
	      trigEv.getObjects(filterID,id,obj);
	      for(size_t i=0;i<obj.size();++i)
		if(obj.at(i).isAvailable())
		    out.push_back(obj[i]->p4());
	      trigEv.getObjects(filterID,trigger::TriggerL1CenJet,obj);
	      for(size_t i=0;i<obj.size();++i)
		if(obj.at(i).isAvailable())
		    out.push_back(obj[i]->p4());

	    }

	  if(id==trigger::TriggerTau) 
	    {
	      VRjet obj;
	      trigEv.getObjects(filterID,id,obj);
	      for(size_t i=0;i<obj.size();++i)
		if(obj.at(i).isAvailable())
		    out.push_back(obj[i]->p4());
	    }

	  return out;
}

#include "DQM/HLTEvF/interface/HLTTauDQMLitePathPlotter.h"
#include "Math/GenVector/VectorUtil.h"

HLTTauDQMLitePathPlotter::HLTTauDQMLitePathPlotter(const edm::ParameterSet& ps,int etbins,int etabins,int phibins,double maxpt,bool ref,double dr): 
  triggerEvent_(ps.getUntrackedParameter<edm::InputTag>("triggerEventObject")),
  triggerTag_(ps.getUntrackedParameter<std::string>("DQMFolder","DoubleTau")),
  filter_(ps.getUntrackedParameter<std::vector<edm::InputTag> >("Filter")),
  name_(ps.getUntrackedParameter<std::vector<std::string> >("PathName")),
  TauType_(ps.getUntrackedParameter<std::vector<int> >("TauType")),
  LeptonType_(ps.getUntrackedParameter<std::vector<int> >("LeptonType")),
  nTriggeredTaus_(ps.getUntrackedParameter<std::vector<unsigned> >("NTriggeredTaus")),
  nTriggeredLeptons_(ps.getUntrackedParameter<std::vector<unsigned> >("NTriggeredLeptons")),
  doRefAnalysis_(ref),
  matchDeltaR_(dr),
  maxEt_(maxpt),
  binsEt_(etbins),
  binsEta_(etabins),
  binsPhi_(phibins)
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
      accepted_events = store->book1D("PathTriggerBits","Accepted Events per path",filter_.size(),0,filter_.size());
      for(size_t k=0;k<filter_.size();++k)
	{
	  accepted_events->setBinLabel(k+1,name_[k],1);
	    if( nTriggeredTaus_[k]>=2 ||(nTriggeredTaus_[k]>=1 && nTriggeredLeptons_[k]>=1))  
	    mass_distribution.push_back(store->book1D(("mass_"+name_[k]),("Mass Distribution for "+name_[k]),100,0,500)); 
	}
      

      if(doRefAnalysis_)
	{
	  accepted_events_matched = store->book1D("MatchedPathTriggerBits","Accepted +Matched Events per path",filter_.size(),0,filter_.size());
	  accepted_events_matched->getTH1F()->Sumw2();

	  for(size_t k=0;k<filter_.size();++k)
	    accepted_events_matched->setBinLabel(k+1,name_[k],1);

	  ref_events = store->book1D("RefEvents","Reference Events per path",filter_.size(),0,filter_.size());
	  ref_events->getTH1F()->Sumw2();

	  for(size_t k=0;k<filter_.size();++k)
	    ref_events->setBinLabel(k+1,name_[k],1);
	}


      tauEt          = store->book1D("TrigTauEt","#tau E_{t}",binsEt_,0,maxEt_);
      tauEta         = store->book1D("TrigTauEta","#tau #eta",binsEta_,-2.5,2.5);
      tauPhi         = store->book1D("TrigTauPhi","#tau #phi",binsPhi_,-3.2,3.2);

      if(doRefAnalysis_)
	{

	  tauEtEffNum   = store->book1D("TrigTauEtEffNum"," #tau E_{t} Efficiency",binsEt_,0,maxEt_);
	  tauEtEffNum->getTH1F()->Sumw2();

	  tauEtEffDenom = store->book1D("TrigTauEtEffDenom"," #tau E_{t} Denominator",binsEt_,0,maxEt_);
	  tauEtEffDenom->getTH1F()->Sumw2();

	  tauEtaEffNum   = store->book1D("TrigTauEtaEffNum"," #tau #eta Efficiency",binsEta_,-2.5,2.5);
	  tauEtaEffNum->getTH1F()->Sumw2();

	  tauEtaEffDenom = store->book1D("TrigTauEtaEffDenom"," #tau #eta Denominator",binsEta_,-2.5,2.5);
	  tauEtaEffDenom->getTH1F()->Sumw2();

	  tauPhiEffNum   = store->book1D("TrigTauPhiEffNum"," #tau #phi Efficiency",binsPhi_,-3.2,3.2);
	  tauPhiEffNum->getTH1F()->Sumw2();

	  tauPhiEffDenom = store->book1D("TrigTauPhiEffDenom"," #tau #phi Denominator",binsPhi_,-3.2,3.2);
	  tauPhiEffDenom->getTH1F()->Sumw2();

	}
    }
}

HLTTauDQMLitePathPlotter::~HLTTauDQMLitePathPlotter()
{
}

//
// member functions
//

void
HLTTauDQMLitePathPlotter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup,const std::vector<LVColl>& refC)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace l1extra;
  using namespace trigger;




  std::vector<bool> isGoodReferenceEvent;


  //Fill ref collection for the filters
  if(doRefAnalysis_)
    for(size_t i=0;i<filter_.size();++i)
      {
	bool tau_ok=true;
	bool lepton_ok=true;
	//Tau reference
	if(refC[0].size()<nTriggeredTaus_[i])
	  {
	    tau_ok = false;
	  }

  
      //lepton reference
	if(refC[1].size()<nTriggeredLeptons_[i])
	  {
	    lepton_ok = false;
	  }
    
      
      if(lepton_ok&&tau_ok)
	{
	  ref_events->Fill(i+0.5);
	  isGoodReferenceEvent.push_back(true);
	}
      else
	  isGoodReferenceEvent.push_back(false);
    }


  //get The triggerEvent


  Handle<TriggerEvent> trigEv;
  if(iEvent.getByLabel(triggerEvent_,trigEv))
    {

      if (trigEv.isValid()) 
	{


	  //Loop through the filters
	  for(size_t i=0;i<filter_.size();++i)
	    {
	
	      size_t ID =trigEv->filterIndex(filter_[i]);
	      if(ID!=trigEv->sizeFilters())
		{
		      LVColl leptons = getFilterCollection(ID,LeptonType_[i],*trigEv);
		      LVColl taus = getFilterCollection(ID,TauType_[i],*trigEv);
		      //Fired
		      
		      if(leptons.size()>=nTriggeredLeptons_[i] && taus.size()>=nTriggeredTaus_[i])
			{

			  accepted_events->Fill(i+0.5);
			  //Now do the matching only though if we have a good reference event

			  if(doRefAnalysis_)
			    {
			      if(isGoodReferenceEvent[i])
				{
			      
				  size_t nT=0;
				  for(size_t j=0;j<taus.size();++j)
				    {
				      std::pair<bool,LV> m=match(taus[j],refC[0],matchDeltaR_); 
				      if(m.first)
					nT++;
				    }
				  size_t nL=0;
				  for(size_t j=0;j<leptons.size();++j)
				    {
				      if(match(leptons[j],refC[1],matchDeltaR_).first)
					nL++;
				      
				    }
				  if(nT>=nTriggeredTaus_[i]&&nL>=nTriggeredLeptons_[i])
				    {
				      accepted_events_matched->Fill(i+0.5);
				      if(nTriggeredTaus_[i]>=2)
					  mass_distribution[i]->Fill(((refC[0])[0]+(refC[0])[1]).M());
				      else if(nTriggeredTaus_[i]>=1 && nTriggeredLeptons_[i]>=1)
					mass_distribution[i]->Fill(((refC[0])[0]+(refC[1])[0]).M());
				    }
				}



			    }
			  else
			    {
      				      if(nTriggeredTaus_[i]>=2)
					  mass_distribution[i]->Fill((taus[0]+taus[1]).M());
				      else if(nTriggeredTaus_[i]>=1 && nTriggeredLeptons_[i]>=1)
					  mass_distribution[i]->Fill((taus[0]+leptons[0]).M());
			    }
			}
		}
	    }
	
       


	  //Do the object thing

	  LVColl taus = getObjectCollection(15,*trigEv);
	  if(!doRefAnalysis_)
	  for(unsigned int tau=0;tau<taus.size();++tau)
	    {
		  tauEt->Fill(taus[tau].pt());
		  tauEta->Fill(taus[tau].eta());
		  tauPhi->Fill(taus[tau].phi());
	    }


	  if(doRefAnalysis_)
	    for(unsigned int tau=0;tau<refC[0].size();++tau)
	    {
	          tauEtEffDenom->Fill((refC[0])[tau].pt());
	          tauEtaEffDenom->Fill((refC[0])[tau].eta());
	          tauPhiEffDenom->Fill((refC[0])[tau].phi());

		  std::pair<bool,LV> m=match((refC[0])[tau],taus,matchDeltaR_); 
		  if(m.first)
		    {
		      tauEt->Fill(m.second.pt());
		      tauEta->Fill(m.second.eta());
		      tauPhi->Fill(m.second.phi());

		      tauEtEffNum->Fill((refC[0])[tau].pt());
		      tauEtaEffNum->Fill((refC[0])[tau].eta());
		      tauPhiEffNum->Fill((refC[0])[tau].phi());

		    }
	    }
	}
    }
}

	





std::pair<bool,LV> 
HLTTauDQMLitePathPlotter::match(const LV& jet,const LVColl& McInfo,double dr)
{
 
  bool matched=false;
  LV out;

  if(McInfo.size()>0)
    for(std::vector<LV>::const_iterator it = McInfo.begin();it!=McInfo.end();++it)
      {
	double delta = ROOT::Math::VectorUtil::DeltaR(jet,*it);
	if(delta<dr)
	  {
	    matched=true;
	    out = *it;
	  }
      }

  std::pair<bool,LV> o = std::make_pair(matched,out);
  return o;
}


LVColl 
HLTTauDQMLitePathPlotter::getObjectCollection(int id,const trigger::TriggerEvent& trigEv)
{
  using namespace edm;
  using namespace trigger;


      TriggerObjectCollection triggerObjects;
	triggerObjects = trigEv.getObjects();

	LVColl out;
	for(unsigned int i=0;i<triggerObjects.size();++i)
	  if(abs(triggerObjects[i].id()) == id)
	   {
	     LV a(triggerObjects[i].px(),triggerObjects[i].py(),triggerObjects[i].pz(),triggerObjects[i].energy());
	     out.push_back(a);
	   }


	return out;
}


LVColl 
HLTTauDQMLitePathPlotter::getFilterCollection(size_t index,int id,const trigger::TriggerEvent& trigEv)
{
  using namespace trigger;

  //Create output Collection
  LVColl out;
      //get All the final trigger objects
      const TriggerObjectCollection& TOC(trigEv.getObjects());
     
      //filter index
      if(index!=trigEv.sizeFilters())
	{
	  const Keys& KEYS = trigEv.filterKeys(index);
	  for(size_t i = 0;i<KEYS.size();++i)
	    {
	      const TriggerObject& TO(TOC[KEYS[i]]);
	      LV a(TO.px(),TO.py(),TO.pz(),sqrt(TO.px()*TO.px()+TO.py()*TO.py()+TO.pz()*TO.pz()));
	      if(abs(TO.id()) == id)
		out.push_back(a);
	    }
	}

  return out;
}



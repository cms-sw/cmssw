#include "DQM/HLTEvF/interface/HLTTauDQMSource.h"


//
// constructors and destructor
//
HLTTauDQMSource::HLTTauDQMSource( const edm::ParameterSet& ps ) :counterEvt_(0)
{

  //Get General Monitoring Parameters
  config_                 = ps.getParameter<std::vector<edm::ParameterSet> >("MonitorSetup");
  configType_             = ps.getParameter<std::vector<std::string> >("ConfigType");
  doRefAnalysis_          = ps.getParameter<bool>("doMatching");
  NPtBins_                = ps.getUntrackedParameter<int>("PtHistoBins",20);
  NEtaBins_               = ps.getUntrackedParameter<int>("EtaHistoBins",20);
  NPhiBins_               = ps.getUntrackedParameter<int>("PhiHistoBins",32);
  EtMax_                  = ps.getUntrackedParameter<double>("EtHistoMax",100);
  L1MatchDr_              = ps.getUntrackedParameter<double>("L1MatchDeltaR",0.5);
  HLTMatchDr_             = ps.getUntrackedParameter<double>("HLTMatchDeltaR",0.3);

  refFilter_              = ps.getUntrackedParameter<std::vector<edm::InputTag> >("matchFilter");
  refID_                  = ps.getUntrackedParameter<std::vector<int> >("matchObjectID");
  ptThres_                = ps.getUntrackedParameter<std::vector<double> >("matchObjectMinPt");
  prescaleEvt_            = ps.getUntrackedParameter<int>("prescaleEvt", -1);

  triggerEvent_           = ps.getParameter < edm::InputTag > ("TriggerEvent");

  //Read The Configuration
  for(unsigned int i=0;i<config_.size();++i)
    {

      if(configType_[i] == "L1")
	{
	  HLTTauDQML1Plotter tmp(config_[i],NPtBins_,NEtaBins_,NPhiBins_,EtMax_,doRefAnalysis_,L1MatchDr_);
	  l1Plotters.push_back(tmp);
	}

      if(configType_[i] == "Calo")
	{
	  HLTTauDQMCaloPlotter tmp(config_[i],NPtBins_,NEtaBins_,NPhiBins_,EtMax_,doRefAnalysis_,HLTMatchDr_);
	  caloPlotters.push_back(tmp);
	}

      else if(configType_[i] == "Track")
	{
	  HLTTauDQMTrkPlotter tmp(config_[i],NPtBins_,NEtaBins_,NPhiBins_,EtMax_,doRefAnalysis_,HLTMatchDr_);
	  trackPlotters.push_back(tmp);
	}

      else if(configType_[i] == "Path")
	{
	  HLTTauDQMPathPlotter tmp(config_[i],doRefAnalysis_);
	  pathPlotters.push_back(tmp);
	}

      else if(configType_[i] == "LitePath")
	{
	  HLTTauDQMLitePathPlotter tmp(config_[i],NPtBins_,NEtaBins_,NPhiBins_,EtMax_,doRefAnalysis_,HLTMatchDr_);
	  litePathPlotters.push_back(tmp);
	}


      else if(configType_[i] == "Summary")
	{
	  HLTTauDQMSummaryPlotter tmp(config_[i]);
	  summaryPlotters.push_back(tmp);
	}


    }
}

HLTTauDQMSource::~HLTTauDQMSource()
{
   
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}


//--------------------------------------------------------
void 
HLTTauDQMSource::beginJob(){

}

//--------------------------------------------------------
void HLTTauDQMSource::beginRun(const edm::Run& r, const edm::EventSetup& context) {

}

//--------------------------------------------------------
void HLTTauDQMSource::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
					   const edm::EventSetup& context) {
  
}

// ----------------------------------------------------------
void 
HLTTauDQMSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup )
{  
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace l1extra;
  using namespace trigger;


  //Apply the prescaler
  if(counterEvt_ > prescaleEvt_)
    {
      //Do Analysis here
      
      //Create dummy Match Collections
      std::vector<LVColl> refC;

      if(doRefAnalysis_)
	{
	  Handle<TriggerEvent> trigEv;

	  //get The triggerEvent
	  bool gotTEV =true;
	  try {
	     gotTEV*= iEvent.getByLabel(triggerEvent_,trigEv);
	      }
	  catch (cms::Exception& exception) {
	    gotTEV =false;
	  }



	    if(gotTEV)
	      for(unsigned int i=0;i<refFilter_.size();++i)
		{
		  size_t ID =trigEv->filterIndex(refFilter_[i]);
		  refC.push_back(getFilterCollection(ID,refID_[i],*trigEv,ptThres_[i]));
		}
	}

      //fill the empty slots with empty collections
      LVColl dummy;
      for(int k=refFilter_.size();k<3;k++)
	{
	  refC.push_back(dummy);
	}

     
      
      //Path Plotters
      for(unsigned int i=0;i<pathPlotters.size();++i)
	pathPlotters[i].analyze(iEvent,iSetup,refC);

      //Lite Path Plotters
      for(unsigned int i=0;i<litePathPlotters.size();++i)
	litePathPlotters[i].analyze(iEvent,iSetup,refC);

      //L1  Plotters
      for(unsigned int i=0;i<l1Plotters.size();++i)
	l1Plotters[i].analyze(iEvent,iSetup,refC);

      //Calo Plotters
      for(unsigned int i=0;i<caloPlotters.size();++i)
	caloPlotters[i].analyze(iEvent,iSetup,refC[0]);

      //Track Plotters
      for(unsigned int i=0;i<trackPlotters.size();++i)
	trackPlotters[i].analyze(iEvent,iSetup,refC[0]);

    }
  else
      counterEvt_++;

}




//--------------------------------------------------------
void HLTTauDQMSource::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
					 const edm::EventSetup& context) {
					 
	//Summary Plotters
     for(unsigned int i=0;i<summaryPlotters.size();++i) summaryPlotters[i].plot();
     
}
//--------------------------------------------------------
void HLTTauDQMSource::endRun(const edm::Run& r, const edm::EventSetup& context){
}
//--------------------------------------------------------
void HLTTauDQMSource::endJob(){
  return;
}


LVColl 
HLTTauDQMSource::getFilterCollection(size_t index,int id,const trigger::TriggerEvent& trigEv,double ptCut)
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
		if(a.pt()>ptCut)
		  out.push_back(a);
	    }
	}
  return out;
}




//LVColl HLTTauDQMSource::getFilterCollection(size_t filterID,int id,const trigger::TriggerEventWithRefs& trigEv,double ptMin)
//{
//  using namespace trigger;
//
//  LVColl out;
//
//
//	  if(id==trigger::TriggerL1IsoEG ||trigger::TriggerL1NoIsoEG) 
//	    {
//	      VRl1em obj;
//	      trigEv.getObjects(filterID,id,obj);
//	      for(size_t i=0;i<obj.size();++i)
//		if(&obj[i])
//		  if(obj[i]->pt()>ptMin)
//		    out.push_back(obj[i]->p4());
//	    }
//
//	  if(id==trigger::TriggerL1Mu) 
//	    {
//	      VRl1muon obj;
//	      trigEv.getObjects(filterID,id,obj);
//	      for(size_t i=0;i<obj.size();++i)
//		if(&obj[i])
//		  if(obj[i]->pt()>ptMin)
//		    out.push_back(obj[i]->p4());
//	    }
//
//
//	   if(id==trigger::TriggerMuon) 
//	    {
//	      VRmuon obj;
//	      trigEv.getObjects(filterID,id,obj);
//	      for(size_t i=0;i<obj.size();++i)
//		if(&obj[i])
//	  if(obj[i]->pt()>ptMin)
//		    out.push_back(obj[i]->p4());
//	    }
//
//	  if(id==trigger::TriggerElectron) 
//	    {
//	      VRelectron obj;
//	      trigEv.getObjects(filterID,id,obj);
//	      for(size_t i=0;i<obj.size();++i)
//		if(&obj[i])
//		  if(obj[i]->pt()>ptMin)
//		    out.push_back(obj[i]->p4());
//	    }

//	  if(id==trigger::TriggerL1TauJet) 
//	    {
//	      VRl1jet obj;
//	      trigEv.getObjects(filterID,id,obj);
//	      for(size_t i=0;i<obj.size();++i)
//		if(&obj[i])
//		  if(obj[i]->pt()>ptMin)
//		    out.push_back(obj[i]->p4());
//	    }

//	  if(id==trigger::TriggerTau) 
//	    {
//	      VRjet obj;
//	      trigEv.getObjects(filterID,id,obj);
//	      for(size_t i=0;i<obj.size();++i)
//		if(&obj[i])
//		  if(obj[i]->pt()>ptMin)
//		    out.push_back(obj[i]->p4());
//	    }

//	  return out;
//

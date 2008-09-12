#include "HLTriggerOffline/Tau/interface/HLTTauValidation.h"
#include "Math/GenVector/VectorUtil.h"

HLTTauValidation::HLTTauValidation(const edm::ParameterSet& ps) : 
  triggerEventObject_(ps.getUntrackedParameter<edm::InputTag>("triggerEventObject")),
  refCollection_(ps.getUntrackedParameter<edm::InputTag>("refTauCollection")),
  refLeptonCollection_(ps.getUntrackedParameter<edm::InputTag>("refLeptonCollection")),
  triggerTag_(ps.getUntrackedParameter<std::string>("DQMFolder","DoubleTau")),
  l1seedFilter_(ps.getUntrackedParameter<edm::InputTag>("L1SeedFilter")),
  l2filter_(ps.getUntrackedParameter<edm::InputTag>("L2EcalIsolFilter")),
  l25filter_(ps.getUntrackedParameter<edm::InputTag>("L25PixelIsolFilter")),
  l3filter_(ps.getUntrackedParameter<edm::InputTag>("L3SiliconIsolFilter")),
  electronFilter_(ps.getUntrackedParameter<edm::InputTag>("ElectronFilter")),
  muonFilter_(ps.getUntrackedParameter<edm::InputTag>("MuonFilter")),
  nTriggeredTaus_(ps.getUntrackedParameter<unsigned>("NTriggeredTaus",2)),
  nTriggeredLeptons_(ps.getUntrackedParameter<unsigned>("NTriggeredLeptons",0)),
  doRefAnalysis_(ps.getUntrackedParameter<bool>("DoReferenceAnalysis",false)),
  outFile_(ps.getUntrackedParameter<std::string>("OutputFileName","")),
  logFile_(ps.getUntrackedParameter<std::string>("LogFileName","log.txt")),
  matchDeltaRL1_(ps.getUntrackedParameter<double>("MatchDeltaRL1",0.3)),
  matchDeltaRHLT_(ps.getUntrackedParameter<double>("MatchDeltaRHLT",0.15))
{
  //initialize 
  NRefEvents=0;
  NLeptonEvents = 0;
  NLeptonEvents_Matched=0;
  NL1Events=0;
  NL1Events_Matched=0;
  NL2Events=0;
  NL2Events_Matched=0;
  NL25Events=0;
  NL25Events_Matched=0;
  NL3Events=0;
  NL3Events_Matched=0;

  

  //Declare DQM Store
  DQMStore* store = &*edm::Service<DQMStore>();

  if(store)
    {
      //Create the histograms
      store->setCurrentFolder(triggerTag_);
      l1eteff = store->book1D("l1eteff","L1 Efficiency vs E_{t}",100,0,200);
      l2eteff = store->book1D("l2eteff","L2 Efficiency vs E_{t}",100,0,200);
      l25eteff = store->book1D("l25eteff","L25 Efficiency vs E_{t}",100,0,200);
      l3eteff = store->book1D("l3eteff","L3 Efficiency vs E_{t}",100,0,200);

      refEt = store->book1D("refEt","Reference E_{t} ",100,0,200);
      refEta = store->book1D("refEta","Reference #eta ",50,-2.5,2.5);

      l1etaeff = store->book1D("l1etaeff","L1 Efficiency vs #eta",50,-2.5,2.5);
      l2etaeff = store->book1D("l2etaeff","L2 Efficiency vs #eta",50,-2.5,2.5);
      l25etaeff = store->book1D("l25etaeff","L25 Efficiency vs #eta",50,-2.5,2.5);
      l3etaeff = store->book1D("l3etaeff","L3 Efficiency vs #eta",50,-2.5,2.5);

      accepted_events = store->book1D("acceptedEvents","Accepted Events per path",5,0,5);
      accepted_events_matched = store->book1D("acceptedEventsMatched","Accepted Events per path",6,0,6);
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
  accepted_events->setBinContent(1,NLeptonEvents);
  accepted_events->setBinContent(2,NL1Events);
  accepted_events->setBinContent(3,NL2Events);
  accepted_events->setBinContent(4,NL25Events);
  accepted_events->setBinContent(5,NL3Events);


  accepted_events_matched->setBinContent(1,NLeptonEvents_Matched);
  accepted_events_matched->setBinContent(2,NL1Events_Matched);
  accepted_events_matched->setBinContent(3,NL2Events_Matched);
  accepted_events_matched->setBinContent(4,NL25Events_Matched);
  accepted_events_matched->setBinContent(5,NL3Events_Matched);
  accepted_events_matched->setBinContent(6,NRefEvents);


  //Write DQM thing..
  if(outFile_.size()>0)
  if (&*edm::Service<DQMStore>()) edm::Service<DQMStore>()->save (outFile_);


  //Write Log File
  if(logFile_.size()>0)
    {

      FILE *fp;
      fp = fopen(logFile_.c_str(),"w");

      fprintf(fp,"GENERATING OUTPUT--------------------------------------->\n");
      fprintf(fp,"Reference:\n");
      fprintf(fp,"   -Number of GOOD Ref Events = %d\n",NRefEvents);
      fprintf(fp,"Trigger:\n");
      fprintf(fp,"   -Leptonic Trigger Accepted Events = %d   Accepted and Matched = %d\n",NLeptonEvents,NLeptonEvents_Matched);
      fprintf(fp,"   -L1 Accepted Events = %d  L1 Accepted and Matched = %d\n",NL1Events,NL1Events_Matched);
      fprintf(fp,"   -L2 Accepted Events = %d  L2 Accepted and Matched = %d\n",NL2Events,NL2Events_Matched);
      fprintf(fp,"   -L25 Accepted Events = %d  L25 Accepted and Matched = %d\n",NL25Events,NL25Events_Matched);
      fprintf(fp,"   -L3 Accepted Events = %d  L3 Accepted and Matched = %d\n",NL3Events,NL3Events_Matched);
      fprintf(fp,"HLT Acceptance with Ref to Previous Trigger(No Matching):\n");
      fprintf(fp,"   -L2 = %f +/- %f \n",calcEfficiency(NL2Events,NL1Events)[0],calcEfficiency(NL2Events,NL1Events)[1]);
      fprintf(fp,"   -L25 = %f +/- %f \n",calcEfficiency(NL25Events,NL2Events)[0],calcEfficiency(NL25Events,NL2Events)[1]);
      fprintf(fp,"   -L3 = %f +/- %f \n",calcEfficiency(NL3Events,NL25Events)[0],calcEfficiency(NL3Events,NL25Events)[1]);
      fprintf(fp,"HLT Efficiency with Ref to Previous Trigger(with Matching):\n");
      fprintf(fp,"   -L2 = %f +/- %f \n",calcEfficiency(NL2Events_Matched,NL1Events_Matched)[0],calcEfficiency(NL2Events_Matched,NL1Events_Matched)[1]);
      fprintf(fp,"   -L25 = %f +/- %f \n",calcEfficiency(NL25Events_Matched,NL2Events_Matched)[0],calcEfficiency(NL25Events_Matched,NL2Events_Matched)[1]);
      fprintf(fp,"   -L3 = %f +/- %f \n",calcEfficiency(NL3Events_Matched,NL25Events_Matched)[0],calcEfficiency(NL3Events_Matched,NL25Events_Matched)[1]);

      fprintf(fp,"HLT Efficiency with Ref to L1 Trigger(with Matching):\n");
      fprintf(fp,"   -L2 = %f +/- %f \n",calcEfficiency(NL2Events_Matched,NL1Events_Matched)[0],calcEfficiency(NL2Events_Matched,NL1Events_Matched)[1]);
      fprintf(fp,"   -L25 = %f +/- %f \n",calcEfficiency(NL25Events_Matched,NL1Events_Matched)[0],calcEfficiency(NL25Events_Matched,NL1Events_Matched)[1]);
      fprintf(fp,"   -L3 = %f +/- %f \n",calcEfficiency(NL3Events_Matched,NL1Events_Matched)[0],calcEfficiency(NL3Events_Matched,NL1Events_Matched)[1]);
      fprintf(fp,"HLT Efficiency with Ref to Matching Object):\n");
      fprintf(fp,"   -L1 = %f +/- %f \n",calcEfficiency(NL1Events_Matched,NRefEvents)[0],calcEfficiency(NL1Events_Matched,NRefEvents)[1]);
      fprintf(fp,"   -L2 = %f +/- %f \n",calcEfficiency(NL2Events_Matched,NRefEvents)[0],calcEfficiency(NL2Events_Matched,NRefEvents)[1]);
      fprintf(fp,"   -L25 = %f +/- %f \n",calcEfficiency(NL25Events_Matched,NRefEvents)[0],calcEfficiency(NL25Events_Matched,NRefEvents)[1]);
      fprintf(fp,"   -L3 = %f +/- %f \n",calcEfficiency(NL3Events_Matched,NRefEvents)[0],calcEfficiency(NL3Events_Matched,NRefEvents)[1]);
      fprintf(fp,"--------------------------------------------------------\n");
      fprintf(fp,"Note: The errors are binomial..");	 
      fclose(fp);
    }
 
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
  
  double RefEt = 0;
  double RefEta = 0;

  if(doRefAnalysis_)
    {
      bool tau_ok = true;
      bool lepton_ok = true;
      

      //Tau reference
      if(iEvent.getByLabel(refCollection_,refC))
      if(refC->size()<nTriggeredTaus_)
	{
	  tau_ok = false;
	}
      else
	{
	  //Find Lead jet Et
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
	 
	  RefEt = et;
	  RefEta = eta;

	}
  
      //lepton reference
 
      if(iEvent.getByLabel(refLeptonCollection_,refCL))
	if(refCL->size()<nTriggeredLeptons_)
	{
	  lepton_ok = false;
	}
      
    
      
      if(lepton_ok&&tau_ok)
	{
	  NRefEvents++;
	  refEt->Fill(RefEt);
	  refEta->Fill(RefEta);
	}
    }


  //get The triggerEvent


  Handle<TriggerEventWithRefs> trigEv;
  iEvent.getByLabel(triggerEventObject_,trigEv);

  if (trigEv.isValid()) 
    {

      //LEPTONS 
      unsigned Leptons = 0;
      unsigned Leptons_Matched = 0;

      unsigned L1Leptons = 0;
      unsigned L1Leptons_Matched = 0;




      //L1 electrons
      size_t L1ELID=0;
      L1ELID =trigEv->filterIndex(l1seedFilter_);
      if(L1ELID!=trigEv->size())
	{
	  VRl1em electrons;
	  trigEv->getObjects(L1ELID,trigger::TriggerL1IsoEG,electrons);
	  L1Leptons+= electrons.size();

	  if(doRefAnalysis_)
	  for(size_t i = 0;i<electrons.size();++i)
	    {
	      if(match((*electrons[i]).p4(),*refCL,matchDeltaRL1_))
		L1Leptons_Matched++;
	    } 


	}

      //L1 muons
      size_t L1MUID=0;
      L1MUID =trigEv->filterIndex(l1seedFilter_);
      if(L1MUID!=trigEv->size())
	{
	  VRl1muon muons;
	  trigEv->getObjects(L1MUID,trigger::TriggerL1Mu,muons);
	  L1Leptons+= muons.size();

	  if(doRefAnalysis_)
	  for(size_t i = 0;i<muons.size();++i)
	    {
	      if(match((*muons[i]).p4(),*refCL,matchDeltaRL1_))
		L1Leptons_Matched++;
	    } 


	}



      //hlt electrons
      size_t ELID=0;
      ELID =trigEv->filterIndex(electronFilter_);
      if(ELID!=trigEv->size())
	{
	  VRelectron electrons;
	  trigEv->getObjects(ELID,trigger::TriggerElectron,electrons);
	  Leptons+= electrons.size();

	  if(doRefAnalysis_)
	  for(size_t i = 0;i<electrons.size();++i)
	    {
	      if(match((*electrons[i]).p4(),*refCL,matchDeltaRHLT_))
		Leptons_Matched++;
	    } 


	}

      //hlt muons
      size_t MUID=0;
      MUID =trigEv->filterIndex(muonFilter_);
      if(MUID!=trigEv->size())
	{
	  VRmuon muons;
	  trigEv->getObjects(MUID,trigger::TriggerMuon,muons);
	  Leptons+= muons.size();

	  if(doRefAnalysis_)
	  for(size_t i = 0;i<muons.size();++i)
	    {
	      //	      const reco::Jet jet = dynamic_cast<reco::Jet>(*(muons[i])); 
	      if(match((*muons[i]).p4(),*refCL,matchDeltaRHLT_))
		Leptons_Matched++;
	    } 


	}

      
      if(Leptons>=nTriggeredLeptons_&&nTriggeredLeptons_>0)
	NLeptonEvents++;

      if(Leptons_Matched>=nTriggeredLeptons_&&nTriggeredLeptons_>0)
	NLeptonEvents_Matched++;



    //L1Analysis Seed
      size_t L1ID=0;
      L1ID =trigEv->filterIndex(l1seedFilter_);
    
      if(L1ID!=trigEv->size())
	{
	  //Get L1Objects
	  VRl1jet L1Taus;
	  trigEv->getObjects(L1ID,trigger::TriggerL1TauJet,L1Taus);
	  //Check if the number of L1 Taus is OK
	  if(L1Taus.size()>=nTriggeredTaus_&& L1Leptons>=nTriggeredLeptons_)
	  {
	    NL1Events++;
	    	   
	  }


	  //Loop on L1 Taus  
	  unsigned jets_matched=0;	  
	  for(size_t i = 0;i<L1Taus.size();++i)
	  {
	    
	    if(doRefAnalysis_)
	      if(match((*L1Taus[i]).p4(),*refC,matchDeltaRL1_))
		{
		    jets_matched++;
		   
		}
	  }  
	  if(jets_matched>=nTriggeredTaus_&&L1Leptons_Matched>=nTriggeredLeptons_)
	    {
	      NL1Events_Matched++;
	      l1eteff->Fill(RefEt);
	      l1etaeff->Fill(RefEta);

	    
	    }

	}


    //L2Analysis Seed
      size_t L2ID=0;
      L2ID =trigEv->filterIndex(l2filter_);
   
   
      if(L2ID!=trigEv->size())
	{
	  //Get L2Objects
	  VRjet L2Taus;
	  trigEv->getObjects(L2ID,trigger::TriggerTau,L2Taus);
	  if(L2Taus.size()>=nTriggeredTaus_&&Leptons>=nTriggeredLeptons_)
	    {
	      NL2Events++;
	   
	    }
	  //Loop on L2 Taus
  	  unsigned jets_matched=0;	  
	  for(size_t i = 0;i<L2Taus.size();++i)
	  {
	   

	    if(doRefAnalysis_)
	      if(match((*L2Taus[i]).p4(),*refC,matchDeltaRHLT_))
		{
		  jets_matched++;
		}
	     
	  }
		  
	  if(jets_matched>=nTriggeredTaus_&&Leptons_Matched>=nTriggeredLeptons_)
	    {
	      NL2Events_Matched++;
    	      l2eteff->Fill(RefEt);
	      l2etaeff->Fill(RefEta);
	    }



	}

      //L25Analysis Seed
      size_t L25ID=0;
      L25ID =trigEv->filterIndex(l25filter_);

      if(L25ID!=trigEv->size())
	{
	  //Get L25Objects
	  VRjet L25Taus;
	  trigEv->getObjects(L25ID,trigger::TriggerTau,L25Taus);
	  if(L25Taus.size()>=nTriggeredTaus_&&Leptons>=nTriggeredLeptons_)
	    {
	      NL25Events++;

	    }
	  //Loop on L25 Taus
  	  unsigned jets_matched=0;	  
	  for(size_t i = 0;i<L25Taus.size();++i)
	  {

	    if(doRefAnalysis_)
	      if(match((*L25Taus[i]).p4(),*refC,matchDeltaRHLT_))
		{
		    jets_matched++;


		}
	  }  
	  if(jets_matched>=nTriggeredTaus_&&Leptons_Matched>=nTriggeredLeptons_)
	    {
	      NL25Events_Matched++;
	      l25eteff->Fill(RefEt);
	      l25etaeff->Fill(RefEta);
	    }


	}

      //L3Analysis Seed
      size_t L3ID=0;
      L3ID =trigEv->filterIndex(l3filter_);

      if(L3ID!=trigEv->size())
	{
	  //Get L3Objects
	  VRjet L3Taus;
	  trigEv->getObjects(L3ID,trigger::TriggerTau,L3Taus);
	  if(L3Taus.size()>=nTriggeredTaus_&&Leptons>=nTriggeredLeptons_)
	    {
	      NL3Events++;
	    
	    }
	  //Loop on L3 Taus
  	  unsigned jets_matched=0;	  
	  for(size_t i = 0;i<L3Taus.size();++i)
	  {
	    
	    if(doRefAnalysis_)
	      if(match((*L3Taus[i]).p4(),*refC,matchDeltaRHLT_))
		{
		    jets_matched++;


		}
	  }
		  
	  if(jets_matched>=nTriggeredTaus_&&Leptons_Matched>=nTriggeredLeptons_)
	    {
	      NL3Events_Matched++;
	      l3eteff->Fill(RefEt);
	      l3etaeff->Fill(RefEta);

	    }
	}
  } 
  else 
  {
    cout << "Handle invalid! Check InputTag provided." << endl;
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

std::vector<double>
HLTTauValidation::calcEfficiency(int num,int denom)
{
  std::vector<double> a;
  if(denom==0)
    {
      a.push_back(0.);
      a.push_back(0.);
    }
  else
    {    
      a.push_back(((double)num)/((double)denom));
      a.push_back(sqrt(a[0]*(1-a[0])/((double)denom)));
    }
  return a;
}

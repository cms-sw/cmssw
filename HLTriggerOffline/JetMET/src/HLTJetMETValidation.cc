#include "HLTriggerOffline/JetMET/interface/HLTJetMETValidation.h"
#include "Math/GenVector/VectorUtil.h"

HLTJetMETValidation::HLTJetMETValidation(const edm::ParameterSet& ps) : 
//JoCa  triggerEventObject_(ps.getUntrackedParameter<edm::InputTag>("triggerEventObject")),
//JoCa  refCollection_(ps.getUntrackedParameter<edm::InputTag>("refTauCollection")),
//JoCa  refLeptonCollection_(ps.getUntrackedParameter<edm::InputTag>("refLeptonCollection")),
//JoCa  triggerTag_(ps.getUntrackedParameter<std::string>("DQMFolder","DoubleTau")),
//JoCa  l1seedFilter_(ps.getUntrackedParameter<edm::InputTag>("L1SeedFilter")),
//JoCa  l2filter_(ps.getUntrackedParameter<edm::InputTag>("L2EcalIsolFilter")),
//JoCa  l25filter_(ps.getUntrackedParameter<edm::InputTag>("L25PixelIsolFilter")),
//JoCa  l3filter_(ps.getUntrackedParameter<edm::InputTag>("L3SiliconIsolFilter")),
//JoCa  electronFilter_(ps.getUntrackedParameter<edm::InputTag>("ElectronFilter")),
//JoCa  muonFilter_(ps.getUntrackedParameter<edm::InputTag>("MuonFilter")),
//JoCa  nTriggeredTaus_(ps.getUntrackedParameter<unsigned>("NTriggeredTaus",2)),
//JoCa  nTriggeredLeptons_(ps.getUntrackedParameter<unsigned>("NTriggeredLeptons",0)),
//JoCa  doRefAnalysis_(ps.getUntrackedParameter<bool>("DoReferenceAnalysis",false)),
//JoCa  outFile_(ps.getUntrackedParameter<std::string>("OutputFileName","")),
//JoCa  logFile_(ps.getUntrackedParameter<std::string>("LogFileName","log.txt")),
//JoCa  matchDeltaRL1_(ps.getUntrackedParameter<double>("MatchDeltaRL1",0.3)),
//JoCa  matchDeltaRHLT_(ps.getUntrackedParameter<double>("MatchDeltaRHLT",0.15))
  triggerEventObject_(ps.getUntrackedParameter<edm::InputTag>("triggerEventObject")),
  triggerTag_(ps.getUntrackedParameter<std::string>("DQMFolder","SingleJet")),
  _reffilter(ps.getUntrackedParameter<edm::InputTag>("RefFilter")),
  _probefilter(ps.getUntrackedParameter<edm::InputTag>("ProbeFilter")),
  outFile_(ps.getUntrackedParameter<std::string>("OutputFileName",""))
{
//JoCa  //initialize 
//JoCa  NRefEvents=0;
//JoCa  NLeptonEvents = 0;
//JoCa  NLeptonEvents_Matched=0;
//JoCa  NL1Events=0;
//JoCa  NL1Events_Matched=0;
//JoCa  NL2Events=0;
//JoCa  NL2Events_Matched=0;
//JoCa  NL25Events=0;
//JoCa  NL25Events_Matched=0;
//JoCa  NL3Events=0;
//JoCa  NL3Events_Matched=0;
  NTag = 0;
  NProbe = 0;
  
//JoCa
  //Declare DQM Store
  DQMStore* store = &*edm::Service<DQMStore>();

  if(store)
    {
      //Create the histograms
      store->setCurrentFolder(triggerTag_);
      test_histo = store->book1D("test_histo","Test Histogram",100,0,100);
      _meSingleJetPt= store->book1D("_meSingleJetPt","HLT Single Jet Pt",100,0,500);
      _meRefPt= store->book1D("_meRefPt","HLT Reference Pt",100,0,500);
      _meProbePt= store->book1D("_meProbePt","HLT Probe Pt",100,0,500);
//JoCa      l1eteff = store->book1D("l1eteff","L1 Efficiency vs E_{t}",100,0,200);
//JoCa      l2eteff = store->book1D("l2eteff","L2 Efficiency vs E_{t}",100,0,200);
//JoCa      l25eteff = store->book1D("l25eteff","L25 Efficiency vs E_{t}",100,0,200);
//JoCa      l3eteff = store->book1D("l3eteff","L3 Efficiency vs E_{t}",100,0,200);
//JoCa
//JoCa      refEt = store->book1D("refEt","Reference E_{t} ",100,0,200);
//JoCa      refEta = store->book1D("refEta","Reference #eta ",50,-2.5,2.5);
//JoCa
//JoCa      l1etaeff = store->book1D("l1etaeff","L1 Efficiency vs #eta",50,-2.5,2.5);
//JoCa      l2etaeff = store->book1D("l2etaeff","L2 Efficiency vs #eta",50,-2.5,2.5);
//JoCa      l25etaeff = store->book1D("l25etaeff","L25 Efficiency vs #eta",50,-2.5,2.5);
//JoCa      l3etaeff = store->book1D("l3etaeff","L3 Efficiency vs #eta",50,-2.5,2.5);


    }

  printf("JoCa: initializing\n");
}

HLTJetMETValidation::~HLTJetMETValidation()
{
}

//
// member functions
//

void
HLTJetMETValidation::endJob()
{

  //Write DQM thing..
  if(outFile_.size()>0)
  if (&*edm::Service<DQMStore>()) edm::Service<DQMStore>()->save (outFile_);

  printf("\n\n");
  printf("NTag = %i\n",NTag);
  printf("NProbe = %i\n",NProbe);


//JoCa  //Write Log File
//JoCa  FILE *fp;
//JoCa  fp = fopen(logFile_.c_str(),"w");
//JoCa
//JoCa  fprintf(fp,"GENERATING OUTPUT--------------------------------------->\n");
//JoCa  fprintf(fp,"Reference:\n");
//JoCa  fprintf(fp,"   -Number of GOOD Ref Events = %d\n",NRefEvents);
//JoCa  fprintf(fp,"Trigger:\n");
//JoCa  fprintf(fp,"   -Leptonic Trigger Accepted Events = %d   Accepted and Matched = %d\n",NLeptonEvents,NLeptonEvents_Matched);
//JoCa  fprintf(fp,"   -L1 Accepted Events = %d  L1 Accepted and Matched = %d\n",NL1Events,NL1Events_Matched);
//JoCa  fprintf(fp,"   -L2 Accepted Events = %d  L2 Accepted and Matched = %d\n",NL2Events,NL2Events_Matched);
//JoCa  fprintf(fp,"   -L25 Accepted Events = %d  L25 Accepted and Matched = %d\n",NL25Events,NL25Events_Matched);
//JoCa  fprintf(fp,"   -L3 Accepted Events = %d  L3 Accepted and Matched = %d\n",NL3Events,NL3Events_Matched);
//JoCa  fprintf(fp,"HLT Acceptance with Ref to Previous Trigger(No Matching):\n");
//JoCa  fprintf(fp,"   -L2 = %f +/- %f \n",calcEfficiency(NL2Events,NL1Events)[0],calcEfficiency(NL2Events,NL1Events)[1]);
//JoCa  fprintf(fp,"   -L25 = %f +/- %f \n",calcEfficiency(NL25Events,NL2Events)[0],calcEfficiency(NL25Events,NL2Events)[1]);
//JoCa  fprintf(fp,"   -L3 = %f +/- %f \n",calcEfficiency(NL3Events,NL25Events)[0],calcEfficiency(NL3Events,NL25Events)[1]);
//JoCa  fprintf(fp,"HLT Efficiency with Ref to Previous Trigger(with Matching):\n");
//JoCa  fprintf(fp,"   -L2 = %f +/- %f \n",calcEfficiency(NL2Events_Matched,NL1Events_Matched)[0],calcEfficiency(NL2Events_Matched,NL1Events_Matched)[1]);
//JoCa  fprintf(fp,"   -L25 = %f +/- %f \n",calcEfficiency(NL25Events_Matched,NL2Events_Matched)[0],calcEfficiency(NL25Events_Matched,NL2Events_Matched)[1]);
//JoCa  fprintf(fp,"   -L3 = %f +/- %f \n",calcEfficiency(NL3Events_Matched,NL25Events_Matched)[0],calcEfficiency(NL3Events_Matched,NL25Events_Matched)[1]);
//JoCa
//JoCa  fprintf(fp,"HLT Efficiency with Ref to L1 Trigger(with Matching):\n");
//JoCa  fprintf(fp,"   -L2 = %f +/- %f \n",calcEfficiency(NL2Events_Matched,NL1Events_Matched)[0],calcEfficiency(NL2Events_Matched,NL1Events_Matched)[1]);
//JoCa  fprintf(fp,"   -L25 = %f +/- %f \n",calcEfficiency(NL25Events_Matched,NL1Events_Matched)[0],calcEfficiency(NL25Events_Matched,NL1Events_Matched)[1]);
//JoCa  fprintf(fp,"   -L3 = %f +/- %f \n",calcEfficiency(NL3Events_Matched,NL1Events_Matched)[0],calcEfficiency(NL3Events_Matched,NL1Events_Matched)[1]);
//JoCa  fprintf(fp,"HLT Efficiency with Ref to Matching Object):\n");
//JoCa  fprintf(fp,"   -L1 = %f +/- %f \n",calcEfficiency(NL1Events_Matched,NRefEvents)[0],calcEfficiency(NL1Events_Matched,NRefEvents)[1]);
//JoCa  fprintf(fp,"   -L2 = %f +/- %f \n",calcEfficiency(NL2Events_Matched,NRefEvents)[0],calcEfficiency(NL2Events_Matched,NRefEvents)[1]);
//JoCa  fprintf(fp,"   -L25 = %f +/- %f \n",calcEfficiency(NL25Events_Matched,NRefEvents)[0],calcEfficiency(NL25Events_Matched,NRefEvents)[1]);
//JoCa  fprintf(fp,"   -L3 = %f +/- %f \n",calcEfficiency(NL3Events_Matched,NRefEvents)[0],calcEfficiency(NL3Events_Matched,NRefEvents)[1]);
//JoCa  fprintf(fp,"--------------------------------------------------------\n");
//JoCa  fprintf(fp,"Note: The errors are binomial..");	 
//JoCa  fclose(fp);
//JoCa 
}


void
HLTJetMETValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace l1extra;
  using namespace trigger;

  printf("JoCa: for each event\n");

  test_histo->Fill(50.);

//JoCa
//JoCa
//JoCa
//JoCa
//JoCa  //Look at the reference collection (i.e MC)
//JoCa  Handle<LVColl> refC;
//JoCa  Handle<LVColl> refCL;
//JoCa  
//JoCa  double RefEt = 0;
//JoCa  double RefEta = 0;
//JoCa
//JoCa  if(doRefAnalysis_)
//JoCa    {
//JoCa      bool tau_ok = true;
//JoCa      bool lepton_ok = true;
//JoCa      
//JoCa
//JoCa      //Tau reference
//JoCa      if(iEvent.getByLabel(refCollection_,refC))
//JoCa      if(refC->size()<nTriggeredTaus_)
//JoCa	{
//JoCa	  tau_ok = false;
//JoCa	}
//JoCa      else
//JoCa	{
//JoCa	  //Find Lead jet Et
//JoCa	  double et=0.;
//JoCa	  double eta=0.;
//JoCa	  
//JoCa	  for(size_t j = 0;j<refC->size();++j)
//JoCa	    {
//JoCa	      if((*refC)[j].Et()>et)
//JoCa		{
//JoCa		  et=(*refC)[j].Et();
//JoCa		  eta = (*refC)[j].Eta();
//JoCa		}
//JoCa		
//JoCa	    }
//JoCa	 
//JoCa	  RefEt = et;
//JoCa	  RefEta = eta;
//JoCa
//JoCa	}
//JoCa  
//JoCa      //lepton reference
//JoCa 
//JoCa      if(iEvent.getByLabel(refLeptonCollection_,refCL))
//JoCa	if(refCL->size()<nTriggeredLeptons_)
//JoCa	{
//JoCa	  lepton_ok = false;
//JoCa	}
//JoCa      
//JoCa    
//JoCa      
//JoCa      if(lepton_ok&&tau_ok)
//JoCa	{
//JoCa	  NRefEvents++;
//JoCa	  refEt->Fill(RefEt);
//JoCa	  refEta->Fill(RefEta);
//JoCa	}
//JoCa    }
//JoCa
//JoCa
  //get The triggerEvent


  Handle<TriggerEventWithRefs> trigEv;
  iEvent.getByLabel(triggerEventObject_,trigEv);

  if (trigEv.isValid()) {
    printf("JoCa   found trigger information\n");
  } else {
    printf("JoCa   ERROR: no trigger information\n");
  }

  if (trigEv.isValid()) {
    int trigsize = trigEv->size();
    printf("JoCa  Number of filters = %i\n",trigsize);
    for(int i=0; i<trigsize; i++){
      cout << trigEv->filterTag(i) << endl;
    }
  }

  // just a test: check if hlt1jet30 has been fired
  size_t FLT_HLTREF = 0;
  FLT_HLTREF = trigEv->filterIndex(_reffilter);
  cout << "   FLT_HLTREF = " << FLT_HLTREF << endl;
  size_t FLT_HLTPROBE = 0;
  FLT_HLTPROBE = trigEv->filterIndex(_probefilter);
  cout << "   FLT_HLTPROBE = " << FLT_HLTPROBE << endl;
  
  // first make sure that the reference trigger was fired
  if (FLT_HLTREF != trigEv->size()){
    NTag++;
    // then count how often the probe trigger was fired
    if (FLT_HLTPROBE != trigEv->size()){
      NProbe++;


      // now get the object's four vector information
      size_t HLTProbeJetID = 0;
      HLTProbeJetID =trigEv->filterIndex(_probefilter);

      // VR* types defined in CMSSW/DataFormats/HLTReco/interface/TriggerRefsCollections.h
      VRjet probejets;
      // look in CMSSW/DataFormats/HLTReco/interface/TriggerTypeDefs.h
      // for definition of trigger:: formats
      trigEv->getObjects(HLTProbeJetID,trigger::TriggerJet,probejets);
      cout << "     probejets.size = " << probejets.size() << endl;
      for (int i=0; i < probejets.size(); i++){
	cout << "  i = " << i << "    pt = " << (*probejets[i]).pt() << endl;
      } // for (int i=0; i < probejets.size(); i++){
    } // if (FLT_HLTPROBE != trigEv->size()){
  } // if (FLT_HLTREF != trigEv->size()){

//JoCa
//JoCa      //LEPTONS 
//JoCa      unsigned Leptons = 0;
//JoCa      unsigned Leptons_Matched = 0;
//JoCa
//JoCa      unsigned L1Leptons = 0;
//JoCa      unsigned L1Leptons_Matched = 0;
//JoCa
//JoCa
//JoCa
//JoCa
//JoCa      //L1 electrons
//JoCa      size_t L1ELID=0;
//JoCa      L1ELID =trigEv->filterIndex(l1seedFilter_);
//JoCa      if(L1ELID!=trigEv->size())
//JoCa	{
//JoCa	  VRl1em electrons;
//JoCa	  trigEv->getObjects(L1ELID,trigger::TriggerL1IsoEG,electrons);
//JoCa	  L1Leptons+= electrons.size();
//JoCa
//JoCa	  if(doRefAnalysis_)
//JoCa	  for(size_t i = 0;i<electrons.size();++i)
//JoCa	    {
//JoCa	      if(match((*electrons[i]).p4(),*refCL,matchDeltaRL1_))
//JoCa		L1Leptons_Matched++;
//JoCa	    } 
//JoCa
//JoCa
//JoCa	}
//JoCa
//JoCa      //L1 muons
//JoCa      size_t L1MUID=0;
//JoCa      L1MUID =trigEv->filterIndex(l1seedFilter_);
//JoCa      if(L1MUID!=trigEv->size())
//JoCa	{
//JoCa	  VRl1muon muons;
//JoCa	  trigEv->getObjects(L1MUID,trigger::TriggerL1Mu,muons);
//JoCa	  L1Leptons+= muons.size();
//JoCa
//JoCa	  if(doRefAnalysis_)
//JoCa	  for(size_t i = 0;i<muons.size();++i)
//JoCa	    {
//JoCa	      if(match((*muons[i]).p4(),*refCL,matchDeltaRL1_))
//JoCa		L1Leptons_Matched++;
//JoCa	    } 
//JoCa
//JoCa
//JoCa	}
//JoCa
//JoCa
//JoCa
//JoCa      //hlt electrons
//JoCa      size_t ELID=0;
//JoCa      ELID =trigEv->filterIndex(electronFilter_);
//JoCa      if(ELID!=trigEv->size())
//JoCa	{
//JoCa	  VRelectron electrons;
//JoCa	  trigEv->getObjects(ELID,trigger::TriggerElectron,electrons);
//JoCa	  Leptons+= electrons.size();
//JoCa
//JoCa	  if(doRefAnalysis_)
//JoCa	  for(size_t i = 0;i<electrons.size();++i)
//JoCa	    {
//JoCa	      if(match((*electrons[i]).p4(),*refCL,matchDeltaRHLT_))
//JoCa		Leptons_Matched++;
//JoCa	    } 
//JoCa
//JoCa
//JoCa	}
//JoCa
//JoCa      //hlt muons
//JoCa      size_t MUID=0;
//JoCa      MUID =trigEv->filterIndex(muonFilter_);
//JoCa      if(MUID!=trigEv->size())
//JoCa	{
//JoCa	  VRmuon muons;
//JoCa	  trigEv->getObjects(MUID,trigger::TriggerMuon,muons);
//JoCa	  Leptons+= muons.size();
//JoCa
//JoCa	  if(doRefAnalysis_)
//JoCa	  for(size_t i = 0;i<muons.size();++i)
//JoCa	    {
//JoCa	      //	      const reco::Jet jet = dynamic_cast<reco::Jet>(*(muons[i])); 
//JoCa	      if(match((*muons[i]).p4(),*refCL,matchDeltaRHLT_))
//JoCa		Leptons_Matched++;
//JoCa	    } 
//JoCa
//JoCa
//JoCa	}
//JoCa
//JoCa      
//JoCa      if(Leptons>=nTriggeredLeptons_&&nTriggeredLeptons_>0)
//JoCa	NLeptonEvents++;
//JoCa
//JoCa      if(Leptons_Matched>=nTriggeredLeptons_&&nTriggeredLeptons_>0)
//JoCa	NLeptonEvents_Matched++;
//JoCa
//JoCa
//JoCa
//JoCa    //L1Analysis Seed
//JoCa      size_t L1ID=0;
//JoCa      L1ID =trigEv->filterIndex(l1seedFilter_);
//JoCa      printf("L1id = %d\n",L1ID);
//JoCa      if(L1ID!=trigEv->size())
//JoCa	{
//JoCa	  //Get L1Objects
//JoCa	  VRl1jet L1Taus;
//JoCa	  trigEv->getObjects(L1ID,trigger::TriggerL1TauJet,L1Taus);
//JoCa	  //Check if the number of L1 Taus is OK
//JoCa	  if(L1Taus.size()>=nTriggeredTaus_&& L1Leptons>=nTriggeredLeptons_)
//JoCa	  {
//JoCa	    NL1Events++;
//JoCa	    	   
//JoCa	  }
//JoCa
//JoCa
//JoCa	  //Loop on L1 Taus  
//JoCa	  unsigned jets_matched=0;	  
//JoCa	  for(size_t i = 0;i<L1Taus.size();++i)
//JoCa	  {
//JoCa	    
//JoCa	    if(doRefAnalysis_)
//JoCa	      if(match((*L1Taus[i]).p4(),*refC,matchDeltaRL1_))
//JoCa		{
//JoCa		    jets_matched++;
//JoCa		   
//JoCa		}
//JoCa	  }  
//JoCa	  if(jets_matched>=nTriggeredTaus_&&L1Leptons_Matched>=nTriggeredLeptons_)
//JoCa	    {
//JoCa	      NL1Events_Matched++;
//JoCa	      l1eteff->Fill(RefEt);
//JoCa	      l1etaeff->Fill(RefEta);
//JoCa
//JoCa	    
//JoCa	    }
//JoCa
//JoCa	}
//JoCa
//JoCa
//JoCa    //L2Analysis Seed
//JoCa      size_t L2ID=0;
//JoCa      L2ID =trigEv->filterIndex(l2filter_);
//JoCa      printf("L2id = %d\n",L2ID);
//JoCa   
//JoCa      if(L2ID!=trigEv->size())
//JoCa	{
//JoCa	  //Get L2Objects
//JoCa	  VRjet L2Taus;
//JoCa	  trigEv->getObjects(L2ID,trigger::TriggerTau,L2Taus);
//JoCa	  if(L2Taus.size()>=nTriggeredTaus_&&Leptons>=nTriggeredLeptons_)
//JoCa	    {
//JoCa	      NL2Events++;
//JoCa	   
//JoCa	    }
//JoCa	  //Loop on L2 Taus
//JoCa  	  unsigned jets_matched=0;	  
//JoCa	  for(size_t i = 0;i<L2Taus.size();++i)
//JoCa	  {
//JoCa	   
//JoCa
//JoCa	    if(doRefAnalysis_)
//JoCa	      if(match((*L2Taus[i]).p4(),*refC,matchDeltaRHLT_))
//JoCa		{
//JoCa		  jets_matched++;
//JoCa		}
//JoCa	     
//JoCa	  }
//JoCa		  
//JoCa	  if(jets_matched>=nTriggeredTaus_&&Leptons_Matched>=nTriggeredLeptons_)
//JoCa	    {
//JoCa	      NL2Events_Matched++;
//JoCa    	      l2eteff->Fill(RefEt);
//JoCa	      l2etaeff->Fill(RefEta);
//JoCa	    }
//JoCa
//JoCa
//JoCa
//JoCa	}
//JoCa
//JoCa      //L25Analysis Seed
//JoCa      size_t L25ID=0;
//JoCa      L25ID =trigEv->filterIndex(l25filter_);
//JoCa
//JoCa      if(L25ID!=trigEv->size())
//JoCa	{
//JoCa	  //Get L25Objects
//JoCa	  VRjet L25Taus;
//JoCa	  trigEv->getObjects(L25ID,trigger::TriggerTau,L25Taus);
//JoCa	  if(L25Taus.size()>=nTriggeredTaus_&&Leptons>=nTriggeredLeptons_)
//JoCa	    {
//JoCa	      NL25Events++;
//JoCa
//JoCa	    }
//JoCa	  //Loop on L25 Taus
//JoCa  	  unsigned jets_matched=0;	  
//JoCa	  for(size_t i = 0;i<L25Taus.size();++i)
//JoCa	  {
//JoCa
//JoCa	    if(doRefAnalysis_)
//JoCa	      if(match((*L25Taus[i]).p4(),*refC,matchDeltaRHLT_))
//JoCa		{
//JoCa		    jets_matched++;
//JoCa
//JoCa
//JoCa		}
//JoCa	  }  
//JoCa	  if(jets_matched>=nTriggeredTaus_&&Leptons_Matched>=nTriggeredLeptons_)
//JoCa	    {
//JoCa	      NL25Events_Matched++;
//JoCa	      l25eteff->Fill(RefEt);
//JoCa	      l25etaeff->Fill(RefEta);
//JoCa	    }
//JoCa
//JoCa
//JoCa	}
//JoCa
//JoCa      //L3Analysis Seed
//JoCa      size_t L3ID=0;
//JoCa      L3ID =trigEv->filterIndex(l3filter_);
//JoCa
//JoCa      if(L3ID!=trigEv->size())
//JoCa	{
//JoCa	  //Get L3Objects
//JoCa	  VRjet L3Taus;
//JoCa	  trigEv->getObjects(L3ID,trigger::TriggerTau,L3Taus);
//JoCa	  if(L3Taus.size()>=nTriggeredTaus_&&Leptons>=nTriggeredLeptons_)
//JoCa	    {
//JoCa	      NL3Events++;
//JoCa	    
//JoCa	    }
//JoCa	  //Loop on L3 Taus
//JoCa  	  unsigned jets_matched=0;	  
//JoCa	  for(size_t i = 0;i<L3Taus.size();++i)
//JoCa	  {
//JoCa	    
//JoCa	    if(doRefAnalysis_)
//JoCa	      if(match((*L3Taus[i]).p4(),*refC,matchDeltaRHLT_))
//JoCa		{
//JoCa		    jets_matched++;
//JoCa
//JoCa
//JoCa		}
//JoCa	  }
//JoCa		  
//JoCa	  if(jets_matched>=nTriggeredTaus_&&Leptons_Matched>=nTriggeredLeptons_)
//JoCa	    {
//JoCa	      NL3Events_Matched++;
//JoCa	      l3eteff->Fill(RefEt);
//JoCa	      l3etaeff->Fill(RefEta);
//JoCa
//JoCa	    }
//JoCa	}
//JoCa  } 
//JoCa  else 
//JoCa  {
//JoCa    cout << "Handle invalid! Check InputTag provided." << endl;
//JoCa  }
//JoCa     
}




//JoCabool 
//JoCaHLTJetMETValidation::match(const LV& jet,const LVColl& McInfo,double dr)
//JoCa{
//JoCa 
//JoCa  bool matched=false;
//JoCa
//JoCa  if(McInfo.size()>0)
//JoCa    for(std::vector<LV>::const_iterator it = McInfo.begin();it!=McInfo.end();++it)
//JoCa      {
//JoCa	double delta = ROOT::Math::VectorUtil::DeltaR(jet,*it);
//JoCa	if(delta<dr)
//JoCa	  {
//JoCa	    matched=true;
//JoCa	  }
//JoCa      }
//JoCa
//JoCa  return matched;
//JoCa}
//JoCa
//JoCastd::vector<double>
//JoCaHLTJetMETValidation::calcEfficiency(int num,int denom)
//JoCa{
//JoCa  std::vector<double> a;
//JoCa  if(denom==0)
//JoCa    {
//JoCa      a.push_back(0.);
//JoCa      a.push_back(0.);
//JoCa    }
//JoCa  else
//JoCa    {    
//JoCa      a.push_back(((double)num)/((double)denom));
//JoCa      a.push_back(sqrt(a[0]*(1-a[0])/((double)denom)));
//JoCa    }
//JoCa  return a;
//JoCa}

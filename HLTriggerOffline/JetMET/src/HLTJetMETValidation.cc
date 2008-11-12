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
//initialize 
  NRef = 0;
  NProbe = 0;
  
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

  // perform operations with monitorables
  //  _meEffPt->Add(_meProbePt);


  //Write DQM thing..
  if(outFile_.size()>0)
  if (&*edm::Service<DQMStore>()) edm::Service<DQMStore>()->save (outFile_);

  printf("\n\n");
  printf("NRef = %i\n",NRef);
  printf("NProbe = %i\n",NProbe);


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

  // get the reference and probe jets
  size_t FLT_HLTREF = 0;
  FLT_HLTREF = trigEv->filterIndex(_reffilter); // get the position for the reference filter
  cout << "   FLT_HLTREF = " << FLT_HLTREF << endl;
  size_t FLT_HLTPROBE = 0;
  FLT_HLTPROBE = trigEv->filterIndex(_probefilter); // get the position for the probe filter
  cout << "   FLT_HLTPROBE = " << FLT_HLTPROBE << endl;
  
  // first make sure that the reference trigger was fired
  if (FLT_HLTREF != trigEv->size()){
    NRef++;
    size_t HLTRefJetID = 0;
    HLTRefJetID =trigEv->filterIndex(_reffilter);
    // VR* types defined in CMSSW/DataFormats/HLTReco/interface/TriggerRefsCollections.h
    //    VRl1jet refjets;
    VRjet refjets;
    // look in CMSSW/DataFormats/HLTReco/interface/TriggerTypeDefs.h
    // for definition of trigger:: formats
    trigEv->getObjects(HLTRefJetID,trigger::TriggerJet,refjets);
    //    trigEv->getObjects(HLTRefJetID,trigger::TriggerL1CenJet,refjets);
    cout << "   refjets.size = " << refjets.size() << endl;
    if (refjets.size() > 0){
      // fill leading jet that fired the reference trigger
      _meRefPt->Fill((*refjets[0]).pt());
      } // if (refjets.size() > 0){

    // then count how often the probe trigger was fired
    if (refjets.size() > 0 && FLT_HLTPROBE != trigEv->size()){
      NProbe++;
      // now get the object's four vector information
      size_t HLTProbeJetID = 0;
      HLTProbeJetID =trigEv->filterIndex(_probefilter);
      VRjet probejets;
      trigEv->getObjects(HLTProbeJetID,trigger::TriggerJet,probejets);
      cout << "     probejets.size = " << probejets.size() << endl;
      for (int i=0; i < probejets.size(); i++){
	cout << "  i = " << i << "    pt = " << (*probejets[i]).pt() << endl;
      } // for (int i=0; i < probejets.size(); i++){

      if (probejets.size() > 0){
	// fill leading jet that fired the probe trigger
	_meProbePt->Fill((*probejets[0]).pt());
      } // if (probejets.size() > 0){
    } // if (FLT_HLTPROBE != trigEv->size()){
  } // if (FLT_HLTREF != trigEv->size()){

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

#include "HLTriggerOffline/JetMET/interface/HLTJetMETValidation.h"
#include "Math/GenVector/VectorUtil.h"

HLTJetMETValidation::HLTJetMETValidation(const edm::ParameterSet& ps) : 
//JC  triggerEventObject_(ps.getUntrackedParameter<edm::InputTag>("triggerEventObject")),
//JC  refCollection_(ps.getUntrackedParameter<edm::InputTag>("refTauCollection")),
//JC  refLeptonCollection_(ps.getUntrackedParameter<edm::InputTag>("refLeptonCollection")),
//JC  triggerTag_(ps.getUntrackedParameter<std::string>("DQMFolder","DoubleTau")),
//JC  l1seedFilter_(ps.getUntrackedParameter<edm::InputTag>("L1SeedFilter")),
//JC  l2filter_(ps.getUntrackedParameter<edm::InputTag>("L2EcalIsolFilter")),
//JC  l25filter_(ps.getUntrackedParameter<edm::InputTag>("L25PixelIsolFilter")),
//JC  l3filter_(ps.getUntrackedParameter<edm::InputTag>("L3SiliconIsolFilter")),
//JC  electronFilter_(ps.getUntrackedParameter<edm::InputTag>("ElectronFilter")),
//JC  muonFilter_(ps.getUntrackedParameter<edm::InputTag>("MuonFilter")),
//JC  nTriggeredTaus_(ps.getUntrackedParameter<unsigned>("NTriggeredTaus",2)),
//JC  nTriggeredLeptons_(ps.getUntrackedParameter<unsigned>("NTriggeredLeptons",0)),
//JC  doRefAnalysis_(ps.getUntrackedParameter<bool>("DoReferenceAnalysis",false)),
//JC  outFile_(ps.getUntrackedParameter<std::string>("OutputFileName","")),
//JC  logFile_(ps.getUntrackedParameter<std::string>("LogFileName","log.txt")),
//JC  matchDeltaRL1_(ps.getUntrackedParameter<double>("MatchDeltaRL1",0.3)),
//JC  matchDeltaRHLT_(ps.getUntrackedParameter<double>("MatchDeltaRHLT",0.15))
  triggerEventObject_(ps.getUntrackedParameter<edm::InputTag>("triggerEventObject")),
  CaloJetAlgorithm( ps.getUntrackedParameter<edm::InputTag>( "CaloJetAlgorithm" ) ),
  GenJetAlgorithm( ps.getUntrackedParameter<edm::InputTag>( "GenJetAlgorithm" ) ),
  CaloMETColl( ps.getUntrackedParameter<edm::InputTag>( "CaloMETCollection" ) ),
  GenMETColl( ps.getUntrackedParameter<edm::InputTag>( "GenMETCollection" ) ),
  HLTriggerResults( ps.getParameter<edm::InputTag>( "HLTriggerResults" ) ),
  triggerTag_(ps.getUntrackedParameter<std::string>("DQMFolder","SingleJet")),
  _reffilter(ps.getUntrackedParameter<edm::InputTag>("RefFilter")),
  _probefilter(ps.getUntrackedParameter<edm::InputTag>("ProbeFilter")),
  _HLTPath(ps.getUntrackedParameter<edm::InputTag>("HLTPath")),
  outFile_(ps.getUntrackedParameter<std::string>("OutputFileName","")),
  HLTinit_(false),
  //JL
  writeFile_(ps.getUntrackedParameter<bool>("WriteFile",false))
{
//initialize 
  NRef = 0;
  NProbe = 0;
  evtCnt=0;

  //Declare DQM Store
  DQMStore* store = &*edm::Service<DQMStore>();

  if(store)
    {
      //Create the histograms
      store->setCurrentFolder(triggerTag_);
      if (writeFile_) test_histo = store->book1D("test_histo","Test Histogram",100,0,100);
      _meRecoJetPt= store->book1D("_meRecoJetPt","Single Reconstructed Jet Pt",100,0,500);
      _meRecoJetPtTrg= store->book1D("_meRecoJetPtTrg","Single Reconstructed Jet Pt -- HLT Triggered",100,0,500);
      _meRecoJetPtRef= store->book1D("_meRecoJetPtRef","Single Reconstructed Jet Pt -- Ref trigger fired",100,0,500);
      _meRecoJetPtProbe= store->book1D("_meRecoJetPtProbe","Single Reconstructed Jet Pt -- Probe trigger fired",100,0,500);

      _meRecoJetEta= store->book1D("_meRecoJetEta","Single Reconstructed Jet Eta",100,-10,10);
      _meRecoJetEtaTrg= store->book1D("_meRecoJetEtaTrg","Single Reconstructed Jet Eta -- HLT Triggered",100,-10,10);
      _meRecoJetEtaRef= store->book1D("_meRecoJetEtaRef","Single Reconstructed Jet Eta -- Ref trigger fired",100,-10,10);
      _meRecoJetEtaProbe= store->book1D("_meRecoJetEtaProbe","Single Reconstructed Jet Eta -- Probe trigger fired",100,-10,10);

      _meRecoJetPhi= store->book1D("_meRecoJetPhi","Single Reconstructed Jet Phi",100,-4.,4.);
      _meRecoJetPhiTrg= store->book1D("_meRecoJetPhiTrg","Single Reconstructed Jet Phi -- HLT Triggered",100,-4.,4.);
      _meRecoJetPhiRef= store->book1D("_meRecoJetPhiRef","Single Reconstructed Jet Phi -- Ref trigger fired",100,-4.,4.);
      _meRecoJetPhiProbe= store->book1D("_meRecoJetPhiProbe","Single Reconstructed Jet Phi -- Probe trigger fired",100,-4.,4.);

      _meGenJetPt= store->book1D("_meGenJetPt","Single Generated Jet Pt",100,0,500);
      _meGenJetPtTrg= store->book1D("_meGenJetPtTrg","Single Generated Jet Pt -- HLT Triggered",100,0,500);
      _meGenJetPtRef= store->book1D("_meGenJetPtRef","Single Generated Jet Pt -- Ref trigger fired",100,0,500);
      _meGenJetPtProbe= store->book1D("_meGenJetPtProbe","Single Generated Jet Pt -- Probe trigger fired",100,0,500);

      _meGenJetEta= store->book1D("_meGenJetEta","Single Generated Jet Eta",100,-10,10);
      _meGenJetEtaTrg= store->book1D("_meGenJetEtaTrg","Single Generated Jet Eta -- HLT Triggered",100,-10,10);
      _meGenJetEtaRef= store->book1D("_meGenJetEtaRef","Single Generated Jet Eta -- Ref trigger fired",100,-10,10);
      _meGenJetEtaProbe= store->book1D("_meGenJetEtaProbe","Single Generated Jet Eta -- Probe trigger fired",100,-10,10);

      _meGenJetPhi= store->book1D("_meGenJetPhi","Single Generated Jet Phi",100,-4.,4.);
      _meGenJetPhiTrg= store->book1D("_meGenJetPhiTrg","Single Generated Jet Phi -- HLT Triggered",100,-4.,4.);
      _meGenJetPhiRef= store->book1D("_meGenJetPhiRef","Single Generated Jet Phi -- Ref trigger fired",100,-4.,4.);
      _meGenJetPhiProbe= store->book1D("_meGenJetPhiProbe","Single Generated Jet Phi -- Probe trigger fired",100,-4.,4.);

      _meRecoMET= store->book1D("_meRecoMET","Reconstructed Missing ET",100,0,500);
      _meRecoMETTrg= store->book1D("_meRecoMETTrg","Reconstructed Missing ET -- HLT Triggered",100,0,500);
      _meRecoMETRef= store->book1D("_meRecoMETRef","Reconstructed Missing ET -- Ref trigger fired",100,0,500);
      _meRecoMETProbe= store->book1D("_meRecoMETProbe","Reconstructed Missing ET -- Probe trigger fired",100,0,500);

      _meGenMET= store->book1D("_meGenMET","Generated Missing ET",100,0,500);
      _meGenMETTrg= store->book1D("_meGenMETTrg","Generated Missing ET -- HLT Triggered",100,0,500);
      _meGenMETRef= store->book1D("_meGenMETRef","Generated Missing ET -- Ref trigger fired",100,0,500);
      _meGenMETProbe= store->book1D("_meGenMETProbe","Generated Missing ET -- Probe trigger fired",100,0,500);

      _meRefPt= store->book1D("_meRefPt","HLT Reference Pt",100,0,500);
      _meProbePt= store->book1D("_meProbePt","HLT Probe Pt",100,0,500);

      _triggerResults = store->book1D( "_triggerResults", "HLT Results", 200, 0, 200 );

      //JL
      //_meTurnOnMET= store->book1D("_meTurnOnMET","Missing ET Turn-On",100,0,500);
      //_meTurnOnJetPt= store->book1D("_meTurnOnJetPt","Jet Pt Turn-On",100,0,500);
    }

  //if (writeFile_) printf("Initializing\n");
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
    //JL
    //if (&*edm::Service<DQMStore>()) edm::Service<DQMStore>()->save (outFile_);
    if (&*edm::Service<DQMStore>() && writeFile_) {
      edm::Service<DQMStore>()->save (outFile_);

      //printf("\n\n");
      //printf("NRef = %i\n",NRef);
      //printf("NProbe = %i\n",NProbe);
    }
  
}

//JL
void
HLTJetMETValidation::endRun(const edm::Run& run, const edm::EventSetup& es)
{
  /*
  _meTurnOnMET->getTH1F()->Add(_meGenMETTrg->getTH1F(),1);
  _meTurnOnMET->getTH1F()->Sumw2();
  _meGenMET->getTH1F()->Sumw2();
  _meTurnOnMET->getTH1F()->Divide(_meTurnOnMET->getTH1F(),_meGenMET->getTH1F(),1,1,"B");
  _meTurnOnJetPt->getTH1F()->Add(_meGenJetPtTrg->getTH1F(),1);
  _meTurnOnJetPt->getTH1F()->Sumw2();
  _meGenJetPt->getTH1F()->Sumw2();
  _meTurnOnJetPt->getTH1F()->Divide(_meTurnOnJetPt->getTH1F(),_meGenJetPt->getTH1F(),1,1,"B");
  
  
  float val, err, binc;
  for (int i=0;i<_meGenMET->getNbinsX();i++) {
    binc = _meGenMET->getBinContent(i+1);
    if (binc) {
      val = _meGenMETTrg->getBinContent(i+1)/binc;
      err = sqrt(val*(1-val)/binc);
      _meTurnOnMET->setBinContent(i+1,val);
      _meTurnOnMET->setBinError(i+1,err);
    }
  }
  */
}


void
HLTJetMETValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace l1extra;
  using namespace trigger;

  evtCnt++;
  if (writeFile_) test_histo->Fill(50.);

  //get The triggerEvent

  Handle<TriggerEventWithRefs> trigEv;
  iEvent.getByLabel(triggerEventObject_,trigEv);

//   if (trigEv.isValid()) {
//     printf("   found trigger information\n");
//   } else {
//     printf("   ERROR: no trigger information\n");
//   }
// 
//    if (trigEv.isValid()) {
//      int trigsize = trigEv->size();
//  
//      printf("  Number of filters = %i\n",trigsize);
//      for(int i=0; i<trigsize; i++){
//        cout << trigEv->filterTag(i) << endl;
//      }
//  
//    }
//  

// get TriggerResults object

  bool gotHLT=true;
  bool myTrig=false;

  Handle<TriggerResults> hltresults,hltresultsDummy;
  iEvent.getByLabel(HLTriggerResults,hltresults);
  if (! hltresults.isValid() ) { 
    //cout << "  -- No HLTRESULTS"; 
    //if (evtCnt==1) edm::LogWarning("HLTJetMETValidation") << "  -- No HLTRESULTS";    
    gotHLT=false;
  }

  if (gotHLT) {
    getHLTResults(*hltresults);
    //    trig_iter=hltTriggerMap.find(MyTrigger);
    trig_iter=hltTriggerMap.find(_HLTPath.label());
    if (trig_iter==hltTriggerMap.end()){
      //cout << "Could not find trigger path with name: " << _probefilter.label() << endl;
      //if (evtCnt==1) edm::LogWarning("HLTJetMETValidation") << "Could not find trigger path with name: " << _probefilter.label(); 
    }else{
      myTrig=trig_iter->second;
    }
  }

  Handle<CaloJetCollection> caloJets,caloJetsDummy;
  iEvent.getByLabel( CaloJetAlgorithm, caloJets );
  double calJetPt=-1.;
  double calJetEta=-999.;
  double calJetPhi=-999.;
  if (caloJets.isValid()) { 
    //Loop over the CaloJets and fill some histograms
    int jetInd = 0;
    for( CaloJetCollection::const_iterator cal = caloJets->begin(); cal != caloJets->end(); ++ cal ) {
      // std::cout << "CALO JET #" << jetInd << std::endl << cal->print() << std::endl;
      // h_ptCal->Fill( cal->pt() );
      if (jetInd == 0){
	//h_ptCalLeading->Fill( cal->pt() );
	calJetPt=cal->pt();
	calJetEta=cal->eta();
	calJetPhi=cal->phi();
	_meRecoJetPt->Fill( calJetPt );
	_meRecoJetEta->Fill( calJetEta );
	_meRecoJetPhi->Fill( calJetPhi );
	if (myTrig) _meRecoJetPtTrg->Fill( calJetPt );
	if (myTrig) _meRecoJetEtaTrg->Fill( calJetEta );
	if (myTrig) _meRecoJetPhiTrg->Fill( calJetPhi );

	//h_etaCalLeading->Fill( cal->eta() );
	//h_phiCalLeading->Fill( cal->phi() );
      

	jetInd++;
      }
    }
  }else{
    //cout << "  -- No CaloJets" << endl;
    //if (evtCnt==1) edm::LogWarning("HLTJetMETValidation") << "  -- No CaloJets"; 
  }

  Handle<GenJetCollection> genJets,genJetsDummy;
  iEvent.getByLabel( GenJetAlgorithm, genJets );
  double genJetPt=-1.;
  double genJetEta=-999.;
  double genJetPhi=-999.;

  if (genJets.isValid()) { 
    //Loop over the GenJets and fill some histograms
    int jetInd = 0;
    for( GenJetCollection::const_iterator gen = genJets->begin(); gen != genJets->end(); ++ gen ) {
      // std::cout << "CALO JET #" << jetInd << std::endl << cal->print() << std::endl;
      // h_ptCal->Fill( cal->pt() );
      if (jetInd == 0){
	//h_ptCalLeading->Fill( gen->pt() );
	genJetPt=gen->pt();
	genJetEta=gen->eta();
	genJetPhi=gen->phi();
	_meGenJetPt->Fill( genJetPt );
	_meGenJetEta->Fill( genJetEta );
	_meGenJetPhi->Fill( genJetPhi );
	if (myTrig) _meGenJetPtTrg->Fill( genJetPt );
	if (myTrig) _meGenJetEtaTrg->Fill( genJetEta );
	if (myTrig) _meGenJetPhiTrg->Fill( genJetPhi );
	//h_etaCalLeading->Fill( gen->eta() );
	//h_phiCalLeading->Fill( gen->phi() );
      
	//if (myTrig) h_ptCalTrig->Fill( gen->pt() );
	jetInd++;
      }
    }
  }else{
    //cout << "  -- No GenJets" << endl;
    //if (evtCnt==1) edm::LogWarning("HLTJetMETValidation") << "  -- No GenJets"; 
  }

  edm::Handle<CaloMETCollection> recmet, recmetDummy;
  iEvent.getByLabel(CaloMETColl,recmet);

  double calMet=-1;
  if (recmet.isValid()) { 
    typedef CaloMETCollection::const_iterator cmiter;
    //cout << "Size of MET collection" <<  recmet.size() << endl;
    for ( cmiter i=recmet->begin(); i!=recmet->end(); i++) {
      calMet = i->pt();
      //mcalphi = i->phi();
      //mcalsum = i->sumEt();
      _meRecoMET -> Fill(calMet);
      if (myTrig) _meRecoMETTrg -> Fill(calMet);
    }
  }else{
    //cout << "  -- No MET Collection with name: " << CaloMETColl << endl;
    //if (evtCnt==1) edm::LogWarning("HLTJetMETValidation") << "  -- No MET Collection with name: "<< CaloMETColl; 
  }

  edm::Handle<GenMETCollection> genmet, genmetDummy;
  iEvent.getByLabel(GenMETColl,genmet);

  double genMet=-1;
  if (genmet.isValid()) { 
    typedef GenMETCollection::const_iterator cmiter;
    //cout << "Size of GenMET collection" <<  recmet.size() << endl;
    for ( cmiter i=genmet->begin(); i!=genmet->end(); i++) {
      genMet = i->pt();
      //mcalphi = i->phi();
      //mcalsum = i->sumEt();
      _meGenMET -> Fill(genMet);
      if (myTrig) _meGenMETTrg -> Fill(genMet);
    }
  }else{
    //cout << "  -- No GenMET Collection with name: " << GenMETColl << endl;
    //if (evtCnt==1) edm::LogWarning("HLTJetMETValidation") << "  -- No GenMET Collection with name: "<< GenMETColl; 
  }

  // get the reference and probe jets
  size_t FLT_HLTREF = 0;
  FLT_HLTREF = trigEv->filterIndex(_reffilter); // get the position for the reference filter
  //cout << "   FLT_HLTREF = " << FLT_HLTREF << endl;
  size_t FLT_HLTPROBE = 0;
  FLT_HLTPROBE = trigEv->filterIndex(_probefilter); // get the position for the probe filter
  //cout << "   FLT_HLTPROBE = " << FLT_HLTPROBE << endl;
  

  if (FLT_HLTREF != trigEv->size()){

  }

  if (FLT_HLTPROBE != trigEv->size()){
      size_t HLTProbeJetID = 0;
      HLTProbeJetID =trigEv->filterIndex(_probefilter);
      VRjet probejets;
      trigEv->getObjects(HLTProbeJetID,trigger::TriggerJet,probejets);
      if (probejets.size() > 0){
	if (calJetPt> 0) _meRecoJetPtProbe->Fill( calJetPt );    
	if (calJetPt> 0) _meRecoJetEtaProbe->Fill( calJetEta );    
	if (calJetPt> 0) _meRecoJetPhiProbe->Fill( calJetPhi );    
	if (genJetPt> 0) _meGenJetPtProbe ->Fill( genJetPt ); 
	if (genJetPt> 0) _meGenJetEtaProbe ->Fill( genJetEta ); 
	if (genJetPt> 0) _meGenJetPhiProbe ->Fill( genJetPhi ); 
	if (calMet  > 0) _meRecoMETProbe  ->Fill(calMet);
      }
  }

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
    // cout << "   refjets.size = " << refjets.size() << endl;
    if (refjets.size() > 0){
      // fill leading jet that fired the reference trigger
      _meRefPt->Fill((*refjets[0]).pt());
      if (calJetPt>0) _meRecoJetPtRef->Fill( calJetPt );    
      if (calJetPt>0) _meRecoJetEtaRef->Fill( calJetEta );    
      if (calJetPt>0) _meRecoJetPhiRef->Fill( calJetPhi );    
      if (genJetPt>0) _meGenJetPtRef ->Fill( genJetPt ); 
      if (genJetPt>0) _meGenJetEtaRef ->Fill( genJetEta ); 
      if (genJetPt>0) _meGenJetPhiRef ->Fill( genJetPhi ); 
      if (calMet  > 0) _meRecoMETRef ->Fill(calMet);
      } // if (refjets.size() > 0){

    // then count how often the probe trigger was fired

    if (refjets.size() > 0 && FLT_HLTPROBE != trigEv->size()){
      NProbe++;
      // now get the object's four vector information
      size_t HLTProbeJetID = 0;
      HLTProbeJetID =trigEv->filterIndex(_probefilter);
      VRjet probejets;
      trigEv->getObjects(HLTProbeJetID,trigger::TriggerJet,probejets);
//       cout << "     probejets.size = " << probejets.size() << endl;
//       for (unsigned int i=0; i < probejets.size(); i++){
// 	cout << "  i = " << i << "    pt = " << (*probejets[i]).pt() << endl;
//       } // for (int i=0; i < probejets.size(); i++){

      if (probejets.size() > 0){
	// fill leading jet that fired the probe trigger
	_meProbePt->Fill((*probejets[0]).pt());
      } // if (probejets.size() > 0){
    } // if (FLT_HLTPROBE != trigEv->size()){
  } // if (FLT_HLTREF != trigEv->size()){

//JC  } 
//JC  else 
//JC  {
//JC    cout << "Handle invalid! Check InputTag provided." << endl;
//JC  }
//JC     
}




//JC   bool 
//JC   HLTJetMETValidation::match(const LV& jet,const LVColl& McInfo,double dr)
//JC   {
//JC    
//JC     bool matched=false;
//JC   
//JC     if(McInfo.size()>0)
//JC       for(std::vector<LV>::const_iterator it = McInfo.begin();it!=McInfo.end();++it)
//JC         {
//JC   	double delta = ROOT::Math::VectorUtil::DeltaR(jet,*it);
//JC   	if(delta<dr)
//JC   	  {
//JC   	    matched=true;
//JC   	  }
//JC         }
//JC   
//JC     return matched;
//JC   }
//JC   
//JC   std::vector<double>
//JC   HLTJetMETValidation::calcEfficiency(int num,int denom)
//JC   {
//JC     std::vector<double> a;
//JC     if(denom==0)
//JC       {
//JC         a.push_back(0.);
//JC         a.push_back(0.);
//JC       }
//JC     else
//JC       {    
//JC         a.push_back(((double)num)/((double)denom));
//JC         a.push_back(sqrt(a[0]*(1-a[0])/((double)denom)));
//JC       }
//JC     return a;
//JC   }

void HLTJetMETValidation::getHLTResults( const edm::TriggerResults& hltresults) {


  int ntrigs=hltresults.size();
  if (! HLTinit_){
    HLTinit_=true;
    triggerNames_.init(hltresults);
    
    //if (writeFile_) cout << "Number of HLT Paths: " << ntrigs << endl;

    // book histogram and label axis with trigger names
    //h_TriggerResults = fs->make<TH1F>( "TriggerResults", "HLT Results", ntrigs, 0, ntrigs );

    for (int itrig = 0; itrig != ntrigs; ++itrig){
      string trigName = triggerNames_.triggerName(itrig);
      // cout << "trigger " << itrig << ": " << trigName << endl; 
      //_triggerResults->GetXaxis()->SetBinLabel(itrig+1,trigName.c_str());
    }
  }

  
  for (int itrig = 0; itrig != ntrigs; ++itrig){
    string trigName = triggerNames_.triggerName(itrig);
     bool accept=hltresults.accept(itrig);

     if (accept) _triggerResults->Fill(float(itrig));

     // fill the trigger map
     typedef std::map<string,bool>::value_type valType;
     trig_iter=hltTriggerMap.find(trigName);
     if (trig_iter==hltTriggerMap.end())
       hltTriggerMap.insert(valType(trigName,accept));
     else
       trig_iter->second=accept;
  }
}

#include "HLTriggerOffline/JetMET/interface/HLTJetMETValidation.h"
#include "Math/GenVector/VectorUtil.h"
#include "FWCore/Common/interface/TriggerNames.h"

HLTJetMETValidation::HLTJetMETValidation(const edm::ParameterSet& ps) : 
  triggerEventObject_(ps.getUntrackedParameter<edm::InputTag>("triggerEventObject")),
  CaloJetAlgorithm( ps.getUntrackedParameter<edm::InputTag>( "CaloJetAlgorithm" ) ),
  GenJetAlgorithm( ps.getUntrackedParameter<edm::InputTag>( "GenJetAlgorithm" ) ),
  CaloMETColl( ps.getUntrackedParameter<edm::InputTag>( "CaloMETCollection" ) ),
  GenMETColl( ps.getUntrackedParameter<edm::InputTag>( "GenMETCollection" ) ),
  HLTriggerResults( ps.getParameter<edm::InputTag>( "HLTriggerResults" ) ),
  triggerTag_(ps.getUntrackedParameter<std::string>("DQMFolder","SingleJet")),
  _HLTPath(ps.getUntrackedParameter<edm::InputTag>("HLTPath")),
  _HLTLow(ps.getUntrackedParameter<edm::InputTag>("HLTLow")),
  outFile_(ps.getUntrackedParameter<std::string>("OutputFileName","")),
  HLTinit_(false),
  //JL
  writeFile_(ps.getUntrackedParameter<bool>("WriteFile",false))
{
  evtCnt=0;

  //Declare DQM Store
  DQMStore* store = &*edm::Service<DQMStore>();

  if(store)
    {
      //Create the histograms
      store->setCurrentFolder(triggerTag_);
      _meRecoJetPt= store->book1D("_meRecoJetPt","Single Reconstructed Jet Pt",100,0,500);
      _meRecoJetPtTrg= store->book1D("_meRecoJetPtTrg","Single Reconstructed Jet Pt -- HLT Triggered",100,0,500);
      _meRecoJetPtTrgLow= store->book1D("_meRecoJetPtTrgLow","Single Reconstructed Jet Pt -- HLT Triggered Low",100,0,500);

      _meRecoJetEta= store->book1D("_meRecoJetEta","Single Reconstructed Jet Eta",100,-10,10);
      _meRecoJetEtaTrg= store->book1D("_meRecoJetEtaTrg","Single Reconstructed Jet Eta -- HLT Triggered",100,-10,10);
      _meRecoJetEtaTrgLow= store->book1D("_meRecoJetEtaTrgLow","Single Reconstructed Jet Eta -- HLT Triggered Low",100,-10,10);

      _meRecoJetPhi= store->book1D("_meRecoJetPhi","Single Reconstructed Jet Phi",100,-4.,4.);
      _meRecoJetPhiTrg= store->book1D("_meRecoJetPhiTrg","Single Reconstructed Jet Phi -- HLT Triggered",100,-4.,4.);
      _meRecoJetPhiTrgLow= store->book1D("_meRecoJetPhiTrgLow","Single Reconstructed Jet Phi -- HLT Triggered Low",100,-4.,4.);

      _meGenJetPt= store->book1D("_meGenJetPt","Single Generated Jet Pt",100,0,500);
      _meGenJetPtTrg= store->book1D("_meGenJetPtTrg","Single Generated Jet Pt -- HLT Triggered",100,0,500);
      _meGenJetPtTrgLow= store->book1D("_meGenJetPtTrgLow","Single Generated Jet Pt -- HLT Triggered Low",100,0,500);

      _meGenJetEta= store->book1D("_meGenJetEta","Single Generated Jet Eta",100,-10,10);
      _meGenJetEtaTrg= store->book1D("_meGenJetEtaTrg","Single Generated Jet Eta -- HLT Triggered",100,-10,10);
      _meGenJetEtaTrgLow= store->book1D("_meGenJetEtaTrgLow","Single Generated Jet Eta -- HLT Triggered Low",100,-10,10);

      _meGenJetPhi= store->book1D("_meGenJetPhi","Single Generated Jet Phi",100,-4.,4.);
      _meGenJetPhiTrg= store->book1D("_meGenJetPhiTrg","Single Generated Jet Phi -- HLT Triggered",100,-4.,4.);
      _meGenJetPhiTrgLow= store->book1D("_meGenJetPhiTrgLow","Single Generated Jet Phi -- HLT Triggered Low",100,-4.,4.);

      _meRecoMET= store->book1D("_meRecoMET","Reconstructed Missing ET",100,0,500);
      _meRecoMETTrg= store->book1D("_meRecoMETTrg","Reconstructed Missing ET -- HLT Triggered",100,0,500);
      _meRecoMETTrgLow= store->book1D("_meRecoMETTrgLow","Reconstructed Missing ET -- HLT Triggered Low",100,0,500);

      _meGenMET= store->book1D("_meGenMET","Generated Missing ET",100,0,500);
      _meGenMETTrg= store->book1D("_meGenMETTrg","Generated Missing ET -- HLT Triggered",100,0,500);
      _meGenMETTrgLow= store->book1D("_meGenMETTrgLow","Generated Missing ET -- HLT Triggered Low",100,0,500);

      _meGenHT= store->book1D("_meGenHT","Generated HT",100,0,1000);
      _meGenHTTrg= store->book1D("_meGenHTTrg","Generated HT -- HLT Triggered",100,0,1000);
      _meGenHTTrgLow= store->book1D("_meGenHTTrgLow","Generated HT -- HLT Triggered Low",100,0,1000);

      _meRecoHT= store->book1D("_meRecoHT","Reconstructed HT",100,0,1000);
      _meRecoHTTrg= store->book1D("_meRecoHTTrg","Reconstructed HT -- HLT Triggered",100,0,1000);
      _meRecoHTTrgLow= store->book1D("_meRecoHTTrgLow","Reconstructed HT -- HLT Triggered Low",100,0,1000);

      _triggerResults = store->book1D( "_triggerResults", "HLT Results", 200, 0, 200 );

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

  //Write DQM thing..
  if(outFile_.size()>0)
    if (&*edm::Service<DQMStore>() && writeFile_) {
      edm::Service<DQMStore>()->save (outFile_);
    }
  
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
  //get The triggerEvent

  Handle<TriggerEventWithRefs> trigEv;
  iEvent.getByLabel(triggerEventObject_,trigEv);

// get TriggerResults object

  bool gotHLT=true;
  bool myTrig=false;
  bool myTrigLow=false;

  Handle<TriggerResults> hltresults,hltresultsDummy;
  iEvent.getByLabel(HLTriggerResults,hltresults);
  if (! hltresults.isValid() ) { 
    //std::cout << "  -- No HLTRESULTS"; 
    //if (evtCnt==1) edm::LogWarning("HLTJetMETValidation") << "  -- No HLTRESULTS";    
    gotHLT=false;
  }

  if (gotHLT) {
    const edm::TriggerNames & triggerNames = iEvent.triggerNames(*hltresults);
    getHLTResults(*hltresults, triggerNames);
    //    trig_iter=hltTriggerMap.find(MyTrigger);
    //trig_iter=hltTriggerMap.find(_HLTPath.label());

    //std::cout << _HLTPath.label() <<std::endl;

    for( map<std::string,bool>::iterator ii=hltTriggerMap.begin(); ii!=hltTriggerMap.end(); ++ii)
    {
      //std::cout << (*ii).first << ": " << (*ii).second << std::endl;
      //std::cout << (*ii).first << " : " << ((*ii).first).find(_HLTPath.label()) << " : " << string::npos << std::endl;

      // if _HLTPath.label() is found in the string
      if ( ((*ii).first).find(_HLTPath.label()) != string::npos) trig_iter=ii;
    }
    if (trig_iter==hltTriggerMap.end()){
      //std::cout << "Could not find trigger path with name: " << _probefilter.label() << std::endl;
      //if (evtCnt==1) edm::LogWarning("HLTJetMETValidation") << "Could not find trigger path with name: " << _probefilter.label(); 
    }else{
      myTrig=trig_iter->second;
    }
    //trig_iter=hltTriggerMap.find(_HLTLow.label());
    for( map<std::string,bool>::iterator ii=hltTriggerMap.begin(); ii!=hltTriggerMap.end(); ++ii)
    {
      // if _HLTPath.label() is found in the string
      if ( ((*ii).first).find(_HLTLow.label()) != string::npos) trig_iter=ii;
    }
    if (trig_iter==hltTriggerMap.end()){
      //std::cout << "Could not find trigger path with name: " << _probefilter.label() << std::endl;
      //if (evtCnt==1) edm::LogWarning("HLTJetMETValidation") << "Could not find trigger path with name: " << _probefilter.label(); 
    }else{
      myTrigLow=trig_iter->second;
    }
  }

  Handle<CaloJetCollection> caloJets,caloJetsDummy;
  iEvent.getByLabel( CaloJetAlgorithm, caloJets );
  double calJetPt=-1.;
  double calJetEta=-999.;
  double calJetPhi=-999.;
  double calHT=0;
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
	if (myTrigLow) _meRecoJetPtTrgLow->Fill( calJetPt );
	if (myTrigLow) _meRecoJetEtaTrgLow->Fill( calJetEta );
	if (myTrigLow) _meRecoJetPhiTrgLow->Fill( calJetPhi );

	//h_etaCalLeading->Fill( cal->eta() );
	//h_phiCalLeading->Fill( cal->phi() );
      

	jetInd++;
      }
      if (cal->pt()>30) {
	calHT+=cal->pt();
      }
    }
    _meRecoHT->Fill( calHT );
    if (myTrig) _meRecoHTTrg->Fill( calHT );
    if (myTrigLow) _meRecoHTTrgLow->Fill( calHT );
  }else{
    //std::cout << "  -- No CaloJets" << std::endl;
    //if (evtCnt==1) edm::LogWarning("HLTJetMETValidation") << "  -- No CaloJets"; 
  }

  Handle<GenJetCollection> genJets,genJetsDummy;
  iEvent.getByLabel( GenJetAlgorithm, genJets );
  double genJetPt=-1.;
  double genJetEta=-999.;
  double genJetPhi=-999.;
  double genHT=0;
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
	if (myTrigLow) _meGenJetPtTrgLow->Fill( genJetPt );
	if (myTrigLow) _meGenJetEtaTrgLow->Fill( genJetEta );
	if (myTrigLow) _meGenJetPhiTrgLow->Fill( genJetPhi );
	//h_etaCalLeading->Fill( gen->eta() );
	//h_phiCalLeading->Fill( gen->phi() );
      
	//if (myTrig) h_ptCalTrig->Fill( gen->pt() );
	jetInd++;
      }
      if (gen->pt()>30) {
	genHT+=gen->pt();
      }
    }
    _meGenHT->Fill( genHT );
    if (myTrig) _meGenHTTrg->Fill( genHT );
    if (myTrigLow) _meGenHTTrgLow->Fill( genHT );
  }else{
    //std::cout << "  -- No GenJets" << std::endl;
    //if (evtCnt==1) edm::LogWarning("HLTJetMETValidation") << "  -- No GenJets"; 
  }

  edm::Handle<CaloMETCollection> recmet, recmetDummy;
  iEvent.getByLabel(CaloMETColl,recmet);

  double calMet=-1;
  if (recmet.isValid()) { 
    typedef CaloMETCollection::const_iterator cmiter;
    //std::cout << "Size of MET collection" <<  recmet.size() << std::endl;
    for ( cmiter i=recmet->begin(); i!=recmet->end(); i++) {
      calMet = i->pt();
      //mcalphi = i->phi();
      //mcalsum = i->sumEt();
      _meRecoMET -> Fill(calMet);
      if (myTrig) _meRecoMETTrg -> Fill(calMet);
      if (myTrigLow) _meRecoMETTrgLow -> Fill(calMet);
    }
  }else{
    //std::cout << "  -- No MET Collection with name: " << CaloMETColl << std::endl;
    //if (evtCnt==1) edm::LogWarning("HLTJetMETValidation") << "  -- No MET Collection with name: "<< CaloMETColl; 
  }

  edm::Handle<GenMETCollection> genmet, genmetDummy;
  iEvent.getByLabel(GenMETColl,genmet);

  double genMet=-1;
  if (genmet.isValid()) { 
    typedef GenMETCollection::const_iterator cmiter;
    //std::cout << "Size of GenMET collection" <<  recmet.size() << std::endl;
    for ( cmiter i=genmet->begin(); i!=genmet->end(); i++) {
      genMet = i->pt();
      //mcalphi = i->phi();
      //mcalsum = i->sumEt();
      _meGenMET -> Fill(genMet);
      if (myTrig) _meGenMETTrg -> Fill(genMet);
      if (myTrigLow) _meGenMETTrgLow -> Fill(genMet);
    }
  }else{
    //std::cout << "  -- No GenMET Collection with name: " << GenMETColl << std::endl;
    //if (evtCnt==1) edm::LogWarning("HLTJetMETValidation") << "  -- No GenMET Collection with name: "<< GenMETColl; 
  }
}

void HLTJetMETValidation::getHLTResults(const edm::TriggerResults& hltresults,
                                        const edm::TriggerNames & triggerNames) {

  int ntrigs=hltresults.size();
  if (! HLTinit_){
    HLTinit_=true;
    
    //if (writeFile_) std::cout << "Number of HLT Paths: " << ntrigs << std::endl;
    for (int itrig = 0; itrig != ntrigs; ++itrig){
      std::string trigName = triggerNames.triggerName(itrig);
      // std::cout << "trigger " << itrig << ": " << trigName << std::endl; 
    }
  }
  
  for (int itrig = 0; itrig != ntrigs; ++itrig){
    std::string trigName = triggerNames.triggerName(itrig);
     bool accept=hltresults.accept(itrig);

     if (accept) _triggerResults->Fill(float(itrig));

     // fill the trigger map
     typedef std::map<std::string,bool>::value_type valType;
     trig_iter=hltTriggerMap.find(trigName);
     if (trig_iter==hltTriggerMap.end())
       hltTriggerMap.insert(valType(trigName,accept));
     else
       trig_iter->second=accept;
  }
}

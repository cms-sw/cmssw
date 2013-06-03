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
  patternJetTrg_(ps.getUntrackedParameter<std::string>("PatternJetTrg","")),
  patternMetTrg_(ps.getUntrackedParameter<std::string>("PatternMetTrg","")),
  patternMuTrg_(ps.getUntrackedParameter<std::string>("PatternMuTrg","")),
  outFile_(ps.getUntrackedParameter<std::string>("OutputFileName","")),
  HLTinit_(false),
  //JL
  writeFile_(ps.getUntrackedParameter<bool>("WriteFile",false))
{
  evtCnt=0;

  store = &*edm::Service<DQMStore>();

}

HLTJetMETValidation::~HLTJetMETValidation()
{
}

//
// member functions
//


void
HLTJetMETValidation::beginRun(const edm::Run & iRun, const edm::EventSetup & iSetup) {

  
  bool foundMuTrg = false;
  std::string trgMuNm;
  bool changedConfig;
  //--define search patterns
  /*
  TPRegexp patternJet("HLT_Jet([0-9]*)?(_v[0-9]*)?$");
  TPRegexp patternMet("HLT_(PF*)?M([E,H]*)?T([0-9]*)?(_v[0-9]*)?$");
  TPRegexp patternMu("HLT_Mu([0-9]*)?(_v[0-9]*)?$");
  */
  TPRegexp patternJet(patternJetTrg_);
  TPRegexp patternMet(patternMetTrg_);
  TPRegexp patternMu(patternMuTrg_);

  if (!hltConfig_.init(iRun, iSetup, "HLT", changedConfig)) {
    edm::LogError("HLTJetMET") << "Initialization of HLTConfigProvider failed!!"; 
    return;
  }
  std::vector<std::string> validTriggerNames = hltConfig_.triggerNames();
  for (size_t j = 0; j < validTriggerNames.size(); j++) {
    //---find the muon path
    if (TString(validTriggerNames[j]).Contains(patternMu)) {
      //std::cout <<validTriggerNames[j].c_str()<<std::endl;
      if (!foundMuTrg) trgMuNm = validTriggerNames[j].c_str();
      foundMuTrg = true;
    }
    //---find the jet paths
    if (TString(validTriggerNames[j]).Contains(patternJet)) {
      hltTrgJet.push_back(validTriggerNames[j]);
    }
    //---find the met paths
    if (TString(validTriggerNames[j]).Contains(patternMet)) {
      hltTrgMet.push_back(validTriggerNames[j]);
    }
  }
  
  //----set the denominator paths
  for (size_t it=0;it<hltTrgJet.size();it++) {
    if (it==0 && foundMuTrg) hltTrgJetLow.push_back(trgMuNm);//--lowest threshold uses muon path
    if (it==0 && !foundMuTrg) hltTrgJetLow.push_back(hltTrgJet[it]);//---if no muon then itself
    if (it!=0) hltTrgJetLow.push_back(hltTrgJet[it-1]);
    //std::cout<<hltTrgJet[it].c_str()<<" "<<hltTrgJetLow[it].c_str()<<std::endl;
  }
  int itm(0), itpm(0), itmh(0), itpmh(0);
  for (size_t it=0;it<hltTrgMet.size();it++) {
    if (TString(hltTrgMet[it]).Contains("PF")) {
      if (TString(hltTrgMet[it]).Contains("MHT")) {
	if( 0 == itpmh ) {
	  if( foundMuTrg ) hltTrgMetLow.push_back(trgMuNm);
	  else hltTrgMetLow.push_back(hltTrgMet[it]);
	}
	else hltTrgMetLow.push_back(hltTrgMet[it-1]);
	itpmh++;
      }
      if (TString(hltTrgMet[it]).Contains("MET")) {
	if( 0 == itpm ) {
	  if( foundMuTrg ) hltTrgMetLow.push_back(trgMuNm);
	  else hltTrgMetLow.push_back(hltTrgMet[it]);
	}
	else hltTrgMetLow.push_back(hltTrgMet[it-1]);
	itpm++;
      }
    }
    else {
      if (TString(hltTrgMet[it]).Contains("MHT")) {
	if( 0 == itmh ) {
	  if( foundMuTrg ) hltTrgMetLow.push_back(trgMuNm);
	  else hltTrgMetLow.push_back(hltTrgMet[it]);
	}
	else hltTrgMetLow.push_back(hltTrgMet[it-1]);
	itmh++;
      }
      if (TString(hltTrgMet[it]).Contains("MET")) {
	if( 0 == itm ) {
	  if( foundMuTrg ) hltTrgMetLow.push_back(trgMuNm);
	  else hltTrgMetLow.push_back(hltTrgMet[it]);
	}
	else hltTrgMetLow.push_back(hltTrgMet[it-1]);
	itm++;
      }
    }
    //std::cout<<hltTrgMet[it].c_str()<<" "<<hltTrgMetLow[it].c_str()<<std::endl;
  }

  //----define dqm folders and elements
  for (size_t it=0;it<hltTrgJet.size();it++) {
    //std::cout<<hltTrgJet[it].c_str()<<" "<<hltTrgJetLow[it].c_str()<<std::endl;
    //store->setCurrentFolder(triggerTag_+hltTrgJet[it].c_str());
    std::string trgPathName = HLTConfigProvider::removeVersion(triggerTag_+hltTrgJet[it].c_str());
    //std::cout << "str = " << triggerTag_+hltTrgJet[it].c_str() << std::endl;
    //std::cout << "trgPathName = " << trgPathName << std::endl;
    store->setCurrentFolder(trgPathName);
    //_meRecoJetPt= store->book1D("_meRecoJetPt","Single Reconstructed Jet Pt",100,0,500);
    _meRecoJetPt.push_back(store->book1D("_meRecoJetPt","Single Reconstructed Jet Pt",100,0,500));
    _meRecoJetPtTrgMC.push_back(store->book1D("_meRecoJetPtTrgMC","Single Reconstructed Jet Pt -- HLT Triggered",100,0,500));
    _meRecoJetPtTrg.push_back(store->book1D("_meRecoJetPtTrg","Single Reconstructed Jet Pt -- HLT Triggered",100,0,500));
    _meRecoJetPtTrgLow.push_back(store->book1D("_meRecoJetPtTrgLow","Single Reconstructed Jet Pt -- HLT Triggered Low",100,0,500));
    
    _meRecoJetEta.push_back(store->book1D("_meRecoJetEta","Single Reconstructed Jet Eta",100,-10,10));
    _meRecoJetEtaTrgMC.push_back(store->book1D("_meRecoJetEtaTrgMC","Single Reconstructed Jet Eta -- HLT Triggered",100,-10,10));
    _meRecoJetEtaTrg.push_back(store->book1D("_meRecoJetEtaTrg","Single Reconstructed Jet Eta -- HLT Triggered",100,-10,10));
    _meRecoJetEtaTrgLow.push_back(store->book1D("_meRecoJetEtaTrgLow","Single Reconstructed Jet Eta -- HLT Triggered Low",100,-10,10));
    
    _meRecoJetPhi.push_back(store->book1D("_meRecoJetPhi","Single Reconstructed Jet Phi",100,-4.,4.));
    _meRecoJetPhiTrgMC.push_back(store->book1D("_meRecoJetPhiTrgMC","Single Reconstructed Jet Phi -- HLT Triggered",100,-4.,4.));
    _meRecoJetPhiTrg.push_back(store->book1D("_meRecoJetPhiTrg","Single Reconstructed Jet Phi -- HLT Triggered",100,-4.,4.));
    _meRecoJetPhiTrgLow.push_back(store->book1D("_meRecoJetPhiTrgLow","Single Reconstructed Jet Phi -- HLT Triggered Low",100,-4.,4.));
    
    _meGenJetPt.push_back(store->book1D("_meGenJetPt","Single Generated Jet Pt",100,0,500));
    _meGenJetPtTrgMC.push_back(store->book1D("_meGenJetPtTrgMC","Single Generated Jet Pt -- HLT Triggered",100,0,500));
    _meGenJetPtTrg.push_back(store->book1D("_meGenJetPtTrg","Single Generated Jet Pt -- HLT Triggered",100,0,500));
    _meGenJetPtTrgLow.push_back(store->book1D("_meGenJetPtTrgLow","Single Generated Jet Pt -- HLT Triggered Low",100,0,500));
    
    _meGenJetEta.push_back(store->book1D("_meGenJetEta","Single Generated Jet Eta",100,-10,10));
    _meGenJetEtaTrgMC.push_back(store->book1D("_meGenJetEtaTrgMC","Single Generated Jet Eta -- HLT Triggered",100,-10,10));
    _meGenJetEtaTrg.push_back(store->book1D("_meGenJetEtaTrg","Single Generated Jet Eta -- HLT Triggered",100,-10,10));
    _meGenJetEtaTrgLow.push_back(store->book1D("_meGenJetEtaTrgLow","Single Generated Jet Eta -- HLT Triggered Low",100,-10,10));
    
    _meGenJetPhi.push_back(store->book1D("_meGenJetPhi","Single Generated Jet Phi",100,-4.,4.));
    _meGenJetPhiTrgMC.push_back(store->book1D("_meGenJetPhiTrgMC","Single Generated Jet Phi -- HLT Triggered",100,-4.,4.));
    _meGenJetPhiTrg.push_back(store->book1D("_meGenJetPhiTrg","Single Generated Jet Phi -- HLT Triggered",100,-4.,4.));
    _meGenJetPhiTrgLow.push_back(store->book1D("_meGenJetPhiTrgLow","Single Generated Jet Phi -- HLT Triggered Low",100,-4.,4.));
    
  }
  for (size_t it=0;it<hltTrgMet.size();it++) {
    //std::cout<<hltTrgMet[it].c_str()<<" "<<hltTrgMetLow[it].c_str()<<std::endl;
    //store->setCurrentFolder(triggerTag_+hltTrgMet[it].c_str());
    std::string trgPathName = HLTConfigProvider::removeVersion(triggerTag_+hltTrgMet[it].c_str());
    store->setCurrentFolder(trgPathName);
    _meRecoMET.push_back(store->book1D("_meRecoMET","Reconstructed Missing ET",100,0,500));
    _meRecoMETTrgMC.push_back(store->book1D("_meRecoMETTrgMC","Reconstructed Missing ET -- HLT Triggered",100,0,500));
    _meRecoMETTrg.push_back(store->book1D("_meRecoMETTrg","Reconstructed Missing ET -- HLT Triggered",100,0,500));
    _meRecoMETTrgLow.push_back(store->book1D("_meRecoMETTrgLow","Reconstructed Missing ET -- HLT Triggered Low",100,0,500));
    
    _meGenMET.push_back(store->book1D("_meGenMET","Generated Missing ET",100,0,500));
    _meGenMETTrgMC.push_back(store->book1D("_meGenMETTrgMC","Generated Missing ET -- HLT Triggered",100,0,500));
    _meGenMETTrg.push_back(store->book1D("_meGenMETTrg","Generated Missing ET -- HLT Triggered",100,0,500));
    _meGenMETTrgLow.push_back(store->book1D("_meGenMETTrgLow","Generated Missing ET -- HLT Triggered Low",100,0,500));
  }
}

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
  //bool myTrig=false;
  //bool myTrigLow=false;
  std::vector<bool> myTrigJ;
  for (size_t it=0;it<hltTrgJet.size();it++) myTrigJ.push_back(false);
  std::vector<bool> myTrigJLow;
  for (size_t it=0;it<hltTrgJetLow.size();it++) myTrigJLow.push_back(false);
  std::vector<bool> myTrigM;
  for (size_t it=0;it<hltTrgMet.size();it++) myTrigM.push_back(false);
  std::vector<bool> myTrigMLow;
  for (size_t it=0;it<hltTrgMetLow.size();it++) myTrigMLow.push_back(false);


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

    //---pick-up the jet trigger decisions
    for (size_t it=0;it<hltTrgJet.size();it++) {
      trig_iter=hltTriggerMap.find(hltTrgJet[it]);
      if (trig_iter==hltTriggerMap.end()){
	//std::cout << "Could not find trigger path with name: " << _probefilter.label() << std::endl;
	//if (evtCnt==1) edm::LogWarning("HLTJetMETValidation") << "Could not find trigger path with name: " << _probefilter.label(); 
      }else{
	myTrigJ[it]=trig_iter->second;
      }
      //std::cout<<hltTrgJet[it].c_str()<<" "<<myTrigJ[it]<<std::endl;
    }
    for (size_t it=0;it<hltTrgJetLow.size();it++) {
      trig_iter=hltTriggerMap.find(hltTrgJetLow[it]);
      if (trig_iter==hltTriggerMap.end()){
	//std::cout << "Could not find trigger path with name: " << _probefilter.label() << std::endl;
	//if (evtCnt==1) edm::LogWarning("HLTJetMETValidation") << "Could not find trigger path with name: " << _probefilter.label(); 
      }else{
	myTrigJLow[it]=trig_iter->second;
      }
      //std::cout<<hltTrgJetLow[it].c_str()<<" "<<myTrigJLow[it]<<std::endl;
    }
    //---pick-up the met trigger decisions
    for (size_t it=0;it<hltTrgMet.size();it++) {
      trig_iter=hltTriggerMap.find(hltTrgMet[it]);
      if (trig_iter==hltTriggerMap.end()){
	//std::cout << "Could not find trigger path with name: " << _probefilter.label() << std::endl;
	//if (evtCnt==1) edm::LogWarning("HLTJetMETValidation") << "Could not find trigger path with name: " << _probefilter.label(); 
      }else{
	myTrigM[it]=trig_iter->second;
      }
      //std::cout<<hltTrgMet[it].c_str()<<" "<<myTrigM[it]<<std::endl;
    }
    for (size_t it=0;it<hltTrgMetLow.size();it++) {
      trig_iter=hltTriggerMap.find(hltTrgMetLow[it]);
      if (trig_iter==hltTriggerMap.end()){
	//std::cout << "Could not find trigger path with name: " << _probefilter.label() << std::endl;
	//if (evtCnt==1) edm::LogWarning("HLTJetMETValidation") << "Could not find trigger path with name: " << _probefilter.label(); 
      }else{
	myTrigMLow[it]=trig_iter->second;
      }
      //std::cout<<hltTrgMetLow[it].c_str()<<" "<<myTrigMLow[it]<<std::endl;
    }
  }

  Handle<PFJetCollection> caloJets,caloJetsDummy;
  iEvent.getByLabel( CaloJetAlgorithm, caloJets );
  double calJetPt=-1.;
  double calJetEta=-999.;
  double calJetPhi=-999.;
  //double calHT=0;
  if (caloJets.isValid()) { 
    //Loop over the CaloJets and fill some histograms
    int jetInd = 0;
    for( PFJetCollection::const_iterator cal = caloJets->begin(); cal != caloJets->end(); ++ cal ) {
      //std::cout << "CALO JET #" << jetInd << std::endl << cal->print() << std::endl;
      if (jetInd == 0){
	calJetPt=cal->pt();
	calJetEta=cal->eta();
	calJetPhi=cal->phi();
	for (size_t it=0;it<hltTrgJet.size();it++) {
	  _meRecoJetPt[it]->Fill( calJetPt );
	  _meRecoJetEta[it]->Fill( calJetEta );
	  _meRecoJetPhi[it]->Fill( calJetPhi );
	  if (myTrigJ[it]) _meRecoJetPtTrgMC[it]->Fill( calJetPt );
	  if (myTrigJ[it]) _meRecoJetEtaTrgMC[it]->Fill( calJetEta );
	  if (myTrigJ[it]) _meRecoJetPhiTrgMC[it]->Fill( calJetPhi );
	  if (myTrigJ[it] && myTrigJLow[it]) _meRecoJetPtTrg[it]->Fill( calJetPt );
	  if (myTrigJ[it] && myTrigJLow[it]) _meRecoJetEtaTrg[it]->Fill( calJetEta );
	  if (myTrigJ[it] && myTrigJLow[it]) _meRecoJetPhiTrg[it]->Fill( calJetPhi );
	  if (myTrigJLow[it]) _meRecoJetPtTrgLow[it]->Fill( calJetPt );
	  if (myTrigJLow[it]) _meRecoJetEtaTrgLow[it]->Fill( calJetEta );
	  if (myTrigJLow[it]) _meRecoJetPhiTrgLow[it]->Fill( calJetPhi );
	}
	jetInd++;
      }
      /*
      if (cal->pt()>30) {
	calHT+=cal->pt();
      }
      */
    }
    /*
    _meRecoHT->Fill( calHT );
    if (myTrig) _meRecoHTTrg->Fill( calHT );
    if (myTrigLow) _meRecoHTTrgLow->Fill( calHT );
    */
  }else{
    //std::cout << "  -- No CaloJets" << std::endl;
    //if (evtCnt==1) edm::LogWarning("HLTJetMETValidation") << "  -- No CaloJets"; 
  }

  Handle<GenJetCollection> genJets,genJetsDummy;
  iEvent.getByLabel( GenJetAlgorithm, genJets );
  double genJetPt=-1.;
  double genJetEta=-999.;
  double genJetPhi=-999.;
  //double genHT=0;
  if (genJets.isValid()) { 
    //Loop over the GenJets and fill some histograms
    int jetInd = 0;
    for( GenJetCollection::const_iterator gen = genJets->begin(); gen != genJets->end(); ++ gen ) {
      // std::cout << "CALO JET #" << jetInd << std::endl << cal->print() << std::endl;
      if (jetInd == 0){
	genJetPt=gen->pt();
	genJetEta=gen->eta();
	genJetPhi=gen->phi();
	for (size_t it=0;it<hltTrgJet.size();it++) {
	  _meGenJetPt[it]->Fill( genJetPt );
	  _meGenJetEta[it]->Fill( genJetEta );
	  _meGenJetPhi[it]->Fill( genJetPhi );
	  if (myTrigJ[it]) _meGenJetPtTrgMC[it]->Fill( genJetPt );
	  if (myTrigJ[it]) _meGenJetEtaTrgMC[it]->Fill( genJetEta );
	  if (myTrigJ[it]) _meGenJetPhiTrgMC[it]->Fill( genJetPhi );
	  if (myTrigJ[it] && myTrigJLow[it]) _meGenJetPtTrg[it]->Fill( genJetPt );
	  if (myTrigJ[it] && myTrigJLow[it]) _meGenJetEtaTrg[it]->Fill( genJetEta );
	  if (myTrigJ[it] && myTrigJLow[it]) _meGenJetPhiTrg[it]->Fill( genJetPhi );
	  if (myTrigJLow[it]) _meGenJetPtTrgLow[it]->Fill( genJetPt );
	  if (myTrigJLow[it]) _meGenJetEtaTrgLow[it]->Fill( genJetEta );
	  if (myTrigJLow[it]) _meGenJetPhiTrgLow[it]->Fill( genJetPhi );
	}
	jetInd++;
      }
      /*
      if (gen->pt()>30) {
	genHT+=gen->pt();
      }
      */
    }
    /*
    _meGenHT->Fill( genHT );
    if (myTrig) _meGenHTTrg->Fill( genHT );
    if (myTrigLow) _meGenHTTrgLow->Fill( genHT );
    */
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
      for (size_t it=0;it<hltTrgMet.size();it++) {
	_meRecoMET[it] -> Fill(calMet);
	if (myTrigM[it]) _meRecoMETTrgMC[it] -> Fill(calMet);
	if (myTrigM[it] && myTrigMLow[it]) _meRecoMETTrg[it] -> Fill(calMet);
	if (myTrigMLow[it]) _meRecoMETTrgLow[it] -> Fill(calMet);
      }
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
      for (size_t it=0;it<hltTrgMet.size();it++) {
	_meGenMET[it] -> Fill(genMet);
	if (myTrigM[it]) _meGenMETTrgMC[it] -> Fill(genMet);
	if (myTrigM[it] && myTrigMLow[it]) _meGenMETTrg[it] -> Fill(genMet);
	if (myTrigMLow[it]) _meGenMETTrgLow[it] -> Fill(genMet);
      }
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

     //if (accept) _triggerResults->Fill(float(itrig));

     // fill the trigger map
     typedef std::map<std::string,bool>::value_type valType;
     trig_iter=hltTriggerMap.find(trigName);
     if (trig_iter==hltTriggerMap.end())
       hltTriggerMap.insert(valType(trigName,accept));
     else
       trig_iter->second=accept;
  }
}

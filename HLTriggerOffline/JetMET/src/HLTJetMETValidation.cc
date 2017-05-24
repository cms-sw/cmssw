// Migrated to use DQMEDAnalyzer by: Jyothsna Rani Komaragiri, Oct 2014

#include "HLTriggerOffline/JetMET/interface/HLTJetMETValidation.h"
#include "Math/GenVector/VectorUtil.h"
#include "FWCore/Common/interface/TriggerNames.h"

using namespace std;
using namespace edm;
using namespace reco;
using namespace l1extra;
using namespace trigger;


HLTJetMETValidation::HLTJetMETValidation(const edm::ParameterSet& ps) : 
  triggerEventObject_(consumes<TriggerEventWithRefs>(ps.getUntrackedParameter<edm::InputTag>("triggerEventObject"))),
  PFJetAlgorithm( consumes<PFJetCollection>(ps.getUntrackedParameter<edm::InputTag>( "PFJetAlgorithm" ) )),
  GenJetAlgorithm( consumes<GenJetCollection>(ps.getUntrackedParameter<edm::InputTag>( "GenJetAlgorithm" ) )),
  CaloMETColl( consumes<CaloMETCollection>(ps.getUntrackedParameter<edm::InputTag>( "CaloMETCollection" ) )),
  GenMETColl( consumes<GenMETCollection>(ps.getUntrackedParameter<edm::InputTag>( "GenMETCollection" ) )),
  HLTriggerResults(consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>( "HLTriggerResults" ) )),
  triggerTag_(ps.getUntrackedParameter<std::string>("DQMFolder","SingleJet")),
  patternJetTrg_(ps.getUntrackedParameter<std::string>("PatternJetTrg","")),
  patternMetTrg_(ps.getUntrackedParameter<std::string>("PatternMetTrg","")),
  patternMuTrg_(ps.getUntrackedParameter<std::string>("PatternMuTrg","")),
  HLTinit_(false)
{
  evtCnt=0;
}

HLTJetMETValidation::~HLTJetMETValidation()
{
}

//
// member functions
//

// ------------ method called when starting to processes a run ------------
void
HLTJetMETValidation::dqmBeginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {

  bool foundMuTrg = false;
  std::string trgMuNm;
  bool changedConfig = true;
  
  //--define search patterns
  TPRegexp patternJet(patternJetTrg_);
  TPRegexp patternMet(patternMetTrg_);
  TPRegexp patternMu(patternMuTrg_);

  if (!hltConfig_.init(iRun, iSetup, "HLT", changedConfig)) {
    edm::LogError("HLTJetMETValidation") << "Initialization of HLTConfigProvider failed!!"; 
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

}

// ------------ method called to book histograms before starting event loop ------------
void 
HLTJetMETValidation::bookHistograms(DQMStore::IBooker & iBooker, edm::Run const & iRun, edm::EventSetup const & iSetup)
{

  //----define DQM folders and elements
  iBooker.setCurrentFolder(triggerTag_+"JetResponse");
  TestJetResponse = iBooker.book1D("TestJetResponse","Single HLT Jet Pt",100,0,500);
  // a whole HCal Area!
  _meHLTGenJetRes[0]        = iBooker.book1D("_meHLTGenJetRes_all","HLTJet Pt/ GenJet Pt",100,0,2);
  _meHLTGenJetRes[1]        = iBooker.book1D("_meHLTGenJetRes_1","HLTJet Pt/ GenJet Pt (HLT Jet Pt (20~60GeV))",100,0,2);
  _meHLTGenJetRes[2]        = iBooker.book1D("_meHLTGenJetRes_2","HLTJet Pt/ GenJet Pt (HLT Jet Pt (60~80GeV))",100,0,2);
  _meHLTGenJetRes[3]        = iBooker.book1D("_meHLTGenJetRes_3","HLTJet Pt/ GenJet Pt (HLT Jet Pt (80~100GeV))",100,0,2);
  _meHLTGenJetRes[4]        = iBooker.book1D("_meHLTGenJetRes_4","HLTJet Pt/ GenJet Pt (HLT Jet Pt (100~150GeV))",100,0,2);
  _meHLTGenJetRes[5]        = iBooker.book1D("_meHLTGenJetRes_5","HLTJet Pt/ GenJet Pt (HLT Jet Pt (150~250GeV))",100,0,2);
  _meHLTGenJetRes[6]        = iBooker.book1D("_meHLTGenJetRes_6","HLTJet Pt/ GenJet Pt (HLT Jet Pt (>250GeV))",100,0,2);
  _meHLTGenJetResVsGenJetPt = iBooker.book2D("_meHLTGenJetResVsGenJetPt","HLTJet Pt/ GenJet Pt Vs GenJet Pt", 98,20,1000,100,0,2);
  // HEM17!
  _meHLTGenJetResHEM17[0]        = iBooker.book1D("_meHLTGenJetResHEM17_all","HLTJet Pt/ GenJet Pt",100,0,2);
  _meHLTGenJetResHEM17[1]        = iBooker.book1D("_meHLTGenJetResHEM17_1","HLTJet Pt/ GenJet Pt (HLT Jet Pt (20~60GeV) HEM17)",100,0,2);
  _meHLTGenJetResHEM17[2]        = iBooker.book1D("_meHLTGenJetResHEM17_2","HLTJet Pt/ GenJet Pt (HLT Jet Pt (60~80GeV) HEM17)",100,0,2);
  _meHLTGenJetResHEM17[3]        = iBooker.book1D("_meHLTGenJetResHEM17_3","HLTJet Pt/ GenJet Pt (HLT Jet Pt (80~100GeV) HEM17)",100,0,2);
  _meHLTGenJetResHEM17[4]        = iBooker.book1D("_meHLTGenJetResHEM17_4","HLTJet Pt/ GenJet Pt (HLT Jet Pt (100~150GeV) HEM17)",100,0,2);
  _meHLTGenJetResHEM17[5]        = iBooker.book1D("_meHLTGenJetResHEM17_5","HLTJet Pt/ GenJet Pt (HLT Jet Pt (150~250GeV) HEM17)",100,0,2);
  _meHLTGenJetResHEM17[6]        = iBooker.book1D("_meHLTGenJetResHEM17_6","HLTJet Pt/ GenJet Pt (HLT Jet Pt (>250GeV) HEM17)",100,0,2);
  _meHLTGenJetResVsGenJetPtHEM17 = iBooker.book2D("_meHLTGenJetResVsGenJetPtHEM17","HLTJet Pt/ GenJet Pt Vs GenJet Pt (HEM17)", 98,20,1000,100,0,2);
  // HEP17!
  _meHLTGenJetResHEP17[0]        = iBooker.book1D("_meHLTGenJetResHEP17_all","HLTJet Pt/ GenJet Pt",100,0,2);
  _meHLTGenJetResHEP17[1]        = iBooker.book1D("_meHLTGenJetResHEP17_1","HLTJet Pt/ GenJet Pt (HLT Jet Pt (20~60GeV) HEP17)",100,0,2);
  _meHLTGenJetResHEP17[2]        = iBooker.book1D("_meHLTGenJetResHEP17_2","HLTJet Pt/ GenJet Pt (HLT Jet Pt (60~80GeV) HEP17)",100,0,2);
  _meHLTGenJetResHEP17[3]        = iBooker.book1D("_meHLTGenJetResHEP17_3","HLTJet Pt/ GenJet Pt (HLT Jet Pt (80~100GeV) HEP17)",100,0,2);
  _meHLTGenJetResHEP17[4]        = iBooker.book1D("_meHLTGenJetResHEP17_4","HLTJet Pt/ GenJet Pt (HLT Jet Pt (100~150GeV) HEP17)",100,0,2);
  _meHLTGenJetResHEP17[5]        = iBooker.book1D("_meHLTGenJetResHEP17_5","HLTJet Pt/ GenJet Pt (HLT Jet Pt (150~250GeV) HEP17)",100,0,2);
  _meHLTGenJetResHEP17[6]        = iBooker.book1D("_meHLTGenJetResHEP17_6","HLTJet Pt/ GenJet Pt (HLT Jet Pt (>250GeV) HEP17)",100,0,2);
  _meHLTGenJetResVsGenJetPtHEP17 = iBooker.book2D("_meHLTGenJetResVsGenJetPtHEP17","HLTJet Pt/ GenJet Pt Vs GenJet Pt (HEP17)", 98,20,1000,100,0,2);

  // HEP18!
  _meHLTGenJetResHEP18[0]        = iBooker.book1D("_meHLTGenJetResHEP18_all","HLTJet Pt/ GenJet Pt",100,0,2);
  _meHLTGenJetResHEP18[1]        = iBooker.book1D("_meHLTGenJetResHEP18_1","HLTJet Pt/ GenJet Pt (HLT Jet Pt (20~60GeV) HEP18)",100,0,2);
  _meHLTGenJetResHEP18[2]        = iBooker.book1D("_meHLTGenJetResHEP18_2","HLTJet Pt/ GenJet Pt (HLT Jet Pt (60~80GeV) HEP18)",100,0,2);
  _meHLTGenJetResHEP18[3]        = iBooker.book1D("_meHLTGenJetResHEP18_3","HLTJet Pt/ GenJet Pt (HLT Jet Pt (80~100GeV) HEP18)",100,0,2);
  _meHLTGenJetResHEP18[4]        = iBooker.book1D("_meHLTGenJetResHEP18_4","HLTJet Pt/ GenJet Pt (HLT Jet Pt (100~150GeV) HEP18)",100,0,2);
  _meHLTGenJetResHEP18[5]        = iBooker.book1D("_meHLTGenJetResHEP18_5","HLTJet Pt/ GenJet Pt (HLT Jet Pt (150~250GeV) HEP18)",100,0,2);
  _meHLTGenJetResHEP18[6]        = iBooker.book1D("_meHLTGenJetResHEP18_6","HLTJet Pt/ GenJet Pt (HLT Jet Pt (>250GeV) HEP18)",100,0,2);
  _meHLTGenJetResVsGenJetPtHEP18 = iBooker.book2D("_meHLTGenJetResVsGenJetPtHEP18","HLTJet Pt/ GenJet Pt Vs GenJet Pt (HEP18)", 98,20,1000,100,0,2);
  for (size_t it=0;it<hltTrgJet.size();it++) {
    //std::cout<<hltTrgJet[it].c_str()<<" "<<hltTrgJetLow[it].c_str()<<std::endl;
    std::string trgPathName = HLTConfigProvider::removeVersion(triggerTag_+hltTrgJet[it].c_str());
    //std::cout << "str = " << triggerTag_+hltTrgJet[it].c_str() << std::endl;
    iBooker.setCurrentFolder(trgPathName);
    _meHLTJetPt.push_back(iBooker.book1D("_meHLTJetPt","Single HLT Jet Pt",100,0,500));
    _meHLTJetPtTrgMC.push_back(iBooker.book1D("_meHLTJetPtTrgMC","Single HLT Jet Pt - HLT Triggered",100,0,500));
    _meHLTJetPtTrg.push_back(iBooker.book1D("_meHLTJetPtTrg","Single HLT Jet Pt - HLT Triggered",100,0,500));
    _meHLTJetPtTrgLow.push_back(iBooker.book1D("_meHLTJetPtTrgLow","Single HLT Jet Pt - HLT Triggered Low",100,0,500));
    
    _meHLTJetEta.push_back(iBooker.book1D("_meHLTJetEta","Single HLT Jet Eta",100,-10,10));
    _meHLTJetEtaTrgMC.push_back(iBooker.book1D("_meHLTJetEtaTrgMC","Single HLT Jet Eta - HLT Triggered",100,-10,10));
    _meHLTJetEtaTrg.push_back(iBooker.book1D("_meHLTJetEtaTrg","Single HLT Jet Eta - HLT Triggered",100,-10,10));
    _meHLTJetEtaTrgLow.push_back(iBooker.book1D("_meHLTJetEtaTrgLow","Single HLT Jet Eta - HLT Triggered Low",100,-10,10));
    
    _meHLTJetPhi.push_back(iBooker.book1D("_meHLTJetPhi","Single HLT Jet Phi",100,-4.,4.));
    _meHLTJetPhiTrgMC.push_back(iBooker.book1D("_meHLTJetPhiTrgMC","Single HLT Jet Phi - HLT Triggered",100,-4.,4.));
    _meHLTJetPhiTrg.push_back(iBooker.book1D("_meHLTJetPhiTrg","Single HLT Jet Phi - HLT Triggered",100,-4.,4.));
    _meHLTJetPhiTrgLow.push_back(iBooker.book1D("_meHLTJetPhiTrgLow","Single HLT Jet Phi - HLT Triggered Low",100,-4.,4.));
    
    _meGenJetPt.push_back(iBooker.book1D("_meGenJetPt","Single Generated Jet Pt",100,0,500));
    _meGenJetPtTrgMC.push_back(iBooker.book1D("_meGenJetPtTrgMC","Single Generated Jet Pt - HLT Triggered",100,0,500));
    _meGenJetPtTrg.push_back(iBooker.book1D("_meGenJetPtTrg","Single Generated Jet Pt - HLT Triggered",100,0,500));
    _meGenJetPtTrgLow.push_back(iBooker.book1D("_meGenJetPtTrgLow","Single Generated Jet Pt - HLT Triggered Low",100,0,500));
    
    _meGenJetEta.push_back(iBooker.book1D("_meGenJetEta","Single Generated Jet Eta",100,-10,10));
    _meGenJetEtaTrgMC.push_back(iBooker.book1D("_meGenJetEtaTrgMC","Single Generated Jet Eta - HLT Triggered",100,-10,10));
    _meGenJetEtaTrg.push_back(iBooker.book1D("_meGenJetEtaTrg","Single Generated Jet Eta - HLT Triggered",100,-10,10));
    _meGenJetEtaTrgLow.push_back(iBooker.book1D("_meGenJetEtaTrgLow","Single Generated Jet Eta - HLT Triggered Low",100,-10,10));
    
    _meGenJetPhi.push_back(iBooker.book1D("_meGenJetPhi","Single Generated Jet Phi",100,-4.,4.));
    _meGenJetPhiTrgMC.push_back(iBooker.book1D("_meGenJetPhiTrgMC","Single Generated Jet Phi - HLT Triggered",100,-4.,4.));
    _meGenJetPhiTrg.push_back(iBooker.book1D("_meGenJetPhiTrg","Single Generated Jet Phi - HLT Triggered",100,-4.,4.));
    _meGenJetPhiTrgLow.push_back(iBooker.book1D("_meGenJetPhiTrgLow","Single Generated Jet Phi - HLT Triggered Low",100,-4.,4.));
    
  }
  for (size_t it=0;it<hltTrgMet.size();it++) {
    //std::cout<<hltTrgMet[it].c_str()<<" "<<hltTrgMetLow[it].c_str()<<std::endl;
    std::string trgPathName = HLTConfigProvider::removeVersion(triggerTag_+hltTrgMet[it].c_str());
    iBooker.setCurrentFolder(trgPathName);
    _meHLTMET.push_back(iBooker.book1D("_meHLTMET","HLT Missing ET",100,0,500));
    _meHLTMETTrgMC.push_back(iBooker.book1D("_meHLTMETTrgMC","HLT Missing ET - HLT Triggered",100,0,500));
    _meHLTMETTrg.push_back(iBooker.book1D("_meHLTMETTrg","HLT Missing ET - HLT Triggered",100,0,500));
    _meHLTMETTrgLow.push_back(iBooker.book1D("_meHLTMETTrgLow","HLT Missing ET - HLT Triggered Low",100,0,500));
    
    _meGenMET.push_back(iBooker.book1D("_meGenMET","Generated Missing ET",100,0,500));
    _meGenMETTrgMC.push_back(iBooker.book1D("_meGenMETTrgMC","Generated Missing ET - HLT Triggered",100,0,500));
    _meGenMETTrg.push_back(iBooker.book1D("_meGenMETTrg","Generated Missing ET - HLT Triggered",100,0,500));
    _meGenMETTrgLow.push_back(iBooker.book1D("_meGenMETTrgLow","Generated Missing ET - HLT Triggered Low",100,0,500));
  }
}

// ------------ method called for each event ------------
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
  iEvent.getByToken(triggerEventObject_,trigEv);

  //get TriggerResults object
  bool gotHLT=true;
  std::vector<bool> myTrigJ;
  for (size_t it=0;it<hltTrgJet.size();it++) myTrigJ.push_back(false);
  std::vector<bool> myTrigJLow;
  for (size_t it=0;it<hltTrgJetLow.size();it++) myTrigJLow.push_back(false);
  std::vector<bool> myTrigM;
  for (size_t it=0;it<hltTrgMet.size();it++) myTrigM.push_back(false);
  std::vector<bool> myTrigMLow;
  for (size_t it=0;it<hltTrgMetLow.size();it++) myTrigMLow.push_back(false);


  Handle<TriggerResults> hltresults;
  iEvent.getByToken(HLTriggerResults,hltresults);
  if (! hltresults.isValid() ) { 
    //if (evtCnt==1) edm::LogWarning("HLTJetMETValidation") << "  -- No HLTRESULTS";    
    gotHLT=false;
  }

  if (gotHLT) {
    const edm::TriggerNames & triggerNames = iEvent.triggerNames(*hltresults);
    getHLTResults(*hltresults, triggerNames);

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
  Handle<PFJetCollection> pfJets;
  iEvent.getByToken( PFJetAlgorithm, pfJets );

  Handle<GenJetCollection> genJets;
  iEvent.getByToken( GenJetAlgorithm, genJets );

  //GenJets
  double genJetPt=-1.;
  double genJetEta=-999.;
  double genJetPhi=-999.;
  std::vector<double> v_genjet_pt;
  std::vector<double> v_genjet_eta;
  std::vector<double> v_genjet_phi;
  std::vector<double> v_genjet_energy;

  if (genJets.isValid()) { 
    //Loop over the GenJets and fill some histograms
    int jetInd = 0;
    for( GenJetCollection::const_iterator gen = genJets->begin(); gen != genJets->end(); ++ gen ) {
      if (jetInd == 0){
	genJetPt=gen->pt();
	genJetEta=gen->eta();
	genJetPhi=gen->phi();
        v_genjet_pt.push_back(genJetPt);
        v_genjet_eta.push_back(genJetEta);
        v_genjet_phi.push_back(genJetPhi);
        v_genjet_energy.push_back(gen->energy());
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
    }
  } 
  else{
    //std::cout << "  -- No GenJets" << std::endl;
    //if (evtCnt==1) edm::LogWarning("HLTJetMETValidation") << "  -- No GenJets"; 
  }
  // --- Fill histos for PFJet paths ---
  // HLT jets namely hltAK4PFJets
  double pfJetPt=-1.;
  double pfJetEta=-999.;
  double pfJetPhi=-999.;

  //for hlt jet 
  std::vector<double> v_hltjet_pt;
  std::vector<double> v_hltjet_eta;
  std::vector<double> v_hltjet_phi;
  std::vector<double> v_hltjet_energy;
  std::vector<double> v_hltjet_dr_genjet;
  std::vector<double> v_idx_genjet_matched;
  v_hltjet_pt.clear();
  v_hltjet_eta.clear();
  v_hltjet_phi.clear();
  v_hltjet_energy.clear();
  v_hltjet_dr_genjet.clear();
  v_idx_genjet_matched.clear();

  if (pfJets.isValid()) { 
    //Loop over the PFJets and fill some histograms
    int jetInd = 0;
    for( PFJetCollection::const_iterator pf = pfJets->begin(); pf != pfJets->end(); ++ pf ) {
      //std::cout << "PF JET #" << jetInd << std::endl << pf->print() << std::endl;
      if (jetInd == 0){
	pfJetPt=pf->pt();
	pfJetEta=pf->eta();
	pfJetPhi=pf->phi();
	for (size_t it=0;it<hltTrgJet.size();it++) {
	  _meHLTJetPt[it]->Fill( pfJetPt );
	  _meHLTJetEta[it]->Fill( pfJetEta );
	  _meHLTJetPhi[it]->Fill( pfJetPhi );
	  if (myTrigJ[it]) _meHLTJetPtTrgMC[it]->Fill( pfJetPt );
	  if (myTrigJ[it]) _meHLTJetEtaTrgMC[it]->Fill( pfJetEta );
	  if (myTrigJ[it]) _meHLTJetPhiTrgMC[it]->Fill( pfJetPhi );
	  if (myTrigJ[it] && myTrigJLow[it]) _meHLTJetPtTrg[it]->Fill( pfJetPt );
	  if (myTrigJ[it] && myTrigJLow[it]) _meHLTJetEtaTrg[it]->Fill( pfJetEta );
	  if (myTrigJ[it] && myTrigJLow[it]) _meHLTJetPhiTrg[it]->Fill( pfJetPhi );
	  if (myTrigJLow[it]) _meHLTJetPtTrgLow[it]->Fill( pfJetPt );
	  if (myTrigJLow[it]) _meHLTJetEtaTrgLow[it]->Fill( pfJetEta );
	  if (myTrigJLow[it]) _meHLTJetPhiTrgLow[it]->Fill( pfJetPhi );
	}
	jetInd++;
        float dRmin(1000);
        int genJetidx = 0;
        int matchedgenJetidx = 0;
        if (genJets.isValid()) { 
          for( GenJetCollection::const_iterator gen = genJets->begin(); gen != genJets->end(); ++ gen ) {
            float dR = deltaR(gen->eta(),gen->phi(),pfJetEta,pfJetPhi);
            if (dR < dRmin) { dRmin = dR; matchedgenJetidx = genJetidx; }
            genJetidx++;
          }
        }// end of genjetmatching
        v_hltjet_pt.push_back(pfJetPt);
        v_hltjet_eta.push_back(pfJetEta);
        v_hltjet_phi.push_back(pfJetPhi);
        v_hltjet_energy.push_back(pf->energy());
        v_hltjet_dr_genjet.push_back(dRmin);
        v_idx_genjet_matched.push_back(matchedgenJetidx);
        //cout << " dR " << dRmin << endl;
        //cout << "matchedgenJetidx !! " << matchedgenJetidx << endl; 
        if (pfJetPt < 20) {continue;}
        if (v_genjet_pt[matchedgenJetidx] < 20) {continue;}
        if (dRmin > 0.2) {continue;}
        TestJetResponse->Fill(pfJetPt); // Test Fill !!!
        _meHLTGenJetResVsGenJetPt->Fill(v_genjet_pt[matchedgenJetidx],pfJetPt/v_genjet_pt[matchedgenJetidx] );
        _meHLTGenJetRes[0]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
        if (isHEP17(pfJetEta,pfJetPhi)){ 
          _meHLTGenJetResHEP17[0]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          _meHLTGenJetResVsGenJetPtHEP17->Fill(v_genjet_pt[matchedgenJetidx],pfJetPt/v_genjet_pt[matchedgenJetidx] );
        }
        if (isHEM17(pfJetEta,pfJetPhi)){ 
          _meHLTGenJetResHEM17[0]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          _meHLTGenJetResVsGenJetPtHEM17->Fill(v_genjet_pt[matchedgenJetidx],pfJetPt/v_genjet_pt[matchedgenJetidx] );
        }
        if (isHEP18(pfJetEta,pfJetPhi)){ 
          _meHLTGenJetResHEP18[0]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          _meHLTGenJetResVsGenJetPtHEP18->Fill(v_genjet_pt[matchedgenJetidx],pfJetPt/v_genjet_pt[matchedgenJetidx] );
        }
        if (pfJetPt < 60) {
          _meHLTGenJetRes[1]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          if (isHEP17(pfJetEta,pfJetPhi)){ 
            _meHLTGenJetResHEP17[1]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          }
          if (isHEM17(pfJetEta,pfJetPhi)){ 
            _meHLTGenJetResHEM17[1]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          }
          if (isHEP18(pfJetEta,pfJetPhi)){ 
            _meHLTGenJetResHEP18[1]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          }
        }
        else if (pfJetPt < 80) {
          _meHLTGenJetRes[2]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          if (isHEP17(pfJetEta,pfJetPhi)){ 
            _meHLTGenJetResHEP17[2]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          }
          if (isHEM17(pfJetEta,pfJetPhi)){ 
            _meHLTGenJetResHEM17[2]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          }
          if (isHEP18(pfJetEta,pfJetPhi)){ 
            _meHLTGenJetResHEP18[2]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          }
        }
        else if (pfJetPt < 100) {
          _meHLTGenJetRes[3]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          if (isHEP17(pfJetEta,pfJetPhi)){ 
            _meHLTGenJetResHEP17[3]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          }
          if (isHEM17(pfJetEta,pfJetPhi)){ 
            _meHLTGenJetResHEM17[3]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          }
          if (isHEP18(pfJetEta,pfJetPhi)){ 
            _meHLTGenJetResHEP18[3]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          }
        }
        else if (pfJetPt < 150) {
          _meHLTGenJetRes[4]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          if (isHEP17(pfJetEta,pfJetPhi)){ 
            _meHLTGenJetResHEP17[4]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          }
          if (isHEM17(pfJetEta,pfJetPhi)){ 
            _meHLTGenJetResHEM17[4]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          }
          if (isHEP18(pfJetEta,pfJetPhi)){ 
            _meHLTGenJetResHEP18[4]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          }
        }
        else if (pfJetPt < 250) {
          _meHLTGenJetRes[5]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          if (isHEP17(pfJetEta,pfJetPhi)){ 
            _meHLTGenJetResHEP17[5]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          }
          if (isHEM17(pfJetEta,pfJetPhi)){ 
            _meHLTGenJetResHEM17[5]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          }
          if (isHEP18(pfJetEta,pfJetPhi)){ 
            _meHLTGenJetResHEP18[5]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          }
        }
        else {
          _meHLTGenJetRes[6]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          if (isHEP17(pfJetEta,pfJetPhi)){ 
            _meHLTGenJetResHEP17[6]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          }
          if (isHEM17(pfJetEta,pfJetPhi)){ 
            _meHLTGenJetResHEM17[6]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          }
          if (isHEP18(pfJetEta,pfJetPhi)){ 
            _meHLTGenJetResHEP18[6]->Fill(pfJetPt/v_genjet_pt[matchedgenJetidx] );
          }
        }
      }
    }//loop over pfjets
    //cout << " test v_hltjet_pt size !!! " << v_hltjet_pt.size() << endl;
  } 
  else{
    //std::cout << "  -- No PFJets" << std::endl;
    //if (evtCnt==1) edm::LogWarning("HLTJetMETValidation") << "  -- No PFJets"; 
  }



  // --- Fill histos for PFMET paths ---
  // HLT MET namely hltmet
  edm::Handle<CaloMETCollection> recmet;
  iEvent.getByToken(CaloMETColl, recmet);

  double calMet=-1;
  if (recmet.isValid()) { 
    typedef CaloMETCollection::const_iterator cmiter;
    //std::cout << "Size of MET collection" <<  recmet.size() << std::endl;
    for ( cmiter i=recmet->begin(); i!=recmet->end(); i++) {
      calMet = i->pt();
      for (size_t it=0;it<hltTrgMet.size();it++) {
	_meHLTMET[it] -> Fill(calMet);
	if (myTrigM.size() > it && myTrigM[it]) _meHLTMETTrgMC[it] -> Fill(calMet);
	if (myTrigM.size() > it && myTrigMLow.size() > it && myTrigM[it] && myTrigMLow[it]) _meHLTMETTrg[it] -> Fill(calMet);
	if (myTrigMLow.size() > it && myTrigMLow[it]) _meHLTMETTrgLow[it] -> Fill(calMet);
      }
    }
  }
  else{
    //std::cout << "  -- No MET Collection with name: " << CaloMETColl << std::endl;
    //if (evtCnt==1) edm::LogWarning("HLTJetMETValidation") << "  -- No MET Collection with name: "<< CaloMETColl; 
  }
  
  edm::Handle<GenMETCollection> genmet;
  iEvent.getByToken(GenMETColl, genmet);

  double genMet=-1;
  if (genmet.isValid()) { 
    typedef GenMETCollection::const_iterator cmiter;
    for ( cmiter i=genmet->begin(); i!=genmet->end(); i++) {
      genMet = i->pt();
      for (size_t it=0;it<hltTrgMet.size();it++) {
	_meGenMET[it] -> Fill(genMet);
	if (myTrigM.size() > it && myTrigM[it]) _meGenMETTrgMC[it] -> Fill(genMet);
	if (myTrigM.size() > it && myTrigMLow.size() > it && myTrigM[it] && myTrigMLow[it]) _meGenMETTrg[it] -> Fill(genMet);
	if (myTrigMLow.size() > it && myTrigMLow[it]) _meGenMETTrgLow[it] -> Fill(genMet);
      }
    }
  }
  else{
    //std::cout << "  -- No GenMET Collection with name: " << GenMETColl << std::endl;
    //if (evtCnt==1) edm::LogWarning("HLTJetMETValidation") << "  -- No GenMET Collection with name: "<< GenMETColl; 
  }
  
}

void HLTJetMETValidation::getHLTResults(const edm::TriggerResults& hltresults,
                                        const edm::TriggerNames & triggerNames) {

  int ntrigs=hltresults.size();
  if (! HLTinit_){
    HLTinit_=true;
    
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
/// For Hcal HEP17 Area
bool HLTJetMETValidation::isHEP17(double eta, double phi){
  bool output = false;
  // phi -0.87 to -0.52 
  if (fabs(eta) <= 3.0 && fabs(eta) > 1.3 && (eta/fabs(eta) > 0) &&
      phi > -0.87 && phi <= -0.52 ) {output=true;}
  return output;
}
/// For Hcal HEM17 Area
bool HLTJetMETValidation::isHEM17(double eta, double phi){
  bool output = false;
  if (fabs(eta) <= 3.0 && fabs(eta) > 1.3 && (eta/fabs(eta) < 0) &&
      phi > -0.87 && phi <= -0.52 ) {output=true;}
  return output;
}
/// For Hcal HEP18 Area
bool HLTJetMETValidation::isHEP18(double eta, double phi){
  bool output = false;
  // phi -0.87 to -0.52 
  if (fabs(eta) <= 3.0 && fabs(eta) > 1.3 && (eta/fabs(eta) > 0) &&
      phi > -0.52 && phi <= -0.17 ) {output=true;}
  return output;
}

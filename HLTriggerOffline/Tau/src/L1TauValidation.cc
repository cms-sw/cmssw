// Original Author:  Chi Nhan Nguyen
//         Created:  Fri Feb 22 09:20:55 CST 2008

#include "HLTriggerOffline/Tau/interface/L1TauValidation.h"
#include "Math/GenVector/VectorUtil.h"
#include <iostream>
#include <iomanip>
#include <fstream>

//
// constructors and destructor
//
L1TauValidation::L1TauValidation(const edm::ParameterSet& iConfig):

  _mcColl(iConfig.getParameter<edm::InputTag>("MatchedCollection")),

  _L1extraTauJetSource(iConfig.getParameter<edm::InputTag>("L1extraTauJetSource")),
  _L1extraCenJetSource(iConfig.getParameter<edm::InputTag>("L1extraCenJetSource")),
  _L1extraForJetSource(iConfig.getParameter<edm::InputTag>("L1extraForJetSource")),
  _L1extraMuonSource(iConfig.getParameter<edm::InputTag>("L1extraMuonSource")),
  _L1extraMETSource(iConfig.getParameter<edm::InputTag>("L1extraMETSource")),
  _L1extraNonIsoEgammaSource(iConfig.getParameter<edm::InputTag>("L1extraNonIsoEgammaSource")),
  _L1extraIsoEgammaSource(iConfig.getParameter<edm::InputTag>("L1extraIsoEgammaSource")),

  _L1GtReadoutRecord(iConfig.getParameter<edm::InputTag>("L1GtReadoutRecord")),
  _L1GtObjectMap(iConfig.getParameter<edm::InputTag>("L1GtObjectMap")),

  _SingleTauThreshold(iConfig.getParameter<double>("SingleTauThreshold")),
  _DoubleTauThreshold(iConfig.getParameter<double>("DoubleTauThreshold")),
  _SingleTauMETThresholds(iConfig.getParameter< std::vector<double> >("SingleTauMETThresholds")),
  _MuTauThresholds(iConfig.getParameter< std::vector<double> >("MuTauThresholds")),
  _IsoEgTauThresholds(iConfig.getParameter< std::vector<double> >("IsoEgTauThresholds")),
  
  _L1SingleTauName(iConfig.getParameter<std::string>("L1SingleTauName")),
  _L1DoubleTauName(iConfig.getParameter<std::string>("L1DoubleTauName")),
  _L1TauMETName(iConfig.getParameter<std::string>("L1TauMETName")),
  _L1MuonTauName(iConfig.getParameter<std::string>("L1MuonTauName")),
  _L1IsoEgTauName(iConfig.getParameter<std::string>("L1IsoEGTauName")),

  _L1MCTauMinDeltaR(iConfig.getParameter<double>("L1MCTauMinDeltaR")),
  _MCTauHadMinEt(iConfig.getParameter<double>("MCTauHadMinEt")),
  _MCTauHadMaxAbsEta(iConfig.getParameter<double>("MCTauHadMaxAbsEta")),
  
  _triggerTag((iConfig.getParameter<std::string>("TriggerTag"))),
  _outFile(iConfig.getParameter<std::string>("OutputFileName"))

{
  DQMStore* store = &*edm::Service<DQMStore>();
  
  if(store)
    {
      //Create the histograms
      store->setCurrentFolder(_triggerTag);
      h_L1TauEt = store->book1D("L1TauEt","L1TauEt",50,0.,100.);
      h_L1TauEt->getTH1F()->Sumw2();  
      h_L1TauEta = store->book1D("L1TauEta","L1TauEta",60,-4.,4.);
      h_L1TauEta->getTH1F()->Sumw2();
      h_L1TauPhi = store->book1D("L1TauPhi","L1TauPhi",50,-3.2,3.2);
      h_L1TauPhi->getTH1F()->Sumw2();
      
      h_L1Tau1Et = store->book1D("L1Tau1Et","L1Tau1Et",50,0.,100.);
      h_L1Tau1Et->getTH1F()->Sumw2();
      h_L1Tau1Eta = store->book1D("L1Tau1Eta","L1Tau1Eta",60,-4.,4.);
      h_L1Tau1Eta->getTH1F()->Sumw2();
      h_L1Tau1Phi = store->book1D("L1Tau1Phi","L1Tau1Phi",50,-3.2,3.2);
      h_L1Tau1Phi->getTH1F()->Sumw2();
      
      h_L1Tau2Et = store->book1D("L1Tau2Et","L1Tau2Et",50,0.,100.);
      h_L1Tau2Et->getTH1F()->Sumw2();
      h_L1Tau2Eta = store->book1D("L1Tau2Eta","L1Tau2Eta",60,-4.,4.);
      h_L1Tau2Eta->getTH1F()->Sumw2();
      h_L1Tau2Phi = store->book1D("L1Tau2Phi","L1Tau2Phi",50,-3.2,3.2);
      h_L1Tau2Phi->getTH1F()->Sumw2();
      

      // L1 response
      h_L1MCTauDeltaR = store->book1D("L1MCTauDeltaR","L1MCTauDeltaR",60,0.,6.);
      h_L1MCTauDeltaR->getTH1F()->Sumw2();
      h_L1minusMCTauEt = store->book1D("L1minusMCTauEt","L1minusMCTauEt",50,-50.,50.);
      h_L1minusMCTauEt->getTH1F()->Sumw2();
      h_L1minusMCoverMCTauEt = store->book1D("L1minusMCoverMCTauEt","L1minusMCoverMCTauEt",40,-1.2,1.2);
      h_L1minusMCoverMCTauEt->getTH1F()->Sumw2();
      
      // MC w/o cuts
      h_GenTauHadEt = store->book1D("GenTauHadEt","GenTauHadEt",50,0.,100.);
      h_GenTauHadEt->getTH1F()->Sumw2();
      h_GenTauHadEta = store->book1D("GenTauHadEta","GenTauHadEt",60,-4.,4.);
      h_GenTauHadEta->getTH1F()->Sumw2();
      h_GenTauHadPhi = store->book1D("GenTauHadPhi","GenTauHadPhi",50,-3.2,3.2);
      h_GenTauHadPhi->getTH1F()->Sumw2();
      
      // MC matching efficiencies
      h_MCTauHadEt = store->book1D("MCTauHadEt","MCTauHadEt",50,0.,100.);
      h_MCTauHadEt->getTH1F()->Sumw2();
      h_MCTauHadEta = store->book1D("MCTauHadEta","MCTauHadEta",60,-4.,4.);
      h_MCTauHadEta->getTH1F()->Sumw2();
      h_MCTauHadPhi = store->book1D("MCTauHadPhi","MCTauHadPhi",50,-3.2,3.2);
      h_MCTauHadPhi->getTH1F()->Sumw2();
      
      h_L1MCMatchedTauEt = store->book1D("L1MCMatchedTauEt","L1MCMatchedTauEt",50,0.,100.);
      h_L1MCMatchedTauEt->getTH1F()->Sumw2();
      h_L1MCMatchedTauEta = store->book1D("L1MCMatchedTauEta","L1MCMatchedTauEta",60,-4.,4.);
      h_L1MCMatchedTauEta->getTH1F()->Sumw2();
      h_L1MCMatchedTauPhi = store->book1D("L1MCMatchedTauPhi","L1MCMatchedTauPhi",50,-3.2,3.2);
      h_L1MCMatchedTauPhi->getTH1F()->Sumw2();
      
      h_EffMCTauEt = store->book1D("EffMCTauEt","EffMCTauEt",50,0.,100.);
      h_EffMCTauEt->getTH1F()->Sumw2();
      h_EffMCTauEta = store->book1D("EffMCTauEta","EffMCTauEta",60,-4.,4.);
      h_EffMCTauEta->getTH1F()->Sumw2();
      h_EffMCTauPhi = store->book1D("EffMCTauPhi","EffMCTauPhi",50,-3.2,3.2);
      h_EffMCTauPhi->getTH1F()->Sumw2();
      
      h_L1SingleTauEffEt = store->book1D("L1SingleTauEffEt","L1SingleTauEffEt",
					 50,0.,100.);
      h_L1SingleTauEffEt->getTH1F()->Sumw2();
      h_L1DoubleTauEffEt = store->book1D("L1DoubleTauEffEt","L1DoubleTauEffEt",
					 40,0.,80.);
      h_L1DoubleTauEffEt->getTH1F()->Sumw2();
      h_L1SingleTauEffMCMatchEt = store->book1D("L1SingleTauEffMCMatchEt","L1SingleTauEffMCMatchEt",
						50,0.,100.);
      h_L1SingleTauEffMCMatchEt->getTH1F()->Sumw2();
      h_L1DoubleTauEffMCMatchEt = store->book1D("L1DoubleTauEffMCMatchEt","L1DoubleTauEffMCMatchEt",
						40,0.,80.);
      h_L1DoubleTauEffMCMatchEt->getTH1F()->Sumw2();
    }
}

// ------------ method called once each job just before starting event loop  ------------
void 
L1TauValidation::beginJob(const edm::EventSetup&)
{
  // Init counters for event based efficiencies
  _nEvents = 0; // all events processed

  _nEventsGenTauHad = 0; 
  _nEventsDoubleGenTauHads = 0; 
  _nEventsGenTauMuonTauHad = 0; 
  _nEventsGenTauElecTauHad = 0;   

  _nfidEventsGenTauHad = 0; 
  _nfidEventsDoubleGenTauHads = 0; 
  _nfidEventsGenTauMuonTauHad = 0; 
  _nfidEventsGenTauElecTauHad = 0;   

  _nEventsPFMatchGenTauHad = 0; 
  _nEventsPFMatchDoubleGenTauHads = 0; 
  _nEventsPFMatchGenTauMuonTauHad = 0; 
  _nEventsPFMatchGenTauElecTauHad = 0;   

  _nEventsL1SingleTauPassed = 0;
  _nEventsL1SingleTauPassedMCMatched = 0;

  _nEventsL1DoubleTauPassed = 0;
  _nEventsL1DoubleTauPassedMCMatched = 0;

  _nEventsL1SingleTauMETPassed = 0;
  _nEventsL1SingleTauMETPassedMCMatched = 0;

  _nEventsL1MuonTauPassed = 0;
  _nEventsL1MuonTauPassedMCMatched = 0;

  _nEventsL1IsoEgTauPassed = 0;
  _nEventsL1IsoEgTauPassedMCMatched = 0;

  // from GT bit info
  _nEventsL1GTSingleTauPassed = 0;
  _nEventsL1GTDoubleTauPassed = 0;
  _nEventsL1GTSingleTauMETPassed = 0;
  _nEventsL1GTMuonTauPassed = 0;
  _nEventsL1GTIsoEgTauPassed = 0;

}

L1TauValidation::~L1TauValidation()
{
}



// ------------ method called to for each event  ------------
void
L1TauValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   _nEvents++;

  // get object
   getL1extraObjects(iEvent);
   evalL1Decisions(iEvent);
   evalL1extraDecisions();
   
   // fill simple histograms
   fillL1Histograms();
   fillL1MCTauMatchedHists(iEvent);

}



// ------------ method called once each job just after ending the event loop  ------------
void 
L1TauValidation::endJob() {


  if (&*edm::Service<DQMStore>()) edm::Service<DQMStore>()->save ("test.root");

  // MC matching efficiencies
  h_EffMCTauEt->getTH1F()->Divide(h_EffMCTauEt->getTH1F(),h_MCTauHadEt->getTH1F(),1.,1.,"b");
  h_EffMCTauEta->getTH1F()->Divide(h_EffMCTauEta->getTH1F(),h_MCTauHadEta->getTH1F(),1.,1.,"b");
  h_EffMCTauPhi->getTH1F()->Divide(h_EffMCTauPhi->getTH1F(),h_MCTauHadPhi->getTH1F(),1.,1.,"b");

  //
  convertToIntegratedEff(h_L1SingleTauEffEt,(double)_nEvents);
  convertToIntegratedEff(h_L1DoubleTauEffEt,(double)_nEvents);
  convertToIntegratedEff(h_L1SingleTauEffMCMatchEt,(double)_nfidEventsGenTauHad);
  convertToIntegratedEff(h_L1DoubleTauEffMCMatchEt,(double)_nfidEventsDoubleGenTauHads);

  //Write file
  if(_outFile.size()>0)
    if (&*edm::Service<DQMStore>()) edm::Service<DQMStore>()->save (_outFile);
}

void
L1TauValidation::getL1extraObjects(const edm::Event& iEvent)
{
  using namespace edm;
  using namespace l1extra;

  //
  _L1Taus.clear();
  Handle<L1JetParticleCollection> l1TauHandle;
  iEvent.getByLabel(_L1extraTauJetSource,l1TauHandle);
  for( L1JetParticleCollection::const_iterator itr = l1TauHandle->begin() ;
       itr != l1TauHandle->end() ; ++itr ) {
    LV p(itr->px(),itr->py(),itr->pz(),itr->energy());
    _L1Taus.push_back(p);
  }

  //
  _L1CenJets.clear();
  Handle<L1JetParticleCollection> l1CenJetHandle;
  iEvent.getByLabel(_L1extraCenJetSource,l1CenJetHandle);
  for( L1JetParticleCollection::const_iterator itr = l1CenJetHandle->begin() ;
       itr != l1CenJetHandle->end() ; ++itr ) {
    LV p(itr->px(),itr->py(),itr->pz(),itr->energy());
    _L1CenJets.push_back(p);
  }

  //
  _L1ForJets.clear();
  Handle<L1JetParticleCollection> l1ForJetHandle;
  iEvent.getByLabel(_L1extraForJetSource,l1ForJetHandle);
  for( L1JetParticleCollection::const_iterator itr = l1ForJetHandle->begin() ;
       itr != l1ForJetHandle->end() ; ++itr ) {
    LV p(itr->px(),itr->py(),itr->pz(),itr->energy());
    _L1ForJets.push_back(p);
  }

  //
  _L1IsoEgammas.clear();
  Handle<L1EmParticleCollection> l1IsoEgammaHandle;
  iEvent.getByLabel(_L1extraIsoEgammaSource,l1IsoEgammaHandle);
  for( L1EmParticleCollection::const_iterator itr = l1IsoEgammaHandle->begin() ;
       itr != l1IsoEgammaHandle->end() ; ++itr ) {
    LV p(itr->px(),itr->py(),itr->pz(),itr->energy());
    _L1IsoEgammas.push_back(p);
  }

  //
  _L1NonIsoEgammas.clear();
  Handle<L1EmParticleCollection> l1NonIsoEgammaHandle;
  iEvent.getByLabel(_L1extraNonIsoEgammaSource,l1NonIsoEgammaHandle);
  for( L1EmParticleCollection::const_iterator itr = l1NonIsoEgammaHandle->begin() ;
       itr != l1NonIsoEgammaHandle->end() ; ++itr ) {
    LV p(itr->px(),itr->py(),itr->pz(),itr->energy());
    _L1NonIsoEgammas.push_back(p);
  }

  //
  _L1Muons.clear();
  _L1MuQuals.clear();
  Handle<L1MuonParticleCollection> l1MuonHandle;
  iEvent.getByLabel(_L1extraMuonSource,l1MuonHandle);
  for( L1MuonParticleCollection::const_iterator itr = l1MuonHandle->begin() ;
       itr != l1MuonHandle->end() ; ++itr ) {
    LV p(itr->px(),itr->py(),itr->pz(),itr->energy());
    _L1Muons.push_back(p);
    L1MuGMTExtendedCand gmtCand = itr->gmtMuonCand();
    _L1MuQuals.push_back(gmtCand.quality());// Muon quality as defined in the GT
  }

  //
  _L1METs.clear();
  Handle<L1EtMissParticleCollection> l1MetHandle;
  iEvent.getByLabel(_L1extraMETSource,l1MetHandle);
  for( L1EtMissParticleCollection::const_iterator itr = l1MetHandle->begin() ;
       itr != l1MetHandle->end() ; ++itr ) {
    LV p(itr->px(),itr->py(),itr->pz(),itr->energy());
    _L1METs.push_back(p);
  }
  /*
  Handle<L1EtMissParticle> l1MetHandle;
  iEvent.getByLabel(_L1extraMETSource,l1MetHandle);
  LV p(l1MetHandle->px(),l1MetHandle->py(),l1MetHandle->pz(),l1MetHandle->energy());
  _L1METs.push_back(p);
  */
  /*
  // Dummy
  LV p(0.,0.,0.,0.);
  _L1METs.push_back(p);
  */
 
}


void
L1TauValidation::fillL1Histograms() {
  for (int i=0; i<(int)_L1Taus.size(); i++) {
    h_L1TauEt->Fill(_L1Taus[i].Et());
    h_L1TauEta->Fill(_L1Taus[i].Eta());
    h_L1TauPhi->Fill(_L1Taus[i].Phi());
    if (i==1) {
      h_L1Tau1Et->Fill(_L1Taus[i].Et());
      h_L1Tau1Eta->Fill(_L1Taus[i].Eta());
      h_L1Tau1Phi->Fill(_L1Taus[i].Phi());
    }
    if (i==2) {
      h_L1Tau2Et->Fill(_L1Taus[i].Et());
      h_L1Tau2Eta->Fill(_L1Taus[i].Eta());
      h_L1Tau2Phi->Fill(_L1Taus[i].Phi());
    }    
  }

}



void
L1TauValidation::evalL1Decisions(const edm::Event& iEvent) {
  using namespace edm;
  using namespace std;

  Handle<L1GlobalTriggerReadoutRecord> l1GtRR;
  iEvent.getByLabel(_L1GtReadoutRecord,l1GtRR);
  Handle<L1GlobalTriggerObjectMapRecord> l1GtOMRec;
  iEvent.getByLabel(_L1GtObjectMap,l1GtOMRec);

  L1GlobalTriggerReadoutRecord L1GTRR = *l1GtRR.product();		
  L1GlobalTriggerObjectMapRecord L1GTOMRec = *l1GtOMRec.product();

  DecisionWord gtDecisionWord = L1GTRR.decisionWord();
  string l1BitName;
  int l1Accept;
  // get ObjectMaps from ObjectMapRecord
  const vector<L1GlobalTriggerObjectMap>& objMapVec =  L1GTOMRec.gtObjectMap();
  for (vector<L1GlobalTriggerObjectMap>::const_iterator itMap = objMapVec.begin();
       itMap != objMapVec.end(); ++itMap) {
    int iBit = (*itMap).algoBitNumber();
    l1BitName = string( (*itMap).algoName() );
    l1Accept = gtDecisionWord[iBit];
    //cout<<l1BitName<<" "<<l1Accept<<endl;
    if (l1BitName.compare(_L1SingleTauName)==0) {
      //cout<<l1BitName<<" "<<l1Accept<<endl;
      if (l1Accept) _nEventsL1GTSingleTauPassed++;
    }
    if (l1BitName.compare(_L1DoubleTauName)==0) {
      if (l1Accept) _nEventsL1GTDoubleTauPassed++;
    }
    if (l1BitName.compare(_L1TauMETName)==0) {
      if (l1Accept) _nEventsL1GTSingleTauMETPassed++;
    }
    if (l1BitName.compare(_L1MuonTauName)==0) {
      if (l1Accept) _nEventsL1GTMuonTauPassed++;
    }
    if (l1BitName.compare(_L1IsoEgTauName)==0) {
      if (l1Accept) _nEventsL1GTIsoEgTauPassed++;
    }
  }
}

void
L1TauValidation::evalL1extraDecisions() {
  bool singleTauPassed = false;
  bool doubleTauPassed = false;
  bool muTauPassed = false;
  bool isoEgTauPassed = false;
  bool singleTauMETPassed = false;
  
  int nL1Taus = _L1Taus.size();
  int nL1Muons = _L1Muons.size();
  int nL1IsoEgammas = _L1IsoEgammas.size();


  if (nL1Taus>=1) {
    h_L1SingleTauEffEt->Fill(_L1Taus[0].Et());
    if (_L1Taus[0].Et()>=_SingleTauThreshold)
      singleTauPassed = true;
  }
  if (nL1Taus>=2 ) {
    h_L1DoubleTauEffEt->Fill(_L1Taus[1].Et());
    if (_L1Taus[1].Et()>=_DoubleTauThreshold)
      doubleTauPassed = true;
  }

  if (nL1Taus>=1 && _L1Taus[0].Et()>=_SingleTauMETThresholds[0] &&
      _L1METs[0].Et()>=_SingleTauMETThresholds[1])
    singleTauMETPassed = true;

  if (nL1Taus>=1 && _L1Taus[0].Et()>=_MuTauThresholds[1] &&
      nL1Muons>=1 && _L1Muons[0].Pt()>=_MuTauThresholds[0]) {
    //if ( _L1MuQuals[0]==4 || _L1MuQuals[0]==5 || _L1MuQuals[0]==6 || _L1MuQuals[0]==7 ) {
    if ( _L1MuQuals[0]==3 || _L1MuQuals[0]==5 || _L1MuQuals[0]==6 || _L1MuQuals[0]==7 ) {
    //if ( _L1MuQuals[0]>=0) {
      muTauPassed = true;
    }
  }
    
  for (int i=0;i<nL1Taus;i++) {
    for (int j=0;j<nL1IsoEgammas;j++) {
      if (_L1Taus[i].Et()>=_IsoEgTauThresholds[1] &&
	  _L1IsoEgammas[j].Et()>=_IsoEgTauThresholds[0]) {
	//double deltaR = ROOT::Math::VectorUtil::DeltaR(_L1Taus[i],_GenTauHads[j]);
	//double deltaPhi = ROOT::Math::VectorUtil::DeltaPhi(_L1Taus[i],_L1IsoEgammas[j]);
	//double deltaEta = std::abs(_L1Taus[i].Eta()-_L1IsoEgammas[j].Eta());
	// Non-collinearity check
	//if (deltaPhi>0.348 && deltaEta>0.348) {
	  isoEgTauPassed = true;
	  break;
	//}
      }
    }
  }

  if (singleTauPassed) _nEventsL1SingleTauPassed++;
  if (doubleTauPassed) _nEventsL1DoubleTauPassed++;
  if (singleTauMETPassed) _nEventsL1SingleTauMETPassed++;
  if (muTauPassed) _nEventsL1MuonTauPassed++;
  if (isoEgTauPassed) _nEventsL1IsoEgTauPassed++;
}



void
L1TauValidation::fillL1MCTauMatchedHists(const edm::Event& iEvent) {
  using namespace edm;

  Handle<LVColl> McInfoH; //Handle To The Truth!!!!
  iEvent.getByLabel(_mcColl,McInfoH);
  LVColl McInfo = *McInfoH;

  bool singleTauPassed = false;
  bool doubleTauPassed = false;
  //bool muTauPassed = false;
  //bool isoEgTauPassed = false;
  bool singleTauMETPassed = false;

  bool singleMatch = false; // for doubletau match
  bool doubleMatch = false; 
  int iSingle = -1;
  int iDouble = -1;

  int nfidTauHads = 0; // count in fiducial region
  for (int i=0; i<(int)McInfo.size(); i++) {
    // w/o further cuts
    h_GenTauHadEt->Fill(McInfo[i].Et());
    h_GenTauHadEta->Fill(McInfo[i].Eta());
    h_GenTauHadPhi->Fill(McInfo[i].Phi());
    // Denominators for MC matching efficiencies
    if (std::abs(McInfo[i].Eta())<=_MCTauHadMaxAbsEta)
      h_MCTauHadEt->Fill(McInfo[i].Et());
    if (McInfo[i].Et()>=_MCTauHadMinEt)
      h_MCTauHadEta->Fill(McInfo[i].Eta());
    if (McInfo[i].Et()>=_MCTauHadMinEt && std::abs(McInfo[i].Eta())<=_MCTauHadMaxAbsEta) {
      h_MCTauHadPhi->Fill(McInfo[i].Phi());
      nfidTauHads++;
    }
  }
  //////// Counters after fiducial cuts!
  if (nfidTauHads >= 1) _nfidEventsGenTauHad++; 
  if (nfidTauHads >= 2) _nfidEventsDoubleGenTauHads++; 

  for (unsigned int i = 0; i<_L1Taus.size();i++) {
    for (unsigned int j = 0; j<McInfo.size();j++) {
      double deltaR = ROOT::Math::VectorUtil::DeltaR(_L1Taus[i],McInfo[j]);
      h_L1MCTauDeltaR->Fill(deltaR);
      if (deltaR < _L1MCTauMinDeltaR) {
	if (McInfo[j].Et()>=_MCTauHadMinEt && std::abs(McInfo[j].Eta())<=_MCTauHadMaxAbsEta) {
	  h_L1minusMCTauEt->Fill(_L1Taus[i].Et() - McInfo[j].Et());
	  h_L1minusMCoverMCTauEt->Fill( (_L1Taus[i].Et() - McInfo[j].Et()) / McInfo[j].Et());
	  // For event efficiencies	
	  if (singleMatch) {
	    doubleMatch = true;	iDouble = i;
	  }
	  singleMatch = true;
	  if (singleMatch && !doubleMatch)
	    iSingle = i;
	}	
	// Numerators for MC matching efficiencies
	if (std::abs(McInfo[j].Eta())<=_MCTauHadMaxAbsEta) {	
	  h_L1MCMatchedTauEt->Fill(McInfo[j].Et());
	  h_EffMCTauEt->Fill(McInfo[j].Et());
	}
	if (McInfo[j].Et()>=_MCTauHadMinEt) {
	  h_L1MCMatchedTauEta->Fill(McInfo[j].Eta());
	  h_EffMCTauEta->Fill(McInfo[j].Eta());
	}
	if (McInfo[j].Et()>=_MCTauHadMinEt && std::abs(McInfo[j].Eta())<=_MCTauHadMaxAbsEta) {
	  h_L1MCMatchedTauPhi->Fill(McInfo[j].Phi());
	  h_EffMCTauPhi->Fill(McInfo[j].Phi());
	}
      }           
    }
  }
  
  // For event efficiencies	
  if (singleMatch && iSingle>=0) {
    h_L1SingleTauEffMCMatchEt->Fill(_L1Taus[iSingle].Et());
    if (_L1Taus[iSingle].Et()>=_SingleTauThreshold)
      singleTauPassed = true;

    if (_L1Taus[iSingle].Et()>=_SingleTauMETThresholds[0] &&
	_L1METs[0].Et()>=_SingleTauMETThresholds[1])
      singleTauMETPassed = true;

  }

  if (doubleMatch && iDouble>=0) {
    h_L1DoubleTauEffMCMatchEt->Fill(_L1Taus[iDouble].Et());
    if (_L1Taus[iDouble].Et()>=_DoubleTauThreshold)
      doubleTauPassed = true;
  }
  
  if (singleTauPassed) _nEventsL1SingleTauPassedMCMatched++;
  if (doubleTauPassed) _nEventsL1DoubleTauPassedMCMatched++;
  if (singleTauMETPassed) _nEventsL1SingleTauMETPassedMCMatched++;

}


void
L1TauValidation::convertToIntegratedEff(MonitorElement* histo, double nGenerated)
{
  // Convert the histogram to efficiency
  // Assuming that the histogram is incremented with weight=1 for each event
  // this function integrates the histogram contents above every bin and stores it
  // in that bin.  The result is plot of integral rate versus threshold plot.
  int nbins = histo->getNbinsX();
  double integral = histo->getBinContent(nbins+1);  // Initialize to overflow
  if (nGenerated<=0)  {
    std::cerr << "***** L1TauValidation::convertToIntegratedEff() Error: nGenerated = " << nGenerated << std::endl;
    //nGenerated=1;
    return;
  }
  for(int i = nbins; i >= 1; i--)
    {
      double thisBin = histo->getBinContent(i);
      integral += thisBin;
      double integralEff;
      double integralError;
      integralEff = (integral / nGenerated);
      histo->setBinContent(i, integralEff);
      // error
      integralError = (sqrt(integral) / nGenerated);
      histo->setBinError(i, integralError);
    }
}

//define this as a plug-in
//DEFINE_FWK_MODULE(L1TauValidation);

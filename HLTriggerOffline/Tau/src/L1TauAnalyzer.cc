// -*- C++ -*-
//
// Package:    L1TauAnalyzer
// Class:      L1TauAnalyzer
// 
/**\class L1TauAnalyzer L1TauAnalyzer.cc UserCode/L1TauAnalyzer/src/L1TauAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chi Nhan Nguyen
//         Created:  Fri Feb 22 09:20:55 CST 2008
// $Id: L1TauAnalyzer.cc,v 1.7 2011/10/31 13:05:55 olzem Exp $
//
//

#include "HLTriggerOffline/Tau/interface/L1TauAnalyzer.h"


//
// constructors and destructor
//
L1TauAnalyzer::L1TauAnalyzer(const edm::ParameterSet& iConfig)

{
  _GenParticleSource = iConfig.getParameter<edm::InputTag>("GenParticleSource");
  _PFTauSource = iConfig.getParameter<edm::InputTag>("PFTauSource");
  _PFTauDiscriminatorSource = iConfig.getParameter<edm::InputTag>("PFTauDiscriminatorSource");

  _L1extraTauJetSource = iConfig.getParameter<edm::InputTag>("L1extraTauJetSource");
  _L1extraCenJetSource = iConfig.getParameter<edm::InputTag>("L1extraCenJetSource");
  _L1extraForJetSource = iConfig.getParameter<edm::InputTag>("L1extraForJetSource");
  _L1extraMuonSource = iConfig.getParameter<edm::InputTag>("L1extraMuonSource");
  _L1extraMETSource = iConfig.getParameter<edm::InputTag>("L1extraMETSource");
  _L1extraNonIsoEgammaSource = iConfig.getParameter<edm::InputTag>("L1extraNonIsoEgammaSource");
  _L1extraIsoEgammaSource = iConfig.getParameter<edm::InputTag>("L1extraIsoEgammaSource");

  _DoMCMatching = iConfig.getParameter<bool>("DoMCMatching");
  _DoPFTauMatching = iConfig.getParameter<bool>("DoPFTauMatching");

  _L1MCTauMinDeltaR = iConfig.getParameter<double>("L1MCTauMinDeltaR");
  _MCTauHadMinEt = iConfig.getParameter<double>("MCTauHadMinEt");
  _MCTauHadMaxAbsEta = iConfig.getParameter<double>("MCTauHadMaxAbsEta");

  _PFMCTauMinDeltaR = iConfig.getParameter<double>("PFMCTauMinDeltaR");
  _PFTauMinEt = iConfig.getParameter<double>("PFTauMinEt");
  _PFTauMaxAbsEta = iConfig.getParameter<double>("PFTauMaxAbsEta");

  _SingleTauThreshold = iConfig.getParameter<double>("SingleTauThreshold");
  _DoubleTauThreshold = iConfig.getParameter<double>("DoubleTauThreshold");
  _SingleTauMETThresholds = iConfig.getParameter< std::vector<double> >("SingleTauMETThresholds");
  _MuTauThresholds = iConfig.getParameter< std::vector<double> >("MuTauThresholds");
  _IsoEgTauThresholds = iConfig.getParameter< std::vector<double> >("IsoEgTauThresholds");

  _L1GtReadoutRecord = iConfig.getParameter<edm::InputTag>("L1GtReadoutRecord");
  _L1GtObjectMap = iConfig.getParameter<edm::InputTag>("L1GtObjectMap");
  
  _L1SingleTauName = iConfig.getParameter<std::string>("L1SingleTauName");
  _L1DoubleTauName = iConfig.getParameter<std::string>("L1DoubleTauName");
  _L1TauMETName = iConfig.getParameter<std::string>("L1TauMETName");
  _L1MuonTauName = iConfig.getParameter<std::string>("L1MuonTauName");
  _L1IsoEgTauName = iConfig.getParameter<std::string>("L1IsoEGTauName");

  _BosonPID = iConfig.getParameter<int>("BosonPID");
}


L1TauAnalyzer::~L1TauAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
L1TauAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   _nEvents++;

  // get object
   getL1extraObjects(iEvent,iSetup);
   if (_DoMCMatching)
     getGenObjects(iEvent,iSetup);
   if (_DoPFTauMatching)
     getPFTauObjects(iEvent,iSetup);

   evalL1Decisions(iEvent);
   evalL1extraDecisions();
   
   // fill simple histograms
   fillL1Histograms();
   if (_DoMCMatching)
     fillGenHistograms();
   if (_DoPFTauMatching)
     fillPFTauHistograms();

   if (_DoMCMatching)
     calcL1MCTauMatching();
   if (_DoMCMatching && _DoPFTauMatching)
     calcL1MCPFTauMatching();

}


// ------------ method called once each job just before starting event loop  ------------
void 
L1TauAnalyzer::beginJob()
{
  edm::Service<TFileService> fs;
  TFileDirectory dir = fs->mkdir("histos");

  //
  h_L1TauEt = dir.make<TH1F>("L1TauEt","L1TauEt",50,0.,100.);
  h_L1TauEt->Sumw2();
  h_L1TauEta = dir.make<TH1F>("L1TauEta","L1TauEt",60,-4.,4.);
  h_L1TauEta->Sumw2();
  h_L1TauPhi = dir.make<TH1F>("L1TauPhi","L1TauPhi",50,-3.2,3.2);
  h_L1TauPhi->Sumw2();

  h_L1Tau1Et = dir.make<TH1F>("L1Tau1Et","L1Tau1Et",50,0.,100.);
  h_L1Tau1Et->Sumw2();
  h_L1Tau1Eta = dir.make<TH1F>("L1Tau1Eta","L1Tau1Et",60,-4.,4.);
  h_L1Tau1Eta->Sumw2();
  h_L1Tau1Phi = dir.make<TH1F>("L1Tau1Phi","L1Tau1Phi",50,-3.2,3.2);
  h_L1Tau1Phi->Sumw2();

  h_L1Tau2Et = dir.make<TH1F>("L1Tau2Et","L1Tau2Et",50,0.,100.);
  h_L1Tau2Et->Sumw2();
  h_L1Tau2Eta = dir.make<TH1F>("L1Tau2Eta","L1Tau2Et",60,-4.,4.);
  h_L1Tau2Eta->Sumw2();
  h_L1Tau2Phi = dir.make<TH1F>("L1Tau2Phi","L1Tau2Phi",50,-3.2,3.2);
  h_L1Tau2Phi->Sumw2();

  //
  h_GenTauHadEt = dir.make<TH1F>("GenTauHadEt","GenTauHadEt",50,0.,100.);
  h_GenTauHadEt->Sumw2();
  h_GenTauHadEta = dir.make<TH1F>("GenTauHadEta","GenTauHadEt",60,-4.,4.);
  h_GenTauHadEta->Sumw2();
  h_GenTauHadPhi = dir.make<TH1F>("GenTauHadPhi","GenTauHadPhi",50,-3.2,3.2);
  h_GenTauHadPhi->Sumw2();

  //
  h_PFTauEt = dir.make<TH1F>("PFTauEt","PFTauEt",50,0.,100.);
  h_PFTauEt->Sumw2();
  h_PFTauEta = dir.make<TH1F>("PFTauEta","PFTauEt",60,-4.,4.);
  h_PFTauEta->Sumw2();
  h_PFTauPhi = dir.make<TH1F>("PFTauPhi","PFTauPhi",50,-3.2,3.2);
  h_PFTauPhi->Sumw2();

  // L1 response
  h_L1MCTauDeltaR = dir.make<TH1F>("L1MCTauDeltaR","L1MCTauDeltaR",60,0.,6.);
  h_L1MCTauDeltaR->Sumw2();
  h_L1minusMCTauEt = dir.make<TH1F>("L1minusMCTauEt","L1minusMCTauEt",50,-50.,50.);
  h_L1minusMCTauEt->Sumw2();
  h_L1minusMCoverMCTauEt = dir.make<TH1F>("L1minusMCoverMCTauEt","L1minusMCoverMCTauEt",40,-1.2,1.2);
  h_L1minusMCoverMCTauEt->Sumw2();
 
  // MC matching efficiencies
  h_MCTauHadEt = dir.make<TH1F>("MCTauHadEt","MCTauHadEt",50,0.,100.);
  h_MCTauHadEt->Sumw2();
  h_MCTauHadEta = dir.make<TH1F>("MCTauHadEta","MCTauHadEt",60,-4.,4.);
  h_MCTauHadEta->Sumw2();
  h_MCTauHadPhi = dir.make<TH1F>("MCTauHadPhi","MCTauHadPhi",50,-3.2,3.2);
  h_MCTauHadPhi->Sumw2();

  h_L1MCMatchedTauEt = dir.make<TH1F>("L1MCMatchedTauEt","L1MCMatchedTauEt",50,0.,100.);
  h_L1MCMatchedTauEt->Sumw2();
  h_L1MCMatchedTauEta = dir.make<TH1F>("L1MCMatchedTauEta","L1MCMatchedTauEt",60,-4.,4.);
  h_L1MCMatchedTauEta->Sumw2();
  h_L1MCMatchedTauPhi = dir.make<TH1F>("L1MCMatchedTauPhi","L1MCMatchedTauPhi",50,-3.2,3.2);
  h_L1MCMatchedTauPhi->Sumw2();

  h_EffMCTauEt = dir.make<TH1F>("EffMCTauEt","EffMCTauEt",50,0.,100.);
  h_EffMCTauEt->Sumw2();
  h_EffMCTauEta = dir.make<TH1F>("EffMCTauEta","EffMCTauEt",60,-4.,4.);
  h_EffMCTauEta->Sumw2();
  h_EffMCTauPhi = dir.make<TH1F>("EffMCTauPhi","EffMCTauPhi",50,-3.2,3.2);
  h_EffMCTauPhi->Sumw2();

  // PFTau-MC matching efficiencies
  h_MCPFTauHadEt = dir.make<TH1F>("MCPFTauHadEt","MCPFTauHadEt",50,0.,100.);
  h_MCPFTauHadEt->Sumw2();
  h_MCPFTauHadEta = dir.make<TH1F>("MCPFTauHadEta","MCPFTauHadEt",60,-4.,4.);
  h_MCPFTauHadEta->Sumw2();
  h_MCPFTauHadPhi = dir.make<TH1F>("MCPFTauHadPhi","MCPFTauHadPhi",50,-3.2,3.2);
  h_MCPFTauHadPhi->Sumw2();

  h_L1MCPFMatchedTauEt = dir.make<TH1F>("L1MCPFMatchedTauEt","L1MCPFMatchedTauEt",50,0.,100.);
  h_L1MCPFMatchedTauEt->Sumw2();
  h_L1MCPFMatchedTauEta = dir.make<TH1F>("L1MCPFMatchedTauEta","L1MCPFMatchedTauEt",60,-4.,4.);
  h_L1MCPFMatchedTauEta->Sumw2();
  h_L1MCPFMatchedTauPhi = dir.make<TH1F>("L1MCPFMatchedTauPhi","L1MCPFMatchedTauPhi",50,-3.2,3.2);
  h_L1MCPFMatchedTauPhi->Sumw2();

  h_EffMCPFTauEt = dir.make<TH1F>("EffMCPFTauEt","EffMCPFTauEt",50,0.,100.);
  h_EffMCPFTauEt->Sumw2();
  h_EffMCPFTauEta = dir.make<TH1F>("EffMCPFTauEta","EffMCPFTauEt",60,-4.,4.);
  h_EffMCPFTauEta->Sumw2();
  h_EffMCPFTauPhi = dir.make<TH1F>("EffMCPFTauPhi","EffMCPFTauPhi",50,-3.2,3.2);
  h_EffMCPFTauPhi->Sumw2();

  h_PFMCTauDeltaR = dir.make<TH1F>("PFMCTauDeltaR","PFMCTauDeltaR",60,0.,6.);
  h_PFMCTauDeltaR->Sumw2();


  h_L1SingleTauEffEt = dir.make<TH1F>("L1SingleTauEffEt","L1SingleTauEffEt",
					     50,0.,100.);
  h_L1SingleTauEffEt->Sumw2();
  h_L1DoubleTauEffEt = dir.make<TH1F>("L1DoubleTauEffEt","L1DoubleTauEffEt",
					     40,0.,80.);
  h_L1DoubleTauEffEt->Sumw2();
  h_L1SingleTauEffMCMatchEt = dir.make<TH1F>("L1SingleTauEffMCMatchEt","L1SingleTauEffMCMatchEt",
					     50,0.,100.);
  h_L1SingleTauEffMCMatchEt->Sumw2();
  h_L1DoubleTauEffMCMatchEt = dir.make<TH1F>("L1DoubleTauEffMCMatchEt","L1DoubleTauEffMCMatchEt",
					     40,0.,80.);
  h_L1DoubleTauEffMCMatchEt->Sumw2();
  h_L1SingleTauEffPFMCMatchEt = dir.make<TH1F>("L1SingleTauEffPFMCMatchEt","L1SingleTauEffPFMCMatchEt",
					     50,0.,100.);
  h_L1SingleTauEffPFMCMatchEt->Sumw2();
  h_L1DoubleTauEffPFMCMatchEt = dir.make<TH1F>("L1DoubleTauEffPFMCMatchEt","L1DoubleTauEffPFMCMatchEt",
					     40,0.,80.);
  h_L1DoubleTauEffPFMCMatchEt->Sumw2();

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
  _nEventsL1SingleTauPassedPFMCMatched = 0;

  _nEventsL1DoubleTauPassed = 0;
  _nEventsL1DoubleTauPassedMCMatched = 0;
  _nEventsL1DoubleTauPassedPFMCMatched = 0;

  _nEventsL1SingleTauMETPassed = 0;
  _nEventsL1SingleTauMETPassedMCMatched = 0;
  _nEventsL1SingleTauMETPassedPFMCMatched = 0;

  _nEventsL1MuonTauPassed = 0;
  _nEventsL1MuonTauPassedMCMatched = 0;
  _nEventsL1MuonTauPassedPFMCMatched = 0;

  _nEventsL1IsoEgTauPassed = 0;
  _nEventsL1IsoEgTauPassedMCMatched = 0;
  _nEventsL1IsoEgTauPassedPFMCMatched = 0;

  // from GT bit info
  _nEventsL1GTSingleTauPassed = 0;
  _nEventsL1GTDoubleTauPassed = 0;
  _nEventsL1GTSingleTauMETPassed = 0;
  _nEventsL1GTMuonTauPassed = 0;
  _nEventsL1GTIsoEgTauPassed = 0;

}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1TauAnalyzer::endJob() {

  // MC matching efficiencies
  h_EffMCTauEt->Divide(h_MCTauHadEt);
  h_EffMCTauEta->Divide(h_MCTauHadEta);
  h_EffMCTauPhi->Divide(h_MCTauHadPhi);

  // MC-PFTau matching efficiencies
  h_EffMCPFTauEt->Divide(h_MCPFTauHadEt);
  h_EffMCPFTauEta->Divide(h_MCPFTauHadEta);
  h_EffMCPFTauPhi->Divide(h_MCPFTauHadPhi);

  //
  convertToIntegratedEff(h_L1SingleTauEffEt,(double)_nEvents);
  convertToIntegratedEff(h_L1DoubleTauEffEt,(double)_nEvents);
  convertToIntegratedEff(h_L1SingleTauEffMCMatchEt,(double)_nfidEventsGenTauHad);
  convertToIntegratedEff(h_L1DoubleTauEffMCMatchEt,(double)_nfidEventsDoubleGenTauHads);
  convertToIntegratedEff(h_L1SingleTauEffPFMCMatchEt,(double)_nEventsPFMatchGenTauHad);
  convertToIntegratedEff(h_L1DoubleTauEffPFMCMatchEt,(double)_nEventsPFMatchDoubleGenTauHads);

  //printTrigReport();
}

void
L1TauAnalyzer::getL1extraObjects(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace l1extra;

  //
  _L1Taus.clear();
  Handle<L1JetParticleCollection> l1TauHandle;
  iEvent.getByLabel(_L1extraTauJetSource,l1TauHandle);
  for( L1JetParticleCollection::const_iterator itr = l1TauHandle->begin() ;
       itr != l1TauHandle->end() ; ++itr ) {
    TLorentzVector p(itr->px(),itr->py(),itr->pz(),itr->energy());
    _L1Taus.push_back(p);
  }

  //
  _L1CenJets.clear();
  Handle<L1JetParticleCollection> l1CenJetHandle;
  iEvent.getByLabel(_L1extraCenJetSource,l1CenJetHandle);
  for( L1JetParticleCollection::const_iterator itr = l1CenJetHandle->begin() ;
       itr != l1CenJetHandle->end() ; ++itr ) {
    TLorentzVector p(itr->px(),itr->py(),itr->pz(),itr->energy());
    _L1CenJets.push_back(p);
  }

  //
  _L1ForJets.clear();
  Handle<L1JetParticleCollection> l1ForJetHandle;
  iEvent.getByLabel(_L1extraForJetSource,l1ForJetHandle);
  for( L1JetParticleCollection::const_iterator itr = l1ForJetHandle->begin() ;
       itr != l1ForJetHandle->end() ; ++itr ) {
    TLorentzVector p(itr->px(),itr->py(),itr->pz(),itr->energy());
    _L1ForJets.push_back(p);
  }

  //
  _L1IsoEgammas.clear();
  Handle<L1EmParticleCollection> l1IsoEgammaHandle;
  iEvent.getByLabel(_L1extraIsoEgammaSource,l1IsoEgammaHandle);
  for( L1EmParticleCollection::const_iterator itr = l1IsoEgammaHandle->begin() ;
       itr != l1IsoEgammaHandle->end() ; ++itr ) {
    TLorentzVector p(itr->px(),itr->py(),itr->pz(),itr->energy());
    _L1IsoEgammas.push_back(p);
  }

  //
  _L1NonIsoEgammas.clear();
  Handle<L1EmParticleCollection> l1NonIsoEgammaHandle;
  iEvent.getByLabel(_L1extraNonIsoEgammaSource,l1NonIsoEgammaHandle);
  for( L1EmParticleCollection::const_iterator itr = l1NonIsoEgammaHandle->begin() ;
       itr != l1NonIsoEgammaHandle->end() ; ++itr ) {
    TLorentzVector p(itr->px(),itr->py(),itr->pz(),itr->energy());
    _L1NonIsoEgammas.push_back(p);
  }

  //
  _L1Muons.clear();
  _L1MuQuals.clear();
  Handle<L1MuonParticleCollection> l1MuonHandle;
  iEvent.getByLabel(_L1extraMuonSource,l1MuonHandle);
  for( L1MuonParticleCollection::const_iterator itr = l1MuonHandle->begin() ;
       itr != l1MuonHandle->end() ; ++itr ) {
    TLorentzVector p(itr->px(),itr->py(),itr->pz(),itr->energy());
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
    TLorentzVector p(itr->px(),itr->py(),itr->pz(),itr->energy());
    _L1METs.push_back(p);
  }

}

void
L1TauAnalyzer::getPFTauObjects(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace reco;

  _PFTaus.clear();
  Handle<PFTauCollection> thePFTauHandle; 
  iEvent.getByLabel(_PFTauSource,thePFTauHandle); 
  Handle<PFTauDiscriminatorByIsolation> thePFTauDiscriminatorByIsolation; 
  iEvent.getByLabel(_PFTauDiscriminatorSource,thePFTauDiscriminatorByIsolation); 
  for (PFTauCollection::size_type iPFTau=0;iPFTau<thePFTauHandle->size();iPFTau++) { 
    PFTauRef thePFTau(thePFTauHandle,iPFTau); 
    if ((*thePFTauDiscriminatorByIsolation)[thePFTau] == 1) {
      TLorentzVector pftau((*thePFTau).px(),(*thePFTau).py(),(*thePFTau).pz(),(*thePFTau).energy());
      _PFTaus.push_back(pftau);
    }
  }
}

void
L1TauAnalyzer::getGenObjects(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace HepMC;

  ////////////////////////////////////////////////////////  
  // MC Truth based on RecoTauTag/HLTAnalyzers/src/TauJetMCFilter.cc
  Handle<HepMCProduct> evt;
  iEvent.getByLabel(_GenParticleSource, evt);
  GenEvent * generated_event = new GenEvent(*(evt->GetEvent()));

  //init
  _GenTauHads.clear();
  _GenTauMuons.clear();
  _GenTauElecs.clear();

  unsigned int nTauHads = 0; int nTauMuons = 0;int nTauElecs = 0; //
  unsigned int nfidTauHads = 0; int nfidTauMuons = 0;int nfidTauElecs = 0; // count in fiducial region
  TLorentzVector taunu,tauelec,taumuon;
  GenEvent::particle_iterator p;

  for (p = generated_event->particles_begin(); p != generated_event->particles_end(); ++p) {
    if(abs((*p)->pdg_id()) == _BosonPID && (*p)->end_vertex()) {
      HepMC::GenVertex::particle_iterator z = (*p)->end_vertex()->particles_begin(HepMC::descendants);
      for(; z != (*p)->end_vertex()->particles_end(HepMC::descendants); z++) {
	if(abs((*z)->pdg_id()) == 15 && (*z)->status()==2) { 
	  bool lept_decay = false;
	  TLorentzVector tau((*z)->momentum().px(),(*z)->momentum().py(),(*z)->momentum().pz(),(*z)->momentum().e());
	  HepMC::GenVertex::particle_iterator t = (*z)->end_vertex()->particles_begin(HepMC::descendants);
	  for(; t != (*z)->end_vertex()->particles_end(HepMC::descendants); t++) {
	    if(abs((*t)->pdg_id()) == 11 || abs((*t)->pdg_id()) == 13) lept_decay=true;
	    if(abs((*t)->pdg_id()) == 11) {
	      tauelec.SetPxPyPzE((*t)->momentum().px(),(*t)->momentum().py(),(*t)->momentum().pz(),(*t)->momentum().e());
	      _GenTauElecs.push_back(tauelec);
	      nTauElecs++;
	      if (tauelec.Et()>=_MCTauHadMinEt && tauelec.Eta()<=_MCTauHadMaxAbsEta )
		nfidTauElecs++;
	    }
	    if(abs((*t)->pdg_id()) == 13) {
	      taumuon.SetPxPyPzE((*t)->momentum().px(),(*t)->momentum().py(),(*t)->momentum().pz(),(*t)->momentum().e());
	      _GenTauMuons.push_back(taumuon);
	      nTauMuons++;
	      if (taumuon.Et()>=_MCTauHadMinEt && taumuon.Eta()<=_MCTauHadMaxAbsEta )
		nfidTauMuons++;
	    }
	    if(abs((*t)->pdg_id()) == 16)
	      taunu.SetPxPyPzE((*t)->momentum().px(),(*t)->momentum().py(),(*t)->momentum().pz(),(*t)->momentum().e());	
	  }
	  if(lept_decay==false) {
	    TLorentzVector jetMom=tau-taunu;
	    _GenTauHads.push_back(jetMom);
	    nTauHads++;
	    if (jetMom.Et()>=_MCTauHadMinEt && jetMom.Eta()<=_MCTauHadMaxAbsEta )
	      nfidTauHads++;
	  }
	}    
      }
    }
  }
  delete generated_event;

  // Counters
  if (nTauHads >= 1) _nEventsGenTauHad++; 
  if (nTauHads >= 2) _nEventsDoubleGenTauHads++; 
  if (nTauHads >= 1 && nTauMuons >= 1) _nEventsGenTauMuonTauHad++; 
  if (nTauHads >= 1 && nTauElecs >= 1) _nEventsGenTauElecTauHad++; 

  //////// Counters after fiducial cuts!
  if (nfidTauHads >= 1) _nfidEventsGenTauHad++; 
  if (nfidTauHads >= 2) _nfidEventsDoubleGenTauHads++; 
  if (nfidTauHads >= 1 && nfidTauMuons >= 1) _nfidEventsGenTauMuonTauHad++; 
  if (nfidTauHads >= 1 && nfidTauElecs >= 1) _nfidEventsGenTauElecTauHad++; 

}


void
L1TauAnalyzer::fillL1Histograms() {
  for (int i=0; i<(int)_L1Taus.size(); i++) {
    h_L1TauEt->Fill(_L1Taus[i].Et());
    h_L1TauEta->Fill(_L1Taus[i].Eta());
    h_L1TauPhi->Fill(_L1Taus[i].Phi());
    if (i==0) {
      h_L1Tau1Et->Fill(_L1Taus[i].Et());
      h_L1Tau1Eta->Fill(_L1Taus[i].Eta());
      h_L1Tau1Phi->Fill(_L1Taus[i].Phi());
    }
    if (i==1) {
      h_L1Tau2Et->Fill(_L1Taus[i].Et());
      h_L1Tau2Eta->Fill(_L1Taus[i].Eta());
      h_L1Tau2Phi->Fill(_L1Taus[i].Phi());
    }    
  }

}

void
L1TauAnalyzer::fillGenHistograms() {
  for (int i=0; i<(int)_GenTauHads.size(); i++) {
    h_GenTauHadEt->Fill(_GenTauHads[i].Et());
    h_GenTauHadEta->Fill(_GenTauHads[i].Eta());
    h_GenTauHadPhi->Fill(_GenTauHads[i].Phi());
  }  
  // Denominators for MC matching efficiencies
  for (int i=0; i<(int)_GenTauHads.size(); i++) {
    if (std::abs(_GenTauHads[i].Eta())<=_MCTauHadMaxAbsEta)
      h_MCTauHadEt->Fill(_GenTauHads[i].Et());
    if (_GenTauHads[i].Et()>=_MCTauHadMinEt)
      h_MCTauHadEta->Fill(_GenTauHads[i].Eta());
    if (_GenTauHads[i].Et()>=_MCTauHadMinEt && std::abs(_GenTauHads[i].Eta())<=_MCTauHadMaxAbsEta)
      h_MCTauHadPhi->Fill(_GenTauHads[i].Phi());
  }
}

void
L1TauAnalyzer::fillPFTauHistograms() {
  for (int i=0; i<(int)_PFTaus.size(); i++) {
    h_PFTauEt->Fill(_PFTaus[i].Et());
    h_PFTauEta->Fill(_PFTaus[i].Eta());
    h_PFTauPhi->Fill(_PFTaus[i].Phi());
  }
}



void
L1TauAnalyzer::evalL1Decisions(const edm::Event& iEvent) {
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
L1TauAnalyzer::evalL1extraDecisions() {
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
    //if ( _L1MuQuals[0]==3 || _L1MuQuals[0]==5 || _L1MuQuals[0]==6 || _L1MuQuals[0]==7 ) {
    if ( _L1MuQuals[0]>=0) {
      muTauPassed = true;
    }
  }
    
  for (int i=0;i<nL1Taus;i++) {
    for (int j=0;j<nL1IsoEgammas;j++) {
      if (_L1Taus[i].Et()>=_IsoEgTauThresholds[1] &&
	  _L1IsoEgammas[j].Et()>=_IsoEgTauThresholds[0]) {
	//double deltaR = ROOT::Math::VectorUtil::DeltaR(_L1Taus[i],_GenTauHads[j]);
	double deltaPhi = ROOT::Math::VectorUtil::DeltaPhi(_L1Taus[i],_L1IsoEgammas[j]);
	double deltaEta = std::abs(_L1Taus[i].Eta()-_L1IsoEgammas[j].Eta());
	// Non-collinearity check
	if (deltaPhi>0.348 && deltaEta>0.348) {
	  isoEgTauPassed = true;
	  break;
	}
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
L1TauAnalyzer::calcL1MCTauMatching() {
  bool singleTauPassed = false;
  bool doubleTauPassed = false;
  bool muTauPassed = false;
  bool isoEgTauPassed = false;
  bool singleTauMETPassed = false;

  bool singleMatch = false; // for doubletau match
  bool doubleMatch = false; 
  int iSingle = -1;
  int iDouble = -1;

  for (unsigned int i = 0; i<_L1Taus.size();i++) {
    for (unsigned int j = 0; j<_GenTauHads.size();j++) {
      double deltaR = ROOT::Math::VectorUtil::DeltaR(_L1Taus[i],_GenTauHads[j]);
      h_L1MCTauDeltaR->Fill(deltaR);
      if (deltaR < _L1MCTauMinDeltaR) {
	if (_GenTauHads[j].Et()>=_MCTauHadMinEt) {
	  //if (std::abs(_GenTauHads[j].Eta())<=_MCTauHadMaxAbsEta) {
	  h_L1minusMCTauEt->Fill(_L1Taus[i].Et() - _GenTauHads[j].Et());
	  h_L1minusMCoverMCTauEt->Fill( (_L1Taus[i].Et() - _GenTauHads[j].Et()) / _GenTauHads[j].Et());
	  // For event efficiencies	
	  if (singleMatch) {
	    doubleMatch = true;
	    iDouble = i;
	  }
	  singleMatch = true;
	  if (singleMatch && !doubleMatch)
	    iSingle = i;
	}	
	// Numerators for MC matching efficiencies
	if (std::abs(_GenTauHads[j].Eta())<=_MCTauHadMaxAbsEta) {	
	  h_L1MCMatchedTauEt->Fill(_GenTauHads[j].Et());
	  h_EffMCTauEt->Fill(_GenTauHads[j].Et());
	}
	if (_GenTauHads[j].Et()>=_MCTauHadMinEt) {
	  h_L1MCMatchedTauEta->Fill(_GenTauHads[j].Eta());
	  h_EffMCTauEta->Fill(_GenTauHads[j].Eta());
	}
	if (_GenTauHads[j].Et()>=_MCTauHadMinEt && std::abs(_GenTauHads[j].Eta())<=_MCTauHadMaxAbsEta) {
	  h_L1MCMatchedTauPhi->Fill(_GenTauHads[j].Phi());
	  h_EffMCTauPhi->Fill(_GenTauHads[j].Phi());
	}
	//break;
      }           
    }
  }
  if (singleMatch && iSingle>=0) {
    h_L1SingleTauEffMCMatchEt->Fill(_L1Taus[iSingle].Et());
    if (_L1Taus[iSingle].Et()>=_SingleTauThreshold)
      singleTauPassed = true;

    /*
    // Ask for only one L1Tau to be matched with PFTau!!!
    if (_L1Taus.size()>=2) {
      h_L1DoubleTauEffPFMCMatchEt->Fill(_L1Taus[1].Et());
      if (_L1Taus[1].Et()>=_DoubleTauThreshold)
	doubleTauPassed = true;
    }
    */
    
    if (_L1Taus[iSingle].Et()>=_SingleTauMETThresholds[0] &&
	_L1METs[0].Et()>=_SingleTauMETThresholds[1])
      singleTauMETPassed = true;
    
    if (_L1Taus[iSingle].Et()>=_MuTauThresholds[1]) {
      for (int i=0;i<(int)_L1Muons.size();i++) {
	if (_L1Muons[i].Pt()>=_MuTauThresholds[0]) {
	  //if ( _L1MuQuals[0]==4 || _L1MuQuals[0]==5 || _L1MuQuals[0]==6 || _L1MuQuals[0]==7 ) {
	  //if ( _L1MuQuals[0]==3 || _L1MuQuals[0]==5 || _L1MuQuals[0]==6 || _L1MuQuals[0]==7 ) {
	  if ( _L1MuQuals[i]>=0) {
	    for (int j=0;j<(int)_GenTauMuons.size();j++) {
	      double deltaR = ROOT::Math::VectorUtil::DeltaR(_GenTauMuons[j],_L1Muons[i]);
	      if (deltaR<0.3) {
		muTauPassed = true;
	      }
	    }
	  }
	}
      }
    }
    
    for (int j=0;j<(int)_L1IsoEgammas.size();j++) {
      if (_L1Taus[iSingle].Et()>=_IsoEgTauThresholds[1] &&
	  _L1IsoEgammas[j].Et()>=_IsoEgTauThresholds[0]) {
	double deltaPhi = ROOT::Math::VectorUtil::DeltaPhi(_L1Taus[iSingle],_L1IsoEgammas[j]);
	double deltaEta = std::abs(_L1Taus[iSingle].Eta()-_L1IsoEgammas[j].Eta());
	// Non-collinearity check
	if (deltaPhi>0.348 && deltaEta>0.348) {
	  for (int k=0;k<(int)_GenTauElecs.size();k++) {
	    double deltaR = ROOT::Math::VectorUtil::DeltaR(_GenTauElecs[k],_L1IsoEgammas[j]);
	    if (deltaR<0.3) {
	      isoEgTauPassed = true;
	      break;
	    }
	  }
	}
      }
    }
  }

  if (doubleMatch && iDouble>=0) {
    h_L1DoubleTauEffMCMatchEt->Fill(_L1Taus[iDouble].Et());
    if (_L1Taus[iDouble].Et()>=_DoubleTauThreshold)
      doubleTauPassed = true;
    for (int j=0;j<(int)_L1IsoEgammas.size();j++) {
      if (_L1Taus[iDouble].Et()>=_IsoEgTauThresholds[1] &&
	  _L1IsoEgammas[j].Et()>=_IsoEgTauThresholds[0]) {
	double deltaPhi = ROOT::Math::VectorUtil::DeltaPhi(_L1Taus[iDouble],_L1IsoEgammas[j]);
	double deltaEta = std::abs(_L1Taus[iDouble].Eta()-_L1IsoEgammas[j].Eta());
	// Non-collinearity check
	if (deltaPhi>0.348 && deltaEta>0.348) {
	  for (int k=0;k<(int)_GenTauElecs.size();k++) {
	    double deltaR = ROOT::Math::VectorUtil::DeltaR(_GenTauElecs[k],_L1IsoEgammas[j]);
	    if (deltaR<0.3) {
	      isoEgTauPassed = true;
	      break;
	    }
	  }
	}
      }
    }
  }
  
  if (singleTauPassed) _nEventsL1SingleTauPassedMCMatched++;
  if (doubleTauPassed) _nEventsL1DoubleTauPassedMCMatched++;
  if (singleTauMETPassed) _nEventsL1SingleTauMETPassedMCMatched++;
  if (muTauPassed) _nEventsL1MuonTauPassedMCMatched++;
  if (isoEgTauPassed) _nEventsL1IsoEgTauPassedMCMatched++;

}

void
L1TauAnalyzer::calcL1MCPFTauMatching() {
  bool singleTauPassed = false;
  bool doubleTauPassed = false;
  bool muTauPassed = false;
  bool isoEgTauPassed = false;
  bool singleTauMETPassed = false;

  bool singleMatch = false; // for doubletau match
  bool doubleMatch = false; 
  int iSingle = -1;
  int iDouble = -1;
  
  unsigned int nPFMatchGenTauHad = 0;
  
  std::vector<TLorentzVector> PFMatchedGenTauHads;
  PFMatchedGenTauHads.clear();// store PFTau matched gentaus
  for (unsigned int j = 0; j<_GenTauHads.size();j++) {
    for (unsigned int k = 0; k<_PFTaus.size();k++) {
      double deltaR = ROOT::Math::VectorUtil::DeltaR(_PFTaus[k],_GenTauHads[j]);
      h_PFMCTauDeltaR->Fill(deltaR);      
      if (_PFTaus[k].Et()>=_PFTauMinEt && _PFTaus[k].Eta()<=_PFTauMaxAbsEta) {
	if (deltaR < _PFMCTauMinDeltaR) {
	  // Denominators for PF-MC matching efficiencies
	  if (std::abs(_GenTauHads[j].Eta())<=_MCTauHadMaxAbsEta)
	    h_MCPFTauHadEt->Fill(_GenTauHads[j].Et());
	  if (_GenTauHads[j].Et()>=_MCTauHadMinEt)
	    h_MCPFTauHadEta->Fill(_GenTauHads[j].Eta());
	  if (_GenTauHads[j].Et()>=_MCTauHadMinEt &&
	      std::abs(_GenTauHads[j].Eta())<=_MCTauHadMaxAbsEta) {
	    h_MCPFTauHadPhi->Fill(_GenTauHads[j].Phi());
	    nPFMatchGenTauHad++; // For denominator
	    PFMatchedGenTauHads.push_back(_GenTauHads[j]);// store PFTau matched gentaus
	  }
	  break;
	}
      }
    }
  }
  // now loop over only PFTau matched gentaus
  for (unsigned int i = 0; i<_L1Taus.size();i++) {
    for (unsigned int j = 0; j<PFMatchedGenTauHads.size();j++) {
      double deltaR = ROOT::Math::VectorUtil::DeltaR(_L1Taus[i],PFMatchedGenTauHads[j]);
      if (deltaR < _L1MCTauMinDeltaR) {
	// For event efficiencies	
	if (singleMatch) {
	  doubleMatch = true;
	  iDouble = i;
	}
	singleMatch = true;
	if (singleMatch && !doubleMatch)
	  iSingle = i;
	
	// Numerators for PF-MC matching efficiencies
	h_L1MCPFMatchedTauEt->Fill(PFMatchedGenTauHads[j].Et());
	h_EffMCPFTauEt->Fill(PFMatchedGenTauHads[j].Et());
	h_L1MCPFMatchedTauEta->Fill(PFMatchedGenTauHads[j].Eta());
	h_EffMCPFTauEta->Fill(PFMatchedGenTauHads[j].Eta());
	h_L1MCPFMatchedTauPhi->Fill(PFMatchedGenTauHads[j].Phi());
	h_EffMCPFTauPhi->Fill(PFMatchedGenTauHads[j].Phi());
      }
      //break;
    }
  }

  if (singleMatch && iSingle>=0) {
    h_L1SingleTauEffPFMCMatchEt->Fill(_L1Taus[iSingle].Et());
    if (_L1Taus[iSingle].Et()>=_SingleTauThreshold)
      singleTauPassed = true;

    /*
    // Ask for only one L1Tau to be matched with PFTau!!!
    if (_L1Taus.size()>=2) {
      h_L1DoubleTauEffPFMCMatchEt->Fill(_L1Taus[1].Et());
      if (_L1Taus[1].Et()>=_DoubleTauThreshold)
	doubleTauPassed = true;
    }
    */
    
    if (_L1Taus[iSingle].Et()>=_SingleTauMETThresholds[0] &&
	_L1METs[0].Et()>=_SingleTauMETThresholds[1])
      singleTauMETPassed = true;

    if (_L1Taus[iSingle].Et()>=_MuTauThresholds[1]) {
      for (int i=0;i<(int)_L1Muons.size();i++) {
	if (_L1Muons[i].Pt()>=_MuTauThresholds[0]) {
	  //if ( _L1MuQuals[0]==4 || _L1MuQuals[0]==5 || _L1MuQuals[0]==6 || _L1MuQuals[0]==7 ) {
	  //if ( _L1MuQuals[0]==3 || _L1MuQuals[0]==5 || _L1MuQuals[0]==6 || _L1MuQuals[0]==7 ) {
	  if ( _L1MuQuals[i]>=0) {
	    for (int j=0;j<(int)_GenTauMuons.size();j++) {
	      double deltaR = ROOT::Math::VectorUtil::DeltaR(_GenTauMuons[j],_L1Muons[i]);
	      if (deltaR<0.3) {
		muTauPassed = true;
	      }
	    }
	  }
	}
      }
    }
    for (int j=0;j<(int)_L1IsoEgammas.size();j++) {
      if (_L1Taus[iSingle].Et()>=_IsoEgTauThresholds[1] &&
	  _L1IsoEgammas[j].Et()>=_IsoEgTauThresholds[0]) {
	double deltaPhi = ROOT::Math::VectorUtil::DeltaPhi(_L1Taus[iSingle],_L1IsoEgammas[j]);
	double deltaEta = std::abs(_L1Taus[iSingle].Eta()-_L1IsoEgammas[j].Eta());
	// Non-collinearity check
	if (deltaPhi>0.348 && deltaEta>0.348) {
	  for (int k=0;k<(int)_GenTauElecs.size();k++) {
	    double deltaR = ROOT::Math::VectorUtil::DeltaR(_GenTauElecs[k],_L1IsoEgammas[j]);
	    if (deltaR<0.3) {
	      isoEgTauPassed = true;
	      break;
	    }
	  }
	}
      }
    }
  }
  if (doubleMatch && iDouble>=0) {
    h_L1DoubleTauEffPFMCMatchEt->Fill(_L1Taus[iDouble].Et());
    if (_L1Taus[iDouble].Et()>=_DoubleTauThreshold)
      doubleTauPassed = true;

    for (int j=0;j<(int)_L1IsoEgammas.size();j++) {
      if (_L1Taus[iDouble].Et()>=_IsoEgTauThresholds[1] &&
	  _L1IsoEgammas[j].Et()>=_IsoEgTauThresholds[0]) {
	double deltaPhi = ROOT::Math::VectorUtil::DeltaPhi(_L1Taus[iDouble],_L1IsoEgammas[j]);
	double deltaEta = std::abs(_L1Taus[iDouble].Eta()-_L1IsoEgammas[j].Eta());
	// Non-collinearity check
	if (deltaPhi>0.348 && deltaEta>0.348) {
	  for (int k=0;k<(int)_GenTauElecs.size();k++) {
	    double deltaR = ROOT::Math::VectorUtil::DeltaR(_GenTauElecs[k],_L1IsoEgammas[j]);
	    if (deltaR<0.3) {
	      isoEgTauPassed = true;
	      break;
	    }
	  }
	}
      }
    }
  }
  
  unsigned int nfidMCGenTauMuon = 0;
  unsigned int nfidMCGenTauElec = 0;
  for (int i = 0; i<(int)_GenTauMuons.size();i++) {
    if (_GenTauMuons[i].Et()>=_MCTauHadMinEt && _GenTauMuons[i].Eta()<=_MCTauHadMaxAbsEta ) {
      nfidMCGenTauMuon++;
    }
  }
  for (int i = 0; i<(int)_GenTauElecs.size();i++) {
    if (_GenTauElecs[i].Et()>=_MCTauHadMinEt && _GenTauElecs[i].Eta()<=_MCTauHadMaxAbsEta ) {
      nfidMCGenTauElec++;
    }
  }
 
  if (nPFMatchGenTauHad>=1) _nEventsPFMatchGenTauHad++; 
  if (nPFMatchGenTauHad>=2) _nEventsPFMatchDoubleGenTauHads++; 

  if (nPFMatchGenTauHad>=1 && nfidMCGenTauMuon>=1) _nEventsPFMatchGenTauMuonTauHad++; 
  if (nPFMatchGenTauHad>=1 && nfidMCGenTauElec>=1) _nEventsPFMatchGenTauElecTauHad++; 

  if (singleTauPassed) _nEventsL1SingleTauPassedPFMCMatched++;
  if (doubleTauPassed) _nEventsL1DoubleTauPassedPFMCMatched++;
  if (singleTauMETPassed) _nEventsL1SingleTauMETPassedPFMCMatched++;
  if (muTauPassed) _nEventsL1MuonTauPassedPFMCMatched++;
  if (isoEgTauPassed) _nEventsL1IsoEgTauPassedPFMCMatched++;

}

void
L1TauAnalyzer::convertToIntegratedEff(TH1* histo, double nGenerated)
{
  // Convert the histogram to efficiency
  // Assuming that the histogram is incremented with weight=1 for each event
  // this function integrates the histogram contents above every bin and stores it
  // in that bin.  The result is plot of integral rate versus threshold plot.
  int nbins = histo->GetNbinsX();
  double integral = histo->GetBinContent(nbins+1);  // Initialize to overflow
  if (nGenerated<=0)  {
    std::cerr << "***** L1TauAnalyzer::convertToIntegratedEff() Error: nGenerated = " << nGenerated << std::endl;
    nGenerated=1;
  }
  for(int i = nbins; i >= 1; i--)
    {
      double thisBin = histo->GetBinContent(i);
      integral += thisBin;
      double integralEff;
      double integralError;
      integralEff = (integral / nGenerated);
      histo->SetBinContent(i, integralEff);
      // error
      integralError = (sqrt(integral) / nGenerated);
      histo->SetBinError(i, integralError);
    }
}

void
L1TauAnalyzer::printTrigReport() {
  using namespace std;
   
  cout<<"****************************************"<<endl;    
  cout<<"* L1extra Efficiency Report"<<endl;    
  cout<<"****************************************"<<endl;    
  cout<<"Total number of Events: "<<_nEvents<<endl;    
  cout<<"---------------------------------------------------------------------------------------------"<<endl;
  cout<<"                 #PassL1  GlobEff  GlEff/BR(fid.)  MCMatchEff  PFMCMatchEff  BR(fid.)    BR"<<endl;    
  cout<<"---------------------------------------------------------------------------------------------"<<endl;
  cout<<"  SingleTau ("<<_SingleTauThreshold<<"): "
      <<setw(6)<<_nEventsL1SingleTauPassed<<"  "
      <<setprecision(2)<<setw(7)<<(double)_nEventsL1SingleTauPassed/(double)_nEvents<<"  "
      <<setprecision(2)<<setw(13)<<(double)_nEventsL1SingleTauPassed/(double)_nfidEventsGenTauHad<<"  "
      <<setprecision(2)<<setw(10)<<(double)_nEventsL1SingleTauPassedMCMatched/(double)_nfidEventsGenTauHad<<"  "
      <<setprecision(2)<<setw(12)<<(double)_nEventsL1SingleTauPassedPFMCMatched/(double)_nEventsPFMatchGenTauHad<<"  "
      <<setprecision(2)<<setw(8)<<(double)_nfidEventsGenTauHad/(double)_nEvents<<"  "
      <<setprecision(2)<<setw(7)<<(double)_nEventsGenTauHad/(double)_nEvents<<"  "
      <<endl;
  cout<<"  DoubleTau ("<<_DoubleTauThreshold<<"): "
      <<setw(6)<<_nEventsL1DoubleTauPassed<<"  "
      <<setprecision(2)<<setw(7)<<(double)_nEventsL1DoubleTauPassed/(double)_nEvents<<"  "
      <<setprecision(2)<<setw(13)<<(double)_nEventsL1DoubleTauPassed/(double)_nfidEventsDoubleGenTauHads<<"  "
      <<setprecision(2)<<setw(10)<<(double)_nEventsL1DoubleTauPassedMCMatched/(double)_nfidEventsDoubleGenTauHads<<"  "
      <<setprecision(2)<<setw(12)<<(double)_nEventsL1DoubleTauPassedPFMCMatched/(double)_nEventsPFMatchDoubleGenTauHads<<"  "
      <<setprecision(2)<<setw(8)<<(double)_nfidEventsDoubleGenTauHads/(double)_nEvents<<"  "
      <<setprecision(2)<<setw(7)<<(double)_nEventsDoubleGenTauHads/(double)_nEvents<<"  "
      <<endl;
  cout<<"  TauMET ("<<_SingleTauMETThresholds[0]<<","<<_SingleTauMETThresholds[1]<<"): "
      <<setw(6)<<_nEventsL1SingleTauMETPassed<<"  "
      <<setprecision(2)<<setw(7)<<(double)_nEventsL1SingleTauMETPassed/(double)_nEvents<<"  "
      <<setprecision(2)<<setw(13)<<(double)_nEventsL1SingleTauMETPassed/(double)_nfidEventsGenTauHad<<"  "
      <<setprecision(2)<<setw(10)<<(double)_nEventsL1SingleTauMETPassedMCMatched/(double)_nfidEventsGenTauHad<<"  "
      <<setprecision(2)<<setw(12)<<(double)_nEventsL1SingleTauMETPassedPFMCMatched/(double)_nEventsPFMatchGenTauHad<<"  "
      <<setprecision(2)<<setw(8)<<(double)_nfidEventsGenTauHad/(double)_nEvents<<"  "
      <<setprecision(2)<<setw(7)<<(double)_nEventsGenTauHad/(double)_nEvents<<"  "
      <<endl;
  cout<<"    MuTau ("<<_MuTauThresholds[0]<<","<<_MuTauThresholds[1]<<"): "
      <<setw(6)<<_nEventsL1MuonTauPassed<<"  "
      <<setprecision(2)<<setw(7)<<(double)_nEventsL1MuonTauPassed/(double)_nEvents<<"  "
      <<setprecision(2)<<setw(13)<<(double)_nEventsL1MuonTauPassed/(double)_nfidEventsGenTauMuonTauHad<<"  "
      <<setprecision(2)<<setw(10)<<(double)_nEventsL1MuonTauPassedMCMatched/(double)_nfidEventsGenTauMuonTauHad<<"  "
      <<setprecision(2)<<setw(12)<<(double)_nEventsL1MuonTauPassedPFMCMatched/(double)_nEventsPFMatchGenTauMuonTauHad<<"  "
      <<setprecision(2)<<setw(8)<<(double)_nfidEventsGenTauMuonTauHad/(double)_nEvents<<"  "
      <<setprecision(2)<<setw(7)<<(double)_nEventsGenTauMuonTauHad/(double)_nEvents<<"  "
      <<endl;
  cout<<"IsoEgTau ("<<_IsoEgTauThresholds[0]<<","<<_IsoEgTauThresholds[1]<<"): "
      <<setw(6)<<_nEventsL1IsoEgTauPassed<<"  "
      <<setprecision(2)<<setw(7)<<(double)_nEventsL1IsoEgTauPassed/(double)_nEvents<<"  "
      <<setprecision(2)<<setw(13)<<(double)_nEventsL1IsoEgTauPassed/(double)_nfidEventsGenTauElecTauHad<<"  "
      <<setprecision(2)<<setw(10)<<(double)_nEventsL1IsoEgTauPassedMCMatched/(double)_nfidEventsGenTauElecTauHad<<"  "
      <<setprecision(2)<<setw(12)<<(double)_nEventsL1IsoEgTauPassedPFMCMatched/(double)_nEventsPFMatchGenTauElecTauHad<<"  "
      <<setprecision(2)<<setw(8)<<(double)_nfidEventsGenTauElecTauHad/(double)_nEvents<<"  "
      <<setprecision(2)<<setw(7)<<(double)_nEventsGenTauElecTauHad/(double)_nEvents<<"  "
      <<endl;
  
  cout<<endl;

  cout<<"****************************************"<<endl;    
  cout<<"* L1 GT Efficiency Report"<<endl;    
  cout<<"****************************************"<<endl;    
  cout<<"Total number of Events: "<<_nEvents<<endl;    
  cout<<"---------------------------------------------------------------------------------------"<<endl;
  cout<<"                       #PassL1  GlobEff  GlobEff/BR(fid.)  BR(fid.)    BR"<<endl;    
  cout<<"---------------------------------------------------------------------------------------"<<endl;
  cout<<setw(20)<<_L1SingleTauName<<": "
      <<setw(7)<<_nEventsL1GTSingleTauPassed<<"  "
      <<setprecision(2)<<setw(7)<<setw(7)<<(double)_nEventsL1GTSingleTauPassed/(double)_nEvents<<"  "
      <<setprecision(2)<<setw(16)<<(double)_nEventsL1GTSingleTauPassed/(double)_nfidEventsGenTauHad<<"  "
      <<setprecision(2)<<setw(8)<<(double)_nfidEventsGenTauHad/(double)_nEvents<<"  "
      <<setprecision(2)<<setw(7)<<(double)_nEventsGenTauHad/(double)_nEvents<<"  "
      <<endl;
  cout<<setw(20)<<_L1DoubleTauName<<": "
      <<setw(7)<<_nEventsL1GTDoubleTauPassed<<"  "
      <<setprecision(2)<<setw(7)<<(double)_nEventsL1GTDoubleTauPassed/(double)_nEvents<<"  "
      <<setprecision(2)<<setw(16)<<(double)_nEventsL1GTDoubleTauPassed/(double)_nfidEventsDoubleGenTauHads<<"  "
      <<setprecision(2)<<setw(8)<<(double)_nfidEventsDoubleGenTauHads/(double)_nEvents<<"  "
      <<setprecision(2)<<setw(7)<<(double)_nEventsDoubleGenTauHads/(double)_nEvents<<"  "
      <<endl;
  cout<<setw(20)<<_L1TauMETName<<": "
      <<setw(7)<<_nEventsL1GTSingleTauMETPassed<<"  "
      <<setprecision(2)<<setw(7)<<(double)_nEventsL1GTSingleTauMETPassed/(double)_nEvents<<"  "
      <<setprecision(2)<<setw(16)<<(double)_nEventsL1GTSingleTauMETPassed/(double)_nfidEventsGenTauHad<<"  "
      <<setprecision(2)<<setw(8)<<(double)_nfidEventsGenTauHad/(double)_nEvents<<"  "
      <<setprecision(2)<<setw(7)<<(double)_nEventsGenTauHad/(double)_nEvents<<"  "
      <<endl;
  cout<<setw(20)<<_L1MuonTauName<<": "
      <<setw(7)<<_nEventsL1GTMuonTauPassed<<"  "
      <<setprecision(2)<<setw(7)<<(double)_nEventsL1GTMuonTauPassed/(double)_nEvents<<"  "
      <<setprecision(2)<<setw(16)<<(double)_nEventsL1GTMuonTauPassed/(double)_nfidEventsGenTauMuonTauHad<<"  "
      <<setprecision(2)<<setw(8)<<(double)_nfidEventsGenTauMuonTauHad/(double)_nEvents<<"  "
      <<setprecision(2)<<setw(7)<<(double)_nEventsGenTauMuonTauHad/(double)_nEvents<<"  "
      <<endl;
  cout<<setw(20)<<_L1IsoEgTauName<<": "
      <<setw(7)<<_nEventsL1GTIsoEgTauPassed<<"  "
      <<setprecision(2)<<setw(7)<<(double)_nEventsL1GTIsoEgTauPassed/(double)_nEvents<<"  "
      <<setprecision(2)<<setw(16)<<(double)_nEventsL1GTIsoEgTauPassed/(double)_nfidEventsGenTauElecTauHad<<"  "
      <<setprecision(2)<<setw(8)<<(double)_nfidEventsGenTauElecTauHad/(double)_nEvents<<"  "
      <<setprecision(2)<<setw(7)<<(double)_nEventsGenTauElecTauHad/(double)_nEvents<<"  "
      <<endl;
  
  cout<<endl;

}

//define this as a plug-in
//DEFINE_FWK_MODULE(L1TauAnalyzer);

// Original Author:  Chi Nhan Nguyen
//         Created:  Fri Feb 22 09:20:55 CST 2008

#include "DQMOffline/Trigger/interface/HLTTauL1DQMOfflineSource.h"
#include "Math/GenVector/VectorUtil.h"
#include <iostream>
#include <iomanip>
#include <fstream>

//
// constructors and destructor
//
HLTTauL1DQMOfflineSource::HLTTauL1DQMOfflineSource(const edm::ParameterSet& iConfig):

  _refTauColl(iConfig.getUntrackedParameter<edm::InputTag>("RefTauCollection")),
  _refElecColl(iConfig.getUntrackedParameter<edm::InputTag>("RefElecCollection")),
  _refMuonColl(iConfig.getUntrackedParameter<edm::InputTag>("RefMuonCollection")),

  _L1extraTauJetSource(iConfig.getParameter<edm::InputTag>("L1extraTauJetSource")),
  _L1extraCenJetSource(iConfig.getParameter<edm::InputTag>("L1extraCenJetSource")),
  _L1extraForJetSource(iConfig.getParameter<edm::InputTag>("L1extraForJetSource")),
  _L1extraMuonSource(iConfig.getParameter<edm::InputTag>("L1extraMuonSource")),
  _L1extraMETSource(iConfig.getParameter<edm::InputTag>("L1extraMETSource")),
  _L1extraNonIsoEgammaSource(iConfig.getParameter<edm::InputTag>("L1extraNonIsoEgammaSource")),
  _L1extraIsoEgammaSource(iConfig.getParameter<edm::InputTag>("L1extraIsoEgammaSource")),

  _SingleTauThreshold(iConfig.getParameter<double>("SingleTauThreshold")),
  _DoubleTauThreshold(iConfig.getParameter<double>("DoubleTauThreshold")),
  _SingleTauMETThresholds(iConfig.getParameter< std::vector<double> >("SingleTauMETThresholds")),
  _MuTauThresholds(iConfig.getParameter< std::vector<double> >("MuTauThresholds")),
  _IsoEgTauThresholds(iConfig.getParameter< std::vector<double> >("IsoEgTauThresholds")),
  
  _L1MCTauMinDeltaR(iConfig.getParameter<double>("L1RefTauMinDeltaR")),
  _MCTauHadMinEt(iConfig.getParameter<double>("RefTauHadMinEt")),
  _MCTauHadMaxAbsEta(iConfig.getParameter<double>("RefTauHadMaxAbsEta")),

  _L1MCElecMinDeltaR(iConfig.getParameter<double>("L1RefElecMinDeltaR")),
  _MCElecMinEt(iConfig.getParameter<double>("RefElecMinEt")),
  _MCElecMaxAbsEta(iConfig.getParameter<double>("RefElecMaxAbsEta")),
  
  _L1MCMuonMinDeltaR(iConfig.getParameter<double>("L1RefMuonMinDeltaR")),
  _MCMuonMinEt(iConfig.getParameter<double>("RefMuonMinEt")),
  _MCMuonMaxAbsEta(iConfig.getParameter<double>("RefMuonMaxAbsEta")),
  
  _triggerTag((iConfig.getParameter<std::string>("TriggerTag"))),
  _outFile(iConfig.getParameter<std::string>("OutputFileName"))

{
  DQMStore* store = &*edm::Service<DQMStore>();
  
  if(store)
    {
      //Create the histograms
      store->setCurrentFolder(_triggerTag);
      h_L1TauEt = store->book1D("L1TauEt","L1TauEt",50,0.,100.);
      //      h_L1TauEt->getTH1F()->Sumw2();  
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
      

      h_L1IsoEg1Et = store->book1D("L1IsoEg1Et","L1IsoEg1Et",80,0.,80.);
      h_L1IsoEg1Et->getTH1F()->Sumw2();
      h_L1IsoEg1Eta = store->book1D("L1IsoEg1Eta","L1IsoEg1Eta",60,-4.,4.);
      h_L1IsoEg1Eta->getTH1F()->Sumw2();
      h_L1IsoEg1Phi = store->book1D("L1IsoEg1Phi","L1IsoEg1Phi",50,-3.2,3.2);
      h_L1IsoEg1Phi->getTH1F()->Sumw2();
      
      h_L1Muon1Et = store->book1D("L1Muon1Et","L1Muon1Et",80,0.,80.);
      h_L1Muon1Et->getTH1F()->Sumw2();
      h_L1Muon1Eta = store->book1D("L1Muon1Eta","L1Muon1Eta",60,-4.,4.);
      h_L1Muon1Eta->getTH1F()->Sumw2();
      h_L1Muon1Phi = store->book1D("L1Muon1Phi","L1Muon1Phi",50,-3.2,3.2);
      h_L1Muon1Phi->getTH1F()->Sumw2();

      // MET      
      h_L1Met = store->book1D("L1MetEt","L1MetEt",100,0.,200.);
      h_L1Met->getTH1F()->Sumw2();
      h_L1MetEta = store->book1D("L1MetEta","L1MetEta",60,-4.,4.);
      h_L1MetEta->getTH1F()->Sumw2();
      h_L1MetPhi = store->book1D("L1MetPhi","L1MetPhi",50,-3.2,3.2);
      h_L1MetPhi->getTH1F()->Sumw2();

      // L1 response
      h_L1MCTauDeltaR = store->book1D("L1RefTauDeltaR","L1RefTauDeltaR",60,0.,6.);
      h_L1MCTauDeltaR->getTH1F()->Sumw2();
      h_L1minusMCTauEt = store->book1D("L1minusRefTauEt","L1minusRefTauEt",50,-50.,50.);
      h_L1minusMCTauEt->getTH1F()->Sumw2();
      h_L1minusMCoverMCTauEt = store->book1D("L1minusMCoverRefTauEt","L1minusRefoverMCTauEt",40,-1.2,1.2);
      h_L1minusMCoverMCTauEt->getTH1F()->Sumw2();
      
      // MC w/o cuts
      h_GenTauHadEt = store->book1D("GenTauHadEt","GenTauHadEt",50,0.,100.);
      h_GenTauHadEt->getTH1F()->Sumw2();
      h_GenTauHadEta = store->book1D("GenTauHadEta","GenTauHadEt",60,-4.,4.);
      h_GenTauHadEta->getTH1F()->Sumw2();
      h_GenTauHadPhi = store->book1D("GenTauHadPhi","GenTauHadPhi",50,-3.2,3.2);
      h_GenTauHadPhi->getTH1F()->Sumw2();
      
      h_GenTauElecEt = store->book1D("GenTauElecEt","GenTauElecEt",50,0.,100.);
      h_GenTauElecEt->getTH1F()->Sumw2();
      h_GenTauElecEta = store->book1D("GenTauElecEta","GenTauElecEt",60,-4.,4.);
      h_GenTauElecEta->getTH1F()->Sumw2();
      h_GenTauElecPhi = store->book1D("GenTauElecPhi","GenTauElecPhi",50,-3.2,3.2);
      h_GenTauElecPhi->getTH1F()->Sumw2();
      
      h_GenTauMuonEt = store->book1D("GenTauMuonEt","GenTauMuonEt",50,0.,100.);
      h_GenTauMuonEt->getTH1F()->Sumw2();
      h_GenTauMuonEta = store->book1D("GenTauMuonEta","GenTauMuonEt",60,-4.,4.);
      h_GenTauMuonEta->getTH1F()->Sumw2();
      h_GenTauMuonPhi = store->book1D("GenTauMuonPhi","GenTauMuonPhi",50,-3.2,3.2);
      h_GenTauMuonPhi->getTH1F()->Sumw2();
      

      // Tau -> Electron
      // MC matching efficiencies
      h_MCTauElecEt = store->book1D("RefTauElecEt","RefTauElecEt",50,0.,100.);
      h_MCTauElecEt->getTH1F()->Sumw2();
      h_MCTauElecEta = store->book1D("RefTauElecEta","RefTauElecEta",60,-4.,4.);
      h_MCTauElecEta->getTH1F()->Sumw2();
      h_MCTauElecPhi = store->book1D("RefTauElecPhi","RefTauElecPhi",50,-3.2,3.2);
      h_MCTauElecPhi->getTH1F()->Sumw2();

      h_L1MCMatchedTauElecEt = store->book1D("L1RefMatchedTauElecEt","L1RefMatchedTauElecEt",50,0.,100.);
      h_L1MCMatchedTauElecEt->getTH1F()->Sumw2();
      h_L1MCMatchedTauElecEta = store->book1D("L1RefMatchedTauElecEta","L1RefMatchedTauElecEta",60,-4.,4.);
      h_L1MCMatchedTauElecEta->getTH1F()->Sumw2();
      h_L1MCMatchedTauElecPhi = store->book1D("L1RefMatchedTauElecPhi","L1RefMatchedTauElecPhi",50,-3.2,3.2);
      h_L1MCMatchedTauElecPhi->getTH1F()->Sumw2();
      
      //      h_EffMCTauElecEt = store->book1D("EffRefTauElecEt","EffRefTauElecEt",50,0.,100.);
      // h_EffMCTauElecEt->getTH1F()->Sumw2();
      //h_EffMCTauElecEta = store->book1D("EffRefTauElecEta","EffRefTauElecEta",60,-4.,4.);
      //h_EffMCTauElecEta->getTH1F()->Sumw2();
      //h_EffMCTauElecPhi = store->book1D("EffRefTauElecPhi","EffRefTauElecPhi",50,-3.2,3.2);
      //h_EffMCTauElecPhi->getTH1F()->Sumw2();



      // Tau -> Muon
      // MC matching efficiencies
      h_MCTauMuonEt = store->book1D("RefTauMuonEt","RefTauMuonEt",50,0.,100.);
      h_MCTauMuonEt->getTH1F()->Sumw2();
      h_MCTauMuonEta = store->book1D("RefTauMuonEta","RefTauMuonEta",60,-4.,4.);
      h_MCTauMuonEta->getTH1F()->Sumw2();
      h_MCTauMuonPhi = store->book1D("RefTauMuonPhi","RefTauMuonPhi",50,-3.2,3.2);
      h_MCTauMuonPhi->getTH1F()->Sumw2();
      
      h_L1MCMatchedTauMuonEt = store->book1D("L1RefMatchedTauMuonEt","L1RefMatchedTauMuonEt",50,0.,100.);
      h_L1MCMatchedTauMuonEt->getTH1F()->Sumw2();
      h_L1MCMatchedTauMuonEta = store->book1D("L1RefMatchedTauMuonEta","L1RefMatchedTauMuonEta",60,-4.,4.);
      h_L1MCMatchedTauMuonEta->getTH1F()->Sumw2();
      h_L1MCMatchedTauMuonPhi = store->book1D("L1RefMatchedTauMuonPhi","L1RefMatchedTauMuonPhi",50,-3.2,3.2);
      h_L1MCMatchedTauMuonPhi->getTH1F()->Sumw2();
      
      // h_EffMCTauMuonEt = store->book1D("EffRefTauMuonEt","EffRefTauMuonEt",50,0.,100.);
      ///h_EffMCTauMuonEt->getTH1F()->Sumw2();
      //h_EffMCTauMuonEta = store->book1D("EffRefTauMuonEta","EffRefTauMuonEta",60,-4.,4.);
      //h_EffMCTauMuonEta->getTH1F()->Sumw2();
      //h_EffMCTauMuonPhi = store->book1D("EffRefTauMuonPhi","EffRefTauMuonPhi",50,-3.2,3.2);
      //h_EffMCTauMuonPhi->getTH1F()->Sumw2();


      // Tau -> Hadr
      // MC matching efficiencies
      h_MCTauHadEt = store->book1D("RefTauHadEt","RefTauHadEt",50,0.,100.);
      h_MCTauHadEt->getTH1F()->Sumw2();
      h_MCTauHadEta = store->book1D("RefTauHadEta","RefTauHadEta",60,-4.,4.);
      h_MCTauHadEta->getTH1F()->Sumw2();
      h_MCTauHadPhi = store->book1D("RefTauHadPhi","RefTauHadPhi",50,-3.2,3.2);
      h_MCTauHadPhi->getTH1F()->Sumw2();
      
      h_L1MCMatchedTauEt = store->book1D("L1RefMatchedTauEt","L1RefMatchedTauEt",50,0.,100.);
      h_L1MCMatchedTauEt->getTH1F()->Sumw2();
      h_L1MCMatchedTauEta = store->book1D("L1RefMatchedTauEta","L1RefMatchedTauEta",60,-4.,4.);
      h_L1MCMatchedTauEta->getTH1F()->Sumw2();
      h_L1MCMatchedTauPhi = store->book1D("L1RefMatchedTauPhi","L1RefMatchedTauPhi",50,-3.2,3.2);
      h_L1MCMatchedTauPhi->getTH1F()->Sumw2();
      
      // h_EffMCTauEt = store->book1D("EffRefTauEt","EffRefTauEt",50,0.,100.);
      //h_EffMCTauEt->getTH1F()->Sumw2();
      //h_EffMCTauEta = store->book1D("EffRefTauEta","EffRefTauEta",60,-4.,4.);
      //h_EffMCTauEta->getTH1F()->Sumw2();
      //h_EffMCTauPhi = store->book1D("EffRefTauPhi","EffRefTauPhi",50,-3.2,3.2);
      //h_EffMCTauPhi->getTH1F()->Sumw2();
      
      h_L1SingleTauEffEt = store->book1D("L1SingleTauEffEt","L1SingleTauEffEt",
					 50,0.,100.);
       h_L1SingleTauEffEt->getTH1F()->Sumw2();
      h_L1DoubleTauEffEt = store->book1D("L1DoubleTauEffEt","L1DoubleTauEffEt",
					 40,0.,80.);
      h_L1DoubleTauEffEt->getTH1F()->Sumw2();
      h_L1SingleTauEffMCMatchEt = store->book1D("L1SingleTauEffRefMatchEt","L1SingleTauEffRefMatchEt",
						50,0.,100.);
      h_L1SingleTauEffMCMatchEt->getTH1F()->Sumw2();
      h_L1DoubleTauEffMCMatchEt = store->book1D("L1DoubleTauEffRefMatchEt","L1DoubleTauEffRefMatchEt",
						40,0.,80.);
      h_L1DoubleTauEffMCMatchEt->getTH1F()->Sumw2();

      h_L1TauMETfixEffEt = store->book1D("L1TauMETfixEffEt","L1TauMETfixEffEt",
					 50,0.,100.);
      h_L1TauMETfixEffEt->getTH1F()->Sumw2();
      h_L1TauMETfixEffMCMatchEt = store->book1D("L1TauMETfixEffRefMatchEt","L1TauMETfixEffRefMatchEt",
					 50,0.,100.);
      h_L1TauMETfixEffMCMatchEt->getTH1F()->Sumw2();

      h_L1METTaufixEffEt = store->book1D("L1METTaufixEffEt","L1METTaufixEffEt",
					 50,0.,100.);
      h_L1METTaufixEffEt->getTH1F()->Sumw2();
      h_L1METTaufixEffMCMatchEt = store->book1D("L1METTaufixEffRefMatchEt","L1METTaufixEffRefMatchEt",
					 50,0.,100.);
      h_L1METTaufixEffMCMatchEt->getTH1F()->Sumw2();

      h_L1TauIsoEgfixEffEt = store->book1D("L1TauIsoEgfixEffEt","L1TauIsoEgfixEffEt",
					 50,0.,100.);
      h_L1TauIsoEgfixEffEt->getTH1F()->Sumw2();
      h_L1TauIsoEgfixEffMCMatchEt = store->book1D("L1TauIsoEgfixEffRefMatchEt","L1TauIsoEgfixEffRefMatchEt",
					 50,0.,100.);
      h_L1TauIsoEgfixEffMCMatchEt->getTH1F()->Sumw2();
      h_L1IsoEgTaufixEffEt = store->book1D("L1IsoEgTaufixEffEt","L1IsoEgTaufixEffEt",
					 50,0.,100.);
      h_L1IsoEgTaufixEffEt->getTH1F()->Sumw2();
      h_L1IsoEgTaufixEffMCMatchEt = store->book1D("L1IsoEgTaufixEffRefMatchEt","L1IsoEgTaufixEffRefMatchEt",
					 50,0.,100.);
      h_L1IsoEgTaufixEffMCMatchEt->getTH1F()->Sumw2();

      h_L1TauMuonfixEffEt = store->book1D("L1TauMuonfixEffEt","L1TauMuonfixEffEt",
					 50,0.,100.);
      h_L1TauMuonfixEffEt->getTH1F()->Sumw2();
      h_L1TauMuonfixEffMCMatchEt = store->book1D("L1TauMuonfixEffRefMatchEt","L1TauMuonfixEffRefMatchEt",
					 50,0.,100.);
      h_L1TauMuonfixEffMCMatchEt->getTH1F()->Sumw2();

      h_L1MuonTaufixEffEt = store->book1D("L1MuonTaufixEffEt","L1MuonTaufixEffEt",
					 50,0.,100.);
      h_L1MuonTaufixEffEt->getTH1F()->Sumw2();
      h_L1MuonTaufixEffMCMatchEt = store->book1D("L1MuonTaufixEffRefMatchEt","L1MuonTaufixEffRefMatchEt",
					 50,0.,100.);
      h_L1MuonTaufixEffMCMatchEt->getTH1F()->Sumw2();

      h_nfidCounter = store->book1D("nfidCounter","NFID COUNTER",5,0.,5.);

    }
}

// ------------ method called once each job just before starting event loop  ------------
void 
HLTTauL1DQMOfflineSource::beginJob(const edm::EventSetup&)
{
  // Init counters for event based efficiencies
  _nEvents = 0; // all events processed

  _nfidEventsGenTauHad = 0; 
  _nfidEventsDoubleGenTauHads = 0; 
  _nfidEventsGenTauMuonTauHad = 0; 
  _nfidEventsGenTauElecTauHad = 0;   


}

HLTTauL1DQMOfflineSource::~HLTTauL1DQMOfflineSource()
{
}



// ------------ method called to for each event  ------------
void
HLTTauL1DQMOfflineSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   _nEvents++;
   h_nfidCounter->Fill(0.5);

   // get object
   getL1extraObjects(iEvent);
   evalL1extraDecisions();
   
   // fill simple histograms
   fillL1Histograms();
   fillL1MCTauMatchedHists(iEvent);

}



// ------------ method called once each job just after ending the event loop  ------------
void 
HLTTauL1DQMOfflineSource::endJob() {


  // MC matching efficiencies
  //  h_EffMCTauEt->getTH1F()->Divide(h_EffMCTauEt->getTH1F(),h_MCTauHadEt->getTH1F(),1.,1.,"b");
  //h_EffMCTauEta->getTH1F()->Divide(h_EffMCTauEta->getTH1F(),h_MCTauHadEta->getTH1F(),1.,1.,"b");
  //h_EffMCTauPhi->getTH1F()->Divide(h_EffMCTauPhi->getTH1F(),h_MCTauHadPhi->getTH1F(),1.,1.,"b");

  //  h_EffMCTauElecEt->getTH1F()->Divide(h_EffMCTauElecEt->getTH1F(),h_MCTauElecEt->getTH1F(),1.,1.,"b");
  // h_EffMCTauElecEta->getTH1F()->Divide(h_EffMCTauElecEta->getTH1F(),h_MCTauElecEta->getTH1F(),1.,1.,"b");
  //h_EffMCTauElecPhi->getTH1F()->Divide(h_EffMCTauElecPhi->getTH1F(),h_MCTauElecPhi->getTH1F(),1.,1.,"b");

  //h_EffMCTauMuonEt->getTH1F()->Divide(h_EffMCTauMuonEt->getTH1F(),h_MCTauMuonEt->getTH1F(),1.,1.,"b");
  //h_EffMCTauMuonEta->getTH1F()->Divide(h_EffMCTauMuonEta->getTH1F(),h_MCTauMuonEta->getTH1F(),1.,1.,"b");
  //h_EffMCTauMuonPhi->getTH1F()->Divide(h_EffMCTauMuonPhi->getTH1F(),h_MCTauMuonPhi->getTH1F(),1.,1.,"b");

  //
  //convertToIntegratedEff(h_L1SingleTauEffEt,(double)_nEvents);
  //convertToIntegratedEff(h_L1DoubleTauEffEt,(double)_nEvents);
  //convertToIntegratedEff(h_L1SingleTauEffMCMatchEt,(double)_nfidEventsGenTauHad);
  //convertToIntegratedEff(h_L1DoubleTauEffMCMatchEt,(double)_nfidEventsDoubleGenTauHads);

  //convertToIntegratedEff(h_L1TauMETfixEffEt,(double)_nEvents);
  //convertToIntegratedEff(h_L1METTaufixEffEt,(double)_nEvents);
  //convertToIntegratedEff(h_L1TauMETfixEffMCMatchEt,(double)_nfidEventsGenTauHad);
  //convertToIntegratedEff(h_L1METTaufixEffMCMatchEt,(double)_nfidEventsGenTauHad);

  //convertToIntegratedEff(h_L1TauIsoEgfixEffEt,(double)_nEvents);
  //convertToIntegratedEff(h_L1IsoEgTaufixEffEt,(double)_nEvents);
  //convertToIntegratedEff(h_L1TauIsoEgfixEffMCMatchEt,(double)_nfidEventsGenTauElecTauHad);
  //convertToIntegratedEff(h_L1IsoEgTaufixEffMCMatchEt,(double)_nfidEventsGenTauElecTauHad);

  //convertToIntegratedEff(h_L1TauMuonfixEffEt,(double)_nEvents);
  //convertToIntegratedEff(h_L1MuonTaufixEffEt,(double)_nEvents);
  //convertToIntegratedEff(h_L1TauMuonfixEffMCMatchEt,(double)_nfidEventsGenTauMuonTauHad);
  //convertToIntegratedEff(h_L1MuonTaufixEffMCMatchEt,(double)_nfidEventsGenTauMuonTauHad);


  //Write file
  if(_outFile.size()>0)
    if (&*edm::Service<DQMStore>()) edm::Service<DQMStore>()->save (_outFile);
}

void
HLTTauL1DQMOfflineSource::getL1extraObjects(const edm::Event& iEvent)
{
  using namespace edm;
  using namespace l1extra;

  //
  _L1Taus.clear();
  Handle<L1JetParticleCollection> l1TauHandle;
  iEvent.getByLabel(_L1extraTauJetSource,l1TauHandle);
  if (l1TauHandle.isValid()){
  for( L1JetParticleCollection::const_iterator itr = l1TauHandle->begin() ;
       itr != l1TauHandle->end() ; ++itr ) {
    LV p(itr->px(),itr->py(),itr->pz(),itr->energy());
    _L1Taus.push_back(p);
  }
  }
  //
  _L1CenJets.clear();
  Handle<L1JetParticleCollection> l1CenJetHandle;
  iEvent.getByLabel(_L1extraCenJetSource,l1CenJetHandle);
  if (l1CenJetHandle.isValid()) {
  for( L1JetParticleCollection::const_iterator itr = l1CenJetHandle->begin() ;
       itr != l1CenJetHandle->end() ; ++itr ) {
    LV p(itr->px(),itr->py(),itr->pz(),itr->energy());
    _L1CenJets.push_back(p);
  }
  }
  //
  _L1ForJets.clear();
  Handle<L1JetParticleCollection> l1ForJetHandle;
  iEvent.getByLabel(_L1extraForJetSource,l1ForJetHandle);
  if (l1ForJetHandle.isValid()) {
  for( L1JetParticleCollection::const_iterator itr = l1ForJetHandle->begin() ;
       itr != l1ForJetHandle->end() ; ++itr ) {
    LV p(itr->px(),itr->py(),itr->pz(),itr->energy());
    _L1ForJets.push_back(p);
  }
  }
  //
  _L1IsoEgammas.clear();
  Handle<L1EmParticleCollection> l1IsoEgammaHandle;
  iEvent.getByLabel(_L1extraIsoEgammaSource,l1IsoEgammaHandle);
  if (l1IsoEgammaHandle.isValid()) {
  for( L1EmParticleCollection::const_iterator itr = l1IsoEgammaHandle->begin() ;
       itr != l1IsoEgammaHandle->end() ; ++itr ) {
    LV p(itr->px(),itr->py(),itr->pz(),itr->energy());
    _L1IsoEgammas.push_back(p);
  }
  }
  //
  _L1NonIsoEgammas.clear();
  Handle<L1EmParticleCollection> l1NonIsoEgammaHandle;
  iEvent.getByLabel(_L1extraNonIsoEgammaSource,l1NonIsoEgammaHandle);
  if (l1NonIsoEgammaHandle.isValid()) {
  for( L1EmParticleCollection::const_iterator itr = l1NonIsoEgammaHandle->begin() ;
       itr != l1NonIsoEgammaHandle->end() ; ++itr ) {
    LV p(itr->px(),itr->py(),itr->pz(),itr->energy());
    _L1NonIsoEgammas.push_back(p);
  }
  }
  //
  _L1Muons.clear();
  _L1MuQuals.clear();
  Handle<L1MuonParticleCollection> l1MuonHandle;
  iEvent.getByLabel(_L1extraMuonSource,l1MuonHandle);
  if (l1MuonHandle.isValid()) {
  for( L1MuonParticleCollection::const_iterator itr = l1MuonHandle->begin() ;
       itr != l1MuonHandle->end() ; ++itr ) {
    LV p(itr->px(),itr->py(),itr->pz(),itr->energy());
    _L1Muons.push_back(p);
    L1MuGMTExtendedCand gmtCand = itr->gmtMuonCand();
    _L1MuQuals.push_back(gmtCand.quality());// Muon quality as defined in the GT
  }
  }
  //
  _L1METs.clear();
  Handle<L1EtMissParticleCollection> l1MetHandle;
  iEvent.getByLabel(_L1extraMETSource,l1MetHandle);
  if (l1MetHandle.isValid()) {
  for( L1EtMissParticleCollection::const_iterator itr = l1MetHandle->begin() ;
       itr != l1MetHandle->end() ; ++itr ) {
    LV p(itr->px(),itr->py(),itr->pz(),itr->energy());
    _L1METs.push_back(p);
  }
  }
  
  //  Handle<L1EtMissParticle> l1MetHandle;
  //  iEvent.getByLabel(_L1extraMETSource,l1MetHandle);
  // if (l1MetHandle.isValid()){
  // LV p(l1MetHandle->px(),l1MetHandle->py(),l1MetHandle->pz(),l1MetHandle->energy());
  // _L1METs.push_back(p);
  //  }
  
  // Dummy
//  LV p(0.,0.,0.,0.);
//  _L1METs.push_back(p);
  
  
}


void
HLTTauL1DQMOfflineSource::fillL1Histograms() {
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
  for (int i=0; i<(int)_L1IsoEgammas.size(); i++) {
    if (i==0) {
      h_L1IsoEg1Et->Fill(_L1IsoEgammas[i].Et());
      h_L1IsoEg1Eta->Fill(_L1IsoEgammas[i].Eta());
      h_L1IsoEg1Phi->Fill(_L1IsoEgammas[i].Phi());
    }
  }
  for (int i=0; i<(int)_L1Muons.size(); i++) {
    if (i==0) {
      h_L1Muon1Et->Fill(_L1Muons[i].Et());
      h_L1Muon1Eta->Fill(_L1Muons[i].Eta());
      h_L1Muon1Phi->Fill(_L1Muons[i].Phi());
    }
  }
  for (int i=0; i<(int)_L1METs.size(); i++) {
    h_L1Met->Fill(_L1METs[i].Et());
    h_L1MetEta->Fill(_L1METs[i].Eta());
    h_L1MetPhi->Fill(_L1METs[i].Phi());
  }

}


void
HLTTauL1DQMOfflineSource::fillL1MCTauMatchedHists(const edm::Event& iEvent) {
  using namespace edm;

  Handle<LVColl> RefTauH; //Handle To The Truth!!!!
  iEvent.getByLabel(_refTauColl,RefTauH);
  if (!RefTauH.isValid())
    {
      return;
    }
  LVColl RefTau = *RefTauH;

  int nfidTauHads = 0; // count in fiducial region
  for (int i=0; i<(int)RefTau.size(); i++) {
    // w/o further cuts
    h_GenTauHadEt->Fill(RefTau[i].Et());
    h_GenTauHadEta->Fill(RefTau[i].Eta());
    h_GenTauHadPhi->Fill(RefTau[i].Phi());
    // Denominators for MC matching efficiencies
    if (std::abs(RefTau[i].Eta())<=_MCTauHadMaxAbsEta)
      h_MCTauHadEt->Fill(RefTau[i].Et());
    if (RefTau[i].Et()>=_MCTauHadMinEt)
      h_MCTauHadEta->Fill(RefTau[i].Eta());
    if (RefTau[i].Et()>=_MCTauHadMinEt && std::abs(RefTau[i].Eta())<=_MCTauHadMaxAbsEta) {
      h_MCTauHadPhi->Fill(RefTau[i].Phi());
      nfidTauHads++;
    }
  }
  //////// Counters after fiducial cuts!
  if (nfidTauHads >= 1) { _nfidEventsGenTauHad++;  h_nfidCounter->Fill(1.5); } 
  if (nfidTauHads >= 2) {_nfidEventsDoubleGenTauHads++; h_nfidCounter->Fill(2.5);} 

  bool singleMatch = false; // for doubletau match
  bool doubleMatch = false; 
  int iSingle = -1;
  int iDouble = -1;
  for (unsigned int i = 0; i<_L1Taus.size();i++) {
    for (unsigned int j = 0; j<RefTau.size();j++) {
      double deltaR = ROOT::Math::VectorUtil::DeltaR(_L1Taus[i],RefTau[j]);
      h_L1MCTauDeltaR->Fill(deltaR);
      if (deltaR < _L1MCTauMinDeltaR) {
	if (RefTau[j].Et()>=_MCTauHadMinEt && std::abs(RefTau[j].Eta())<=_MCTauHadMaxAbsEta) {
	  h_L1minusMCTauEt->Fill(_L1Taus[i].Et() - RefTau[j].Et());
	  h_L1minusMCoverMCTauEt->Fill( (_L1Taus[i].Et() - RefTau[j].Et()) / RefTau[j].Et());
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
	if (std::abs(RefTau[j].Eta())<=_MCTauHadMaxAbsEta) {	
	  h_L1MCMatchedTauEt->Fill(RefTau[j].Et());
	  //  h_EffMCTauEt->Fill(RefTau[j].Et());
	}
	if (RefTau[j].Et()>=_MCTauHadMinEt) {
	  h_L1MCMatchedTauEta->Fill(RefTau[j].Eta());
	  // h_EffMCTauEta->Fill(RefTau[j].Eta());
	}
	if (RefTau[j].Et()>=_MCTauHadMinEt && std::abs(RefTau[j].Eta())<=_MCTauHadMaxAbsEta) {
	  h_L1MCMatchedTauPhi->Fill(RefTau[j].Phi());
	  // h_EffMCTauPhi->Fill(RefTau[j].Phi());
	}
      }           
    }
  }
  
  // Now Electrons and Muons
  Handle<LVColl> RefElecH; //Handle To The Truth!!!!
  iEvent.getByLabel(_refElecColl,RefElecH);
  LVColl RefElec = *RefElecH;

  Handle<LVColl> RefMuonH; //Handle To The Truth!!!!
  iEvent.getByLabel(_refMuonColl,RefMuonH);
  LVColl RefMuon = *RefMuonH;

  int nfidTauElecs = 0; // count in fiducial region
  for (int i=0; i<(int)RefElec.size(); i++) {
    // w/o further cuts
    h_GenTauElecEt->Fill(RefElec[i].Et());
    h_GenTauElecEta->Fill(RefElec[i].Eta());
    h_GenTauElecPhi->Fill(RefElec[i].Phi());
    // Denominators for MC matching efficiencies
    if (std::abs(RefElec[i].Eta())<=_MCElecMaxAbsEta)
      h_MCTauElecEt->Fill(RefElec[i].Et());
    if (RefElec[i].Et()>=_MCElecMinEt)
      h_MCTauElecEta->Fill(RefElec[i].Eta());
    if (RefElec[i].Et()>=_MCElecMinEt && std::abs(RefElec[i].Eta())<=_MCElecMaxAbsEta) {
      h_MCTauElecPhi->Fill(RefElec[i].Phi());
      nfidTauElecs++;
    }
  }

  int nfidTauMuons = 0; // count in fiducial region
  for (int i=0; i<(int)RefMuon.size(); i++) {
    // w/o further cuts
    h_GenTauMuonEt->Fill(RefMuon[i].Et());
    h_GenTauMuonEta->Fill(RefMuon[i].Eta());
    h_GenTauMuonPhi->Fill(RefMuon[i].Phi());
    // Denominators for MC matching efficiencies
    if (std::abs(RefMuon[i].Eta())<=_MCMuonMaxAbsEta)
      h_MCTauMuonEt->Fill(RefMuon[i].Et());
    if (RefMuon[i].Et()>=_MCMuonMinEt)
      h_MCTauMuonEta->Fill(RefMuon[i].Eta());
    if (RefMuon[i].Et()>=_MCMuonMinEt && std::abs(RefMuon[i].Eta())<=_MCMuonMaxAbsEta) {
      h_MCTauMuonPhi->Fill(RefMuon[i].Phi());
	nfidTauMuons++;
    }
  }

  for (unsigned int i = 0; i<_L1IsoEgammas.size();i++) {
    for (unsigned int j = 0; j<RefElec.size();j++) {
      double deltaR = ROOT::Math::VectorUtil::DeltaR(_L1IsoEgammas[i],RefElec[j]);
      if (deltaR < _L1MCElecMinDeltaR) {
	// Numerators for MC matching efficiencies
	if (std::abs(RefElec[j].Eta())<=_MCElecMaxAbsEta) {	
	  h_L1MCMatchedTauElecEt->Fill(RefElec[j].Et());
	  // h_EffMCTauElecEt->Fill(RefElec[j].Et());
	}
	if (RefElec[j].Et()>=_MCElecMinEt) {
	  h_L1MCMatchedTauElecEta->Fill(RefElec[j].Eta());
	  //h_EffMCTauElecEta->Fill(RefElec[j].Eta());
	}
	if (RefElec[j].Et()>=_MCElecMinEt && std::abs(RefElec[j].Eta())<=_MCElecMaxAbsEta) {
	  h_L1MCMatchedTauElecPhi->Fill(RefElec[j].Phi());
	  //h_EffMCTauElecPhi->Fill(RefElec[j].Phi());
	}
      }           
    }
  }

  for (unsigned int i = 0; i<_L1Muons.size();i++) {
    for (unsigned int j = 0; j<RefMuon.size();j++) {
      double deltaR = ROOT::Math::VectorUtil::DeltaR(_L1IsoEgammas[i],RefMuon[j]);
      if (deltaR < _L1MCMuonMinDeltaR) {
	// Numerators for MC matching efficiencies
	if (std::abs(RefMuon[j].Eta())<=_MCMuonMaxAbsEta) {	
	  h_L1MCMatchedTauMuonEt->Fill(RefMuon[j].Et());
	  //h_EffMCTauMuonEt->Fill(RefMuon[j].Et());
	}
	if (RefMuon[j].Et()>=_MCMuonMinEt) {
	  h_L1MCMatchedTauMuonEta->Fill(RefMuon[j].Eta());
	  //h_EffMCTauMuonEta->Fill(RefMuon[j].Eta());
	}
	if (RefMuon[j].Et()>=_MCMuonMinEt && std::abs(RefMuon[j].Eta())<=_MCMuonMaxAbsEta) {
	  h_L1MCMatchedTauMuonPhi->Fill(RefMuon[j].Phi());
	  //h_EffMCTauMuonPhi->Fill(RefMuon[j].Phi());
	}
      }           
    }
  }
  ////


  // Now Event efficiencies
  if (singleMatch && iSingle>=0) {
    h_L1SingleTauEffMCMatchEt->Fill(_L1Taus[iSingle].Et()); 
   
    if (_L1Taus[iSingle].Et()>=_SingleTauMETThresholds[0])
      h_L1METTaufixEffMCMatchEt->Fill(_L1METs[0].Et());    
    if (_L1METs[0].Et()>=_SingleTauMETThresholds[1])
      h_L1TauMETfixEffMCMatchEt->Fill(_L1Taus[iSingle].Et());

    
    for (int i=0;i<(int)_L1Muons.size();i++) {
      //if (_L1Muons[i].Et()>=_MuTauThresholds[0]) {
	//if ( _L1MuQuals[0]==4 || _L1MuQuals[0]==5 || _L1MuQuals[0]==6 || _L1MuQuals[0]==7 ) {
	if ( _L1MuQuals[0]==3 || _L1MuQuals[0]==5 || _L1MuQuals[0]==6 || _L1MuQuals[0]==7 ) {
	  //if ( _L1MuQuals[i]>=0) {
	  for (int j=0;j<(int)RefMuon.size();j++) {
	    double deltaR = ROOT::Math::VectorUtil::DeltaR(RefMuon[j],_L1Muons[i]);
	    if (deltaR < _L1MCMuonMinDeltaR) {	      
	      if (_L1Taus[iSingle].Et()>=_MuTauThresholds[1]) 
		h_L1MuonTaufixEffMCMatchEt->Fill(_L1Muons[i].Et());    
	      if (_L1Muons[i].Et()>=_MuTauThresholds[0])
		h_L1TauMuonfixEffMCMatchEt->Fill(_L1Taus[iSingle].Et());
	      break;
	    }
	  }
	  //}
	}
    }
    
    // No collinearity check yet!
    for (int j=0;j<(int)_L1IsoEgammas.size();j++) {
      //if (_L1Taus[iSingle].Et()>=_IsoEgTauThresholds[1] &&
      //    _L1IsoEgammas[j].Et()>=_IsoEgTauThresholds[0]) {
      //double deltaPhi = ROOT::Math::VectorUtil::DeltaPhi(_L1Taus[iSingle],_L1IsoEgammas[j]);
      //double deltaEta = std::abs(_L1Taus[iSingle].Eta()-_L1IsoEgammas[j].Eta());
      // Non-collinearity check
	//if (deltaPhi>0.348 && deltaEta>0.348) {
      for (int k=0;k<(int)RefElec.size();k++) {
	double deltaR = ROOT::Math::VectorUtil::DeltaR(RefElec[k],_L1IsoEgammas[j]);
	if (deltaR < _L1MCElecMinDeltaR) {
	  if (_L1Taus[iSingle].Et()>=_IsoEgTauThresholds[1]) 
	    h_L1IsoEgTaufixEffMCMatchEt->Fill(_L1IsoEgammas[j].Et());    
	  if (_L1IsoEgammas[j].Et()>=_IsoEgTauThresholds[0])
	    h_L1TauIsoEgfixEffMCMatchEt->Fill(_L1Taus[iSingle].Et());    
	  break; 
	}
      }
      //}
      //}
    }

  }

  if (doubleMatch && iDouble>=0) {
    h_L1DoubleTauEffMCMatchEt->Fill(_L1Taus[iDouble].Et());
    if (_L1Taus[iDouble].Et()>=_DoubleTauThreshold)
      ;
    for (int j=0;j<(int)_L1IsoEgammas.size();j++) {
      if (_L1Taus[iDouble].Et()>=_IsoEgTauThresholds[1] &&
	  _L1IsoEgammas[j].Et()>=_IsoEgTauThresholds[0]) {
	//double deltaPhi = ROOT::Math::VectorUtil::DeltaPhi(_L1Taus[iDouble],_L1IsoEgammas[j]);
	//double deltaEta = std::abs(_L1Taus[iDouble].Eta()-_L1IsoEgammas[j].Eta());
	// Non-collinearity check
	//if (deltaPhi>0.348 && deltaEta>0.348) {
	  for (int k=0;k<(int)RefElec.size();k++) {
	    double deltaR = ROOT::Math::VectorUtil::DeltaR(RefElec[k],_L1IsoEgammas[j]);
	    if (deltaR<_L1MCElecMinDeltaR) {
	      ;
	      break;
	    }
	//}
	}
      }
    }
  }

  //////// Counters after fiducial cuts!
  if (nfidTauHads >= 1 && nfidTauMuons >= 1) {_nfidEventsGenTauMuonTauHad++; h_nfidCounter->Fill(3.5);} 
  if (nfidTauHads >= 1 && nfidTauElecs >= 1) {_nfidEventsGenTauElecTauHad++; h_nfidCounter->Fill(4.5);} 

  
}

void
HLTTauL1DQMOfflineSource::evalL1extraDecisions() {
  if (_L1Taus.size()>=1) {
    h_L1SingleTauEffEt->Fill(_L1Taus[0].Et());
    if (_L1Taus[0].Et()>=_SingleTauMETThresholds[0]) 
      h_L1METTaufixEffEt->Fill(_L1METs[0].Et());    
    if (_L1METs[0].Et()>=_SingleTauMETThresholds[1])
      h_L1TauMETfixEffEt->Fill(_L1Taus[0].Et());

    if (_L1Muons.size()>=1) {
      if (_L1Taus[0].Et()>=_MuTauThresholds[1]) 
	h_L1MuonTaufixEffEt->Fill(_L1Muons[0].Et());    
      if (_L1Muons[0].Et()>=_MuTauThresholds[0])
	h_L1TauMuonfixEffEt->Fill(_L1Taus[0].Et());
    }
    if (_L1IsoEgammas.size()>=1) {
      if (_L1Taus[0].Et()>=_IsoEgTauThresholds[1]) 
	h_L1IsoEgTaufixEffEt->Fill(_L1IsoEgammas[0].Et());    
      if (_L1IsoEgammas[0].Et()>=_IsoEgTauThresholds[0])
	h_L1TauIsoEgfixEffEt->Fill(_L1Taus[0].Et());    
    }
  }
  if (_L1Taus.size()>=2 ) {
    h_L1DoubleTauEffEt->Fill(_L1Taus[1].Et());
  }
}


//void
//HLTTauL1DQMOfflineSource::convertToIntegratedEff(MonitorElement* histo, double nGenerated)
//{
  // Convert the histogram to efficiency
  // Assuming that the histogram is incremented with weight=1 for each event
  // this function integrates the histogram contents above every bin and stores it
  // in that bin.  The result is plot of integral rate versus threshold plot.
//  int nbins = histo->getNbinsX();
//  double integral = histo->getBinContent(nbins+1);  // Initialize to overflow
//  if (nGenerated<=0)  {
//    std::cerr << "***** HLTTauL1DQMOfflineSource::convertToIntegratedEff() Error: nGenerated = " << nGenerated << std::endl;
//    //nGenerated=1;
//    return;
//  }
//  for(int i = nbins; i >= 1; i--)
//    {
//      double thisBin = histo->getBinContent(i);
//      integral += thisBin;
//      double integralEff;
//      double integralError;
//     integralEff = (integral / nGenerated);
//      histo->setBinContent(i, integralEff);
//      // error
//      integralError = (sqrt(integral) / nGenerated);
//      histo->setBinError(i, integralError);
//    }
//}
//define this as a plug-in
//DEFINE_FWK_MODULE(HLTTauL1DQMOfflineSource);

#ifndef L25TauAnalyzer_h
#define L25TauAnalyzer_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"


// HEP library include files
#include <Math/GenVector/VectorUtil.h>
//#include "CLHEP/HepMC/GenEvent.h"
//#include "CLHEP/HepMC/GenVertex.h"
//#include "CLHEP/HepMC/GenParticle.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "HLTriggerOffline/Tau/interface/MCTauCand.h"


#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"

#include "DataFormats/Math/interface/Vector.h"

// basic cpp:
#include <string>
#include <vector>

// some root includes
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>


class L25TauAnalyzer : public edm::EDAnalyzer {

   public:


  typedef struct MatchElement {
    bool matched;
    double deltar;
    double mcEta;
    double mcEt;
  };

   
      explicit L25TauAnalyzer(const edm::ParameterSet&);
              ~L25TauAnalyzer();

 
   private:
   
  typedef math::XYZTLorentzVectorD   LV;
  typedef std::vector<LV>            LVColl;
  typedef std::vector<edm::InputTag> VInputTag;
  typedef std::vector<std::string>   VString;

  // Utility to sort LVColls by Et (increasing)
  class LVCollEtSorter {
  public:
    bool operator()( const LV &x1, const LV &x2 ) const {
      return x1.Et() > x2.Et();
    }
  };
  class MyTrackSorter {
  public:
    
    bool 
      operator()( const reco::Track & a, const reco::Track & b ) {
      return (a.pt()>b.pt());
    }
    bool 
      operator()( const reco::Track * a, const reco::Track * b ) {
      return (a->pt()>b->pt());
    }
  };

  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
 
  // ----------member data ---------------------------
  
  int debug;
    
  LVColl        mcTauJet, mcLepton, mcNeutrina; // products from HLTMcInfo
  LVColl        level1Tau, level1Lep, level1Obj, level1TauMatched, level1LepMatched, level1ObjMatched;
  LVColl        level2Tau, level2FiltTau, level2TauMatched, level2FiltTauMatched;
  LVColl        level25Tau, level25TauMatched;
  LVColl        level3Tau, level3Lep;


  // List of generated leptons from Z, W or H
  std::vector<HepMC::GenParticle> _GenBosons;
  std::vector<HepMC::GenParticle> _GenElecs;
  std::vector<HepMC::GenParticle> _GenMuons;
  std::vector<HepMC::GenParticle> _GenTauElecs;
  std::vector<HepMC::GenParticle> _GenTauMuons;
  std::vector<HepMC::GenParticle> _GenTauHadrons;
  std::vector<HepMC::GenParticle> _GenTauChargedHadrons;
  std::vector<MCTauCand> _GenTaus;

  bool          isL25Accepted;
  size_t        nEventsL25,nEventsL25Tagged;
  size_t        nEventsL25Matched;
  
  size_t        nEventsL25Riso[10], nEventsL3Riso[10]; // to evaluate efficiency of isolation cone
  
  size_t	nbTaus, nbLeps; // number of taus to be checked to validate a step in the trigger path ( 1 for all, 2 for double tau )

  bool          passAll; // Pass all events through

  VInputTag     mcProducts; // input products from HLTMcInfo

  double        mcDeltaRTau; // Max dR tau    w.r.t mc tau
  double        mcDeltaRLep; // Max dR lepton w.r.t mc lepton

  edm::InputTag l1ParticleMap;
  std::string   l1TauTrigger;
  edm::InputTag l2TauInfoAssoc_;

  VInputTag     l2TauJets;
  VInputTag     l2TauJetsFiltered;
  VInputTag     l25TauJets;
  VInputTag     l3TauJets;

  VInputTag     hltLeptonSrc; // lepton input for lepton+tau triggers
  std::string   _GeneratorSource;

  std::string   rootFile; // output ROOT file name
  std::string   logFile;
  
  bool          isSignal; // false if QCD, true if Z,H,A,Z',...

  TFile*        tauFile;  // output ROOT file

  // Monte Carlo plots
  TH1F *hMcTauEt, *hMcTauEta, *hMcLepEt, *hMcLepEta;

  TH1F *hMCTauTrueEt, *hMCTauVisibleEt, *hMCTauEta, *hMCTauPhi;
  TH1F *hMCTrkPt;
  TH2F *hMCTrkEta, *hMCTrkPhi, *hMCNTrk;
  TH1F *hMCTauNProngs;


  TH1F * hL25JetEtL2Bare;
  TH1F * hL25JetEtaL2Bare;
  TH1F * hL25JetPhiL2Bare;


  TH1F * hL25JetEtL2BareMatched;
  TH1F * hL25JetEtaL2BareMatched;
  TH1F * hL25JetPhiL2BareMatched;

  // Plots from L25
  TH1F *hL25Acc,*hL25Matched,
    *hL25JetEt, *hL25JetEta, *hL25JetPhi, *hL25JetDr, *hL25JetNtrk,
    *hL25JetEtMatched, *hL25JetEtaMatched, *hL25JetPhiMatched, *hL25JetDrMatched, *hL25JetNtrkMatched,
    *hL25TrkPt, *hL25TrkChi2, *hL25TrkRecHits, *hL25TrkPixHits, *hL25TrkD0, *hL25TrkZ0,
    *hL25MTauTau, *hL25MTauTauAll, *hL25Trk1NHits;
  TH2F *hL25JetIsoDisc, *hL25Trk1NTrk, *hL25Trk2NTrk, 
    *hL25Trk1Hits,
    *hL25Trk1Layer0Hits,*hL25Trk1Layer1Hits,*hL25Trk1Layer2Hits;

  // MC Tracks and decays at level 2.5
  TH1F *hL25MCTauTrueEt, *hL25MCTauVisibleEt;
  TH1F *hL25MCTauEta, *hL25MCTauPhi;
  TH1F *hL25MCTrkPt;
  TH2F *hL25MCTrkEta, *hL25MCTrkPhi, *hL25MCNTrk;
  TH1F *hL25MCTauNProngs;
  TH1F *hL25MCTrkPtWithMatch;
  TH2F *hL25MCTrkEtaWithMatch, *hL25MCTrkPhiWithMatch, *hL25MCNTrkWithMatch;
  TH1F *hL25MCTauNProngsWithMatch;


  TH1F *hL25MCTauTrueEtAllMC, *hL25MCTauVisibleEtAllMC;
  TH1F *hL25MCTauEtaAllMC, *hL25MCTauPhiAllMC;
  TH1F *hL25MCTauNProngsAllMC;

  TH1F *hL25MCTauTrueEtHadTaus, *hL25MCTauVisibleEtHadTaus;
  TH1F *hL25MCTauEtaHadTaus, *hL25MCTauPhiHadTaus;
  TH1F *hL25MCTauNProngsHadTaus;

  TH1F *hL25MCTauTrueEtMCTrkFid, *hL25MCTauVisibleEtMCTrkFid;
  TH1F *hL25MCTauEtaMCTrkFid, *hL25MCTauPhiMCTrkFid;
  TH1F *hL25MCTauNProngsMCTrkFid;

  TH1F *hL25MCTauTrueEtPFMatch, *hL25MCTauVisibleEtPFMatch;
  TH1F *hL25MCTauEtaPFMatch, *hL25MCTauPhiPFMatch;
  TH1F *hL25MCTauNProngsPFMatch;

  TH1F *hL25MCTauTrueEtReco, *hL25MCTauVisibleEtReco;
  TH1F *hL25MCTauEtaReco, *hL25MCTauPhiReco;
  TH1F *hL25MCTauNProngsReco;

  TH1F *hL25MCTauTrueEtSeed, *hL25MCTauVisibleEtSeed;
  TH1F *hL25MCTauEtaSeed, *hL25MCTauPhiSeed;
  TH1F *hL25MCTauNProngsSeed;

  TH1F *hL25MCTauTrueEtTrkPt, *hL25MCTauVisibleEtTrkPt;
  TH1F *hL25MCTauEtaTrkPt, *hL25MCTauPhiTrkPt;
  TH1F *hL25MCTauNProngsTrkPt;

  TH1F *hL25MCTauTrueEtTrkIso, *hL25MCTauVisibleEtTrkIso;
  TH1F *hL25MCTauEtaTrkIso, *hL25MCTauPhiTrkIso;
  TH1F *hL25MCTauNProngsTrkIso;


  TH1F *hL25MCMatchedTrkPt;
  TH1F *hL25MCMatchedTrkMCPt;
  TH1F *hL25MCMatchedTrkDr;
  TH2F *hL25MCMatchedTrkEta, *hL25MCMatchedTrkPhi;
  TH2F *hL25MCNMatchedTrk;
  TH2F *hL25MCMatchedTrkPtVsMCTrkPt;

  void MakeGeneratorAnalysis( const edm::Event& iEvent, const edm::EventSetup & iSetup );
  void MakeLevel25Analysis( const edm::Event& iEvent, const edm::EventSetup & iSetup );
  void MakeSummary( const std::string & Option );

  void ComputeEfficiency( const int Num = 0, const int Den = 0, float & Eff = 0., float & EffErr = 0. );
  float isDrMatched( const LV& v, const LVColl& Coll, float dRCut = 0. );

  
  void getGenObjects(const edm::Event& iEvent, const edm::EventSetup & iSetup);
  MCTauCand * getMatchedTauCand( const LV & p4, double dRcut );

  std::vector<HepMC::GenParticle*> getGenStableDecayProducts(const HepMC::GenVertex* vertex);

  int recoTrackDrMatch( const CLHEP::HepLorentzVector & p4, const reco::IsolatedTauTagInfo & tau, double &dR_out, double dRcut ) const;


  MatchElement match(const reco::Jet& jet,const LVColl& McInfo);

  CLHEP::HepLorentzVector convertTo( const HepMC::FourVector& v );
};

#endif

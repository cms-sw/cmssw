// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
// #include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

// #include <DataFormats/Common/interface/RefVector.h>
// #include <DataFormats/Common/interface/RefVectorIterator.h>
// #include <DataFormats/Common/interface/Ref.h>
// #include "DataFormats/Common/interface/RefToBase.h"


#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
//#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

// basic cpp:
#include <string>
#include <vector>

// some root includes
#include <TFile.h>
#include <TH1F.h>

// class TauJet : public TLorentzVector
// {
//  
//  float IsMatched( Coll, float dRCut = 0. ) { 
//      for( int i = 0; i < Coll.GetEntriesFast(); ++i ) {
//       float dR = DeltaR( *(TLorentzVector*)Coll.At(i) );
//       if( dR < dRCut ) { break; return dR; }
//      }
//     }
// 
// };
// 
// class HLTTauEvent : public TObject
// {
//  
// };

class HLTTauAnalyzer : public edm::EDAnalyzer {

   public:
   
      explicit HLTTauAnalyzer(const edm::ParameterSet&);
              ~HLTTauAnalyzer();
 
   private:
   
  typedef math::XYZTLorentzVectorD   LV;
  typedef std::vector<LV>            LVColl;
  typedef std::vector<edm::InputTag> VInputTag;
  typedef std::vector<std::string>   VString;

  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
 
  // ----------member data ---------------------------
  
  int debug;
    
  LVColl        mcTauJet, mcLepton, mcNeutrina; // products from HLTMcInfo
  LVColl        level1Tau, level1Lep, level3Tau, level3Lep;

  bool          isMcMatched, isL1Accepted, isL2Accepted,isL2FilterAccepted,isL2FilterMcMatched, isL25Accepted, isL3Accepted, isL1McMatched, isL2McMatched, isL25McMatched, isL3McMatched,isL2METAccepted;
  size_t        nEventsTot, nEventsTotMcMatched, nEventsL1, nEventsL1McMatched, nEventsL2, nEventsL2McMatched, nEventsL2Filtered, nEventsL2FilteredMcMatched, nEventsL25,nEventsL25McMatched,nEventsL3,nEventsL3McMatched;
  size_t nEventsL2MET;
  
  size_t	nbTaus, nbLeps; // number of taus to be checked to validate a step in the trigger path ( 1 for all, 2 for double tau )

  VInputTag     mcProducts; // input products from HLTMcInfo

  double        mcDeltaRTau; // Max dR tau    w.r.t mc tau
  double        mcDeltaRLep; // Max dR lepton w.r.t mc lepton

  edm::InputTag l1ParticleMap;
  std::string   l1TauTrigger;
  edm::InputTag      metReco;

  bool          usingMET;

  VInputTag     l2TauJets;
  VInputTag     l2TauJetsFiltered;
  VInputTag     l25TauJets;
  VInputTag     l3TauJets;

  VInputTag     hltLeptonSrc; // lepton input for lepton+tau triggers

  std::string   rootFile; // output ROOT file name
  std::string   logFile;
  
  bool          isSignal; // false if QCD, true if Z,H,A,Z',...
  TFile*        tauFile;  // output ROOT file
  
  TH1F *hMcTauEt, *hMcTauEta, *hMcLepEt, *hMcLepEta;


  void MakeGeneratorAnalysis( const edm::Event& iEvent );
  void MakeLevel1Analysis( const edm::Event& iEvent );
  void MakeLevel2METAnalysis(const edm::Event& iEvent );
  void MakeLevel2Analysis( const edm::Event& iEvent );
  void MakeLevel25Analysis( const edm::Event& iEvent );
  void MakeLevel3Analysis( const edm::Event& iEvent );
  void MakeSummary( const std::string & Option );

  void ComputeEfficiency( const int Num , const int Den , float& Eff, float &EffErr );
  float isDrMatched( const LV& v, const LVColl& Coll, float dRCut = 0. );
};


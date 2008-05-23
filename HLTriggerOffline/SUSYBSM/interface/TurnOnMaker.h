#ifndef TurnOnMaker_h
#define TurnOnMaker_h

/*  \class TurnOnMaker
*
*  Class to produce some turn on curves in the TriggerValidation Code
*  the curves are produced by associating the reco and mc objects to l1 and hlt objects
*
*  Author: Massimiliano Chiorboli      Date: November 2007
//         Maurizio Pierini
//         Maria Spiropulu
*
*/
#include <memory>
#include <string>
#include <iostream>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"


//For L1 and Hlt objects
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/Candidate/interface/Candidate.h"


//For RECO objects
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

//For Gen Objects
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"


//For Muon tracks used by the trigger
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"



#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "TH1.h"

using namespace edm;
using namespace reco;
using namespace std;
using namespace trigger;
using namespace l1extra;

class TurnOnMaker {

 public:
  TurnOnMaker(edm::ParameterSet turnOn_params);
  virtual ~TurnOnMaker(){};

  void handleObjects(const edm::Event&);
  void fillPlots(const edm::Event&);
  void bookHistos();
  void writeHistos();
  //  void finalOperations();


 private:
  typedef std::vector<std::string> vstring;

  //define the parameters
  std::string m_genSrc;
  std::string m_genMother;
  std::string m_recoMuonSrc;
  std::string m_hltMuonTrackSrc;
  std::vector<std::string> m_hlt1MuonIsoSrc;     
  std::vector<std::string> m_hlt1MuonNonIsoSrc;  


/*   bool recoToTriggerMatched(reco::Candidate*, std::vector< edm::RefToBase< reco::Candidate > >); */
/*   bool recoToTriggerMatched(const reco::Candidate*, std::vector< edm::RefToBase< reco::Candidate > >); */
  bool recoToTriggerMatched(reco::Candidate*, std::vector< std::vector<RecoChargedCandidateRef> >, int);
  bool recoToTriggerMatched(const reco::Candidate*, std::vector< std::vector<RecoChargedCandidateRef> >, int);

  bool recoToTracksMatched(reco::Candidate*, reco::RecoChargedCandidateCollection, double, std::string);
  bool recoToTracksMatched(const reco::Candidate*, reco::RecoChargedCandidateCollection, double, std::string);

  bool triggerToTriggerMatched(RecoChargedCandidateRef, std::vector< std::vector<RecoChargedCandidateRef> >, int);




  //the HLT Collections
  // these objects are vectors of vectors since they contain the different level of HLT
  // and the different objects for every level
  // so: theHLT1MuonIsoObjectVector[i][j] means the j-th HltMuonIso in the i-th level of trigger
  std::vector< std::vector<RecoChargedCandidateRef> > theHLT1MuonIsoObjectVector;
  std::vector< std::vector<RecoChargedCandidateRef> > theHLT1MuonNonIsoObjectVector;

  //the Reco Muon Collection
  reco::MuonCollection                  theMuonCollection;
  //the Muon Tracks used for the trigger
  reco::RecoChargedCandidateCollection theMuonTrackCollection;

  //the Gen Particles
  const reco::CandidateCollection* theGenParticleCollection;


  //histos

  //Trigger Muons
  // I consider the HLT paths:
  // HLT1MuonIso
  // HLT1MuonNonIso


  // The following vectors are the distributions of the various "layers"
  // of the HLT
  // e.g.
  // hHLT1MuonIsoPt[0] is the Pt distribution of the level 0 of the HLT path HLT1MuonIso
  // which is SingleMuIsoLevel1Seed

  //Levels:
  //
  //HLT1MuonIso (from HLTrigger/Muon/data/PathSingleMu_1032_Iso.cff)
  //
  //SingleMuIsoLevel1Seed
  //SingleMuIsoL1Filtered
  //SingleMuIsoL2PreFiltered
  //SingleMuIsoL2IsoFiltered
  //SingleMuIsoL3PreFiltered
  //SingleMuIsoL3IsoFiltered
  //
  //
  //
  //HLT1MuonNonIso (from HLTrigger/Muon/data/PathSingleMu_1032_NoIso.cff)
  //
  //SingleMuNoIsoLevel1Seed
  //SingleMuNoIsoL1Filtered
  //SingleMuNoIsoL2PreFiltered
  //SingleMuNoIsoL3PreFiltered
  //
  //
  // The first level (SingleMuNoIsoLevel1Seed) can be identified with L1
  //
  //
  // The objects corresponding to these levels are taken from
  // HLTFilterObjectWithRefs using the label of the wanted level

  std::vector<TH1D*> hHLT1MuonIsoMult;
  std::vector<TH1D*> hHLT1MuonIsoPt;
  std::vector<TH1D*> hHLT1MuonIsoEta;
  std::vector<TH1D*> hHLT1MuonIsoPhi;

  std::vector<TH1D*> hHLT1MuonNonIsoMult;
  std::vector<TH1D*> hHLT1MuonNonIsoPt;
  std::vector<TH1D*> hHLT1MuonNonIsoEta;
  std::vector<TH1D*> hHLT1MuonNonIsoPhi;


  // Distributions of muon tracks used for trigger
  // These are NOT reco muons
  // These plots are built just to check the correct behaviour
  // of the plots built with HLTObjectsWithRef

  TH1D* hMuonTrackPt;
  TH1D* hMuonTrackEta;
  TH1D* hMuonTrackPhi;
  TH1D* hMuonTrackMult;
  



  
  
  //Distributions of the Reco Muons
  TH1D* hRecoMuonPt;             // Reco Muons 
  TH1D* hRecoMuonEta;             // Reco Muons 
  TH1D* hRecoMuonPhi;             // Reco Muons 
  TH1D* hRecoMuonMult;             // Reco Muons 

  TH1D* hRecoMuonPtBarrel;             // Reco Muons 
  TH1D* hRecoMuonPtEndcap;             // Reco Muons 

  TH1D* hRecoMuonEtaPt10;             // Reco Muons 
  TH1D* hRecoMuonEtaPt20;             // Reco Muons 

  //Distributions of the Reco Muons associated with the HLT object (for the various levels)
  //for Pt the distribution is calculated in 3 scenarios:
  // - without any cut in pt
  // - for |eta|<1.2 (Barrel)
  // - for 1.2<|eta|<2.1 (Endcap)
  //for Eta also 3 scenarios:
  // - without any cut in pt
  // - pt>10 GeV
  // - pt>20 GeV
  std::vector<TH1D*> hRecoMuonPtAssHLT1MuonIso;
  std::vector<TH1D*> hRecoMuonPtAssHLT1MuonIsoBarrel;
  std::vector<TH1D*> hRecoMuonPtAssHLT1MuonIsoEndcap;
  std::vector<TH1D*> hRecoMuonEtaAssHLT1MuonIso;
  std::vector<TH1D*> hRecoMuonEtaAssHLT1MuonIsoPt10;
  std::vector<TH1D*> hRecoMuonEtaAssHLT1MuonIsoPt20;
  
  std::vector<TH1D*> hRecoMuonPtAssHLT1MuonNonIso;
  std::vector<TH1D*> hRecoMuonPtAssHLT1MuonNonIsoBarrel;
  std::vector<TH1D*> hRecoMuonPtAssHLT1MuonNonIsoEndcap;
  std::vector<TH1D*> hRecoMuonEtaAssHLT1MuonNonIso;
  std::vector<TH1D*> hRecoMuonEtaAssHLT1MuonNonIsoPt10;
  std::vector<TH1D*> hRecoMuonEtaAssHLT1MuonNonIsoPt20;


 

  // Pt dirtibutions of Reco Muons associated
  // to Muon tracks used to build the trigger
  TH1D* hRecoMuonPtAssMuonTrackIso;
  TH1D* hRecoMuonPtAssMuonTrackIsoBarrel;
  TH1D* hRecoMuonPtAssMuonTrackIsoEndcap;
  TH1D* hRecoMuonPtAssMuonTrackIsoDr2;
  TH1D* hRecoMuonPtAssMuonTrackIsoDr2Barrel;
  TH1D* hRecoMuonPtAssMuonTrackIsoDr2Endcap;
  TH1D* hRecoMuonPtAssMuonTrackIsoDr02;
  TH1D* hRecoMuonPtAssMuonTrackIsoDr02Barrel;
  TH1D* hRecoMuonPtAssMuonTrackIsoDr02Endcap;
  TH1D* hRecoMuonPtAssMuonTrackIsoDr002;
  TH1D* hRecoMuonPtAssMuonTrackIsoDr002Barrel;
  TH1D* hRecoMuonPtAssMuonTrackIsoDr002Endcap;

  TH1D* hRecoMuonPtAssMuonTrackNonIso;
  TH1D* hRecoMuonPtAssMuonTrackNonIsoBarrel;
  TH1D* hRecoMuonPtAssMuonTrackNonIsoEndcap;
  TH1D* hRecoMuonPtAssMuonTrackNonIsoDr2;
  TH1D* hRecoMuonPtAssMuonTrackNonIsoDr2Barrel;
  TH1D* hRecoMuonPtAssMuonTrackNonIsoDr2Endcap;
  TH1D* hRecoMuonPtAssMuonTrackNonIsoDr02;
  TH1D* hRecoMuonPtAssMuonTrackNonIsoDr02Barrel;
  TH1D* hRecoMuonPtAssMuonTrackNonIsoDr02Endcap;
  TH1D* hRecoMuonPtAssMuonTrackNonIsoDr002;
  TH1D* hRecoMuonPtAssMuonTrackNonIsoDr002Barrel;
  TH1D* hRecoMuonPtAssMuonTrackNonIsoDr002Endcap;

  // Eta dirtibutions of Reco Muons associated
  // to Muon tracks used to build the trigger
  TH1D* hRecoMuonEtaAssMuonTrackIso;
  TH1D* hRecoMuonEtaAssMuonTrackIsoPt10;
  TH1D* hRecoMuonEtaAssMuonTrackIsoPt20;
  TH1D* hRecoMuonEtaAssMuonTrackIsoDr2;
  TH1D* hRecoMuonEtaAssMuonTrackIsoDr2Pt10;
  TH1D* hRecoMuonEtaAssMuonTrackIsoDr2Pt20;
  TH1D* hRecoMuonEtaAssMuonTrackIsoDr02;
  TH1D* hRecoMuonEtaAssMuonTrackIsoDr02Pt10;
  TH1D* hRecoMuonEtaAssMuonTrackIsoDr02Pt20;
  TH1D* hRecoMuonEtaAssMuonTrackIsoDr002;
  TH1D* hRecoMuonEtaAssMuonTrackIsoDr002Pt10;
  TH1D* hRecoMuonEtaAssMuonTrackIsoDr002Pt20;

  TH1D* hRecoMuonEtaAssMuonTrackNonIso;
  TH1D* hRecoMuonEtaAssMuonTrackNonIsoPt10;
  TH1D* hRecoMuonEtaAssMuonTrackNonIsoPt20;
  TH1D* hRecoMuonEtaAssMuonTrackNonIsoDr2;
  TH1D* hRecoMuonEtaAssMuonTrackNonIsoDr2Pt10;
  TH1D* hRecoMuonEtaAssMuonTrackNonIsoDr2Pt20;
  TH1D* hRecoMuonEtaAssMuonTrackNonIsoDr02;
  TH1D* hRecoMuonEtaAssMuonTrackNonIsoDr02Pt10;
  TH1D* hRecoMuonEtaAssMuonTrackNonIsoDr02Pt20;
  TH1D* hRecoMuonEtaAssMuonTrackNonIsoDr002;
  TH1D* hRecoMuonEtaAssMuonTrackNonIsoDr002Pt10;
  TH1D* hRecoMuonEtaAssMuonTrackNonIsoDr002Pt20;



  //Distributions of the Gen Muons
  TH1D* hGenMuonPt;             // Gen Muons 
  TH1D* hGenMuonEta;             // Gen Muons 
  TH1D* hGenMuonPhi;             // Gen Muons 
  TH1D* hGenMuonMult;             // Gen Muons 


  TH1D* hGenMuonPtBarrel;             // Gen Muons 
  TH1D* hGenMuonPtEndcap;             // Gen Muons 

  TH1D* hGenMuonEtaPt10;             // Gen Muons 
  TH1D* hGenMuonEtaPt20;             // Gen Muons 

  //Distributions of the Gen Muons associated with the HLT object (for the various levels)
  //for Pt the distribution is calculated in 3 scenarios:
  // - without any cut in pt
  // - for |eta|<1.2 (Barrel)
  // - for 1.2<|eta|<2.1 (Endcap)
  //for Eta also 3 scenarios:
  // - without any cut in pt
  // - pt>10 GeV
  // - pt>20 GeV
  std::vector<TH1D*> hGenMuonPtAssHLT1MuonIso;
  std::vector<TH1D*> hGenMuonPtAssHLT1MuonIsoBarrel;
  std::vector<TH1D*> hGenMuonPtAssHLT1MuonIsoEndcap;
  std::vector<TH1D*> hGenMuonEtaAssHLT1MuonIso;
  std::vector<TH1D*> hGenMuonEtaAssHLT1MuonIsoPt10;
  std::vector<TH1D*> hGenMuonEtaAssHLT1MuonIsoPt20;
  
  std::vector<TH1D*> hGenMuonPtAssHLT1MuonNonIso;
  std::vector<TH1D*> hGenMuonPtAssHLT1MuonNonIsoBarrel;
  std::vector<TH1D*> hGenMuonPtAssHLT1MuonNonIsoEndcap;
  std::vector<TH1D*> hGenMuonEtaAssHLT1MuonNonIso;
  std::vector<TH1D*> hGenMuonEtaAssHLT1MuonNonIsoPt10;
  std::vector<TH1D*> hGenMuonEtaAssHLT1MuonNonIsoPt20;

  // Pt dirtibutions of Gen Muons associated
  // to Muon tracks used to build the trigger
  TH1D* hGenMuonPtAssMuonTrackIso;
  TH1D* hGenMuonPtAssMuonTrackIsoBarrel;
  TH1D* hGenMuonPtAssMuonTrackIsoEndcap;
  TH1D* hGenMuonPtAssMuonTrackIsoDr2;
  TH1D* hGenMuonPtAssMuonTrackIsoDr2Barrel;
  TH1D* hGenMuonPtAssMuonTrackIsoDr2Endcap;
  TH1D* hGenMuonPtAssMuonTrackIsoDr02;
  TH1D* hGenMuonPtAssMuonTrackIsoDr02Barrel;
  TH1D* hGenMuonPtAssMuonTrackIsoDr02Endcap;
  TH1D* hGenMuonPtAssMuonTrackIsoDr002;
  TH1D* hGenMuonPtAssMuonTrackIsoDr002Barrel;
  TH1D* hGenMuonPtAssMuonTrackIsoDr002Endcap;

  TH1D* hGenMuonPtAssMuonTrackNonIso;
  TH1D* hGenMuonPtAssMuonTrackNonIsoBarrel;
  TH1D* hGenMuonPtAssMuonTrackNonIsoEndcap;
  TH1D* hGenMuonPtAssMuonTrackNonIsoDr2;
  TH1D* hGenMuonPtAssMuonTrackNonIsoDr2Barrel;
  TH1D* hGenMuonPtAssMuonTrackNonIsoDr2Endcap;
  TH1D* hGenMuonPtAssMuonTrackNonIsoDr02;
  TH1D* hGenMuonPtAssMuonTrackNonIsoDr02Barrel;
  TH1D* hGenMuonPtAssMuonTrackNonIsoDr02Endcap;
  TH1D* hGenMuonPtAssMuonTrackNonIsoDr002;
  TH1D* hGenMuonPtAssMuonTrackNonIsoDr002Barrel;
  TH1D* hGenMuonPtAssMuonTrackNonIsoDr002Endcap;

  // Eta dirtibutions of Gen Muons associated
  // to Muon tracks used to build the trigger
  TH1D* hGenMuonEtaAssMuonTrackIso;
  TH1D* hGenMuonEtaAssMuonTrackIsoPt10;
  TH1D* hGenMuonEtaAssMuonTrackIsoPt20;
  TH1D* hGenMuonEtaAssMuonTrackIsoDr2;
  TH1D* hGenMuonEtaAssMuonTrackIsoDr2Pt10;
  TH1D* hGenMuonEtaAssMuonTrackIsoDr2Pt20;
  TH1D* hGenMuonEtaAssMuonTrackIsoDr02;
  TH1D* hGenMuonEtaAssMuonTrackIsoDr02Pt10;
  TH1D* hGenMuonEtaAssMuonTrackIsoDr02Pt20;
  TH1D* hGenMuonEtaAssMuonTrackIsoDr002;
  TH1D* hGenMuonEtaAssMuonTrackIsoDr002Pt10;
  TH1D* hGenMuonEtaAssMuonTrackIsoDr002Pt20;

  TH1D* hGenMuonEtaAssMuonTrackNonIso;
  TH1D* hGenMuonEtaAssMuonTrackNonIsoPt10;
  TH1D* hGenMuonEtaAssMuonTrackNonIsoPt20;
  TH1D* hGenMuonEtaAssMuonTrackNonIsoDr2;
  TH1D* hGenMuonEtaAssMuonTrackNonIsoDr2Pt10;
  TH1D* hGenMuonEtaAssMuonTrackNonIsoDr2Pt20;
  TH1D* hGenMuonEtaAssMuonTrackNonIsoDr02;
  TH1D* hGenMuonEtaAssMuonTrackNonIsoDr02Pt10;
  TH1D* hGenMuonEtaAssMuonTrackNonIsoDr02Pt20;
  TH1D* hGenMuonEtaAssMuonTrackNonIsoDr002;
  TH1D* hGenMuonEtaAssMuonTrackNonIsoDr002Pt10;
  TH1D* hGenMuonEtaAssMuonTrackNonIsoDr002Pt20;




  //No turnon curves implemented here
  //they should be implemented in the second half of the code (Maurizio's)

  std::string s_Iso;
  std::string s_NonIso;


  std::string myHistoName;
  std::string myHistoTitle;

};

#endif

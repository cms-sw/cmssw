#ifndef DQMOffline_Trigger_HLTMuonMatchAndPlot_H
#define DQMOffline_Trigger_HLTMuonMatchAndPlot_H

/** \class HLTMuonMatchAndPlot
 *  Get L1/HLT efficiency/rate plots
 *  Documentation available on the CMS TWiki:
 *  https://twiki.cern.ch/twiki/bin/view/CMS/MuonHLTOfflinePerformance
 *
 *  $Date: 2009/07/21 08:47:26 $
 *  $Revision: 1.4 $
 *  \author  M. Vander Donckt, J. Klukas  (copied from J. Alcaraz)
 *  \author  J. Slaunwhite (modified from above
 */

// Base Class Headers

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
//#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/MuonReco/interface/Muon.h"
//#include "PhysicsTools/Utilities/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include <vector>
#include "TFile.h"
#include "TNtuple.h"



typedef math::XYZTLorentzVector LorentzVector;
using reco::Muon;
using trigger::TriggerObject;

//-----------------------------------------
// Add in a struct that gathers together
// the selectors and string that form the
// definition of each object
//-----------------------------------------

struct MuonSelectionStruct {

  // constructor is empty, but passes in arguments
  MuonSelectionStruct(StringCutObjectSelector<Muon> rsel, StringCutObjectSelector<TriggerObject> hltsel,
                      std::string cl, double pD0cut, double pZ0cut, std::string trackCol, std::vector<std::string> reqTrigs)
    :recoMuonSelector(rsel), hltMuonSelector(hltsel), customLabel(cl),
     d0cut(pD0cut), z0cut(pZ0cut), trackCollection(trackCol),
     requiredTriggers(reqTrigs) {};
  
  StringCutObjectSelector<Muon> recoMuonSelector;
  StringCutObjectSelector<TriggerObject> hltMuonSelector;
  std::string customLabel;

  
  // the track cuts you want to use
  // note: for future dev, you may want to
  // also include some way to force use of beamspot
  // paramters.


  double d0cut;
  double z0cut;

  
  // the track collection you want to use
  std::string trackCollection;


  // included here for completeness, but not
  // yet fully implemented

  std::vector<std::string> requiredTriggers;
  
  // note... do we want to make a raw selector?
  // raw trigger events are called trig event with refs
  
};



class HLTMuonMatchAndPlot {

public:

  /// Constructor
  HLTMuonMatchAndPlot( const edm::ParameterSet& pset, std::string triggerName,
                      std::vector<std::string> moduleNames, MuonSelectionStruct inputSelection,
                      std::string customName,
                      std::vector<std::string> validTriggers );

  // Operations
  void            begin  ( );
  void            analyze( const edm::Event & iEvent );
  void            finish ( );
  MonitorElement* bookIt ( TString name, TString title, std::vector<double> );
  MonitorElement* bookIt ( TString name, TString title, int nbins, float* xBinLowEdges);


  // Struct and methods for matching

  // Big Change from RelVal
  // Now we store Muons and TriggerObjects
  // instead of Tracks and 4-vectors
  struct MatchStruct {
    
    //const reco::GenParticle*   genCand;
    const reco::Muon*         recCand;
    // Can't understsand how to use these references
    //l1extra::L1MuonParticleRef   l1Cand;
    //l1extra::L1MuonParticleRef   l1RawCand;

    trigger::TriggerObject l1Cand;
    LorentzVector l1RawCand;
    std::vector<trigger::TriggerObject> hltCands;
    // Can't handle the raw objects
    // just use 4-vector
    std::vector<LorentzVector> hltRawCands;
    //std::vector<const reco::RecoChargedCandidate*> hltTracks;

    // Do we really want to store just the lorentz vectors
    // for the trigger objects? No charge, d0 information
    
  };

  // Structure that  holds a Lorentz Vector corresponding
  // to an HLT object and a bool to indicate whether or
  // not it is a fake
  struct HltFakeStruct {    
    trigger::TriggerObject              myHltCand;
    bool                       isAFake;    
  };

  // store the matches for each event
  std::vector<MatchStruct> recMatches;
  
private:
  
  const reco::Candidate* findMother( const reco::Candidate* );
  int findGenMatch( double eta, double phi, double maxDeltaR,
		    std::vector<MatchStruct> matches );
  int findRecMatch( double eta, double phi, double maxdeltaR,
		    std::vector<MatchStruct> matches );

  bool applyTrackSelection (MuonSelectionStruct mySelection, reco::Muon candMuon);
  reco::TrackRef getCandTrackRef (MuonSelectionStruct mySelection, reco::Muon candMuon);

  bool applyTriggerSelection (MuonSelectionStruct mySelection, const edm::Event & event);
  
  // Data members

  // flag to decide how you want to label output histos
  // old label scheme kept lots of the trigger name,
  // but complicated harvesting step
  bool    useOldLabels;
  bool    useFullDebugInformation;
  int     HLT_PLOT_OFFSET;
  bool    isL1Path, isL2Path, isL3Path;

  bool    makeNtuple;
  float   theNtuplePars[100]; 
  TNtuple *theNtuple;
  TFile   *theFile;

  // Input from cfg file

  std::string              theHltProcessName;
  std::string              theTriggerName;
  std::string              theL1CollectionLabel;
  std::vector<std::string> theHltCollectionLabels;
  unsigned int             theNumberOfObjects;

  //bool         useMuonFromGenerator;
  bool         useMuonFromReco;
  //std::string  theGenLabel;
  //std::string  theRecoLabel;
  
  edm::InputTag RecoMuonInputTag;
  edm::InputTag BeamSpotInputTag;
  edm::InputTag HltRawInputTag;
  edm::InputTag HltAodInputTag;
  
  bool         useAod;
  std::string  theAodL1Label;
  std::string  theAodL2Label;


  std::string matchType;

  MuonSelectionStruct mySelection;

  //======= Trigger Selection info ==========
  edm::InputTag TriggerResultLabel;

  std::vector<std::string> selectedValidTriggers;
  
  
  //StringCutObjectSelector<Muon> myMuonSelector;
  //std::string myCustomName;
  
  // constants and a method
  // to simplify charge plots

  static const int POS_CHARGE = 1;
  static const int NEG_CHARGE = 0;
  int getCharge (int pdgId); 

  // 1-D histogram paramters
  std::vector<double> theMaxPtParameters;
  std::vector<double> thePtParameters;
  std::vector<double> theEtaParameters;
  std::vector<double> thePhiParameters;
  std::vector<double> thePhiParameters0Pi;
  std::vector<double> theD0Parameters; 
  std::vector<double> theZ0Parameters;
  std::vector<double> theChargeParameters;
  std::vector<double> theDRParameters;
  std::vector<double> theChargeFlipParameters;

  // variable width pt bins
  // don't allow more than 100
  float  ptBins[100];
  int numBinsInPtHisto;

  // 2-D histogram parameters
  std::vector<double> theMaxPtParameters2d;
  std::vector<double> theEtaParameters2d;
  std::vector<double> thePhiParameters2d;
  std::vector<double> thePhiEtaParameters2d;
  std::vector<double> theDeltaPhiVsPhiParameters;
  std::vector<double> theDeltaPhiVsZ0Parameters;
  std::vector<double> theDeltaPhiVsD0Parameters;
  

  // Resolution hisotgram parameters
  std::vector<double> theResParameters;

  // isolation parameters
  std::vector<double> theIsolationParameters;
  


  double       theMinPtCut;
  double       theMaxEtaCut;
  double       theL1DrCut;
  double       theL2DrCut;
  double       theL3DrCut;
  int          theMotherParticleId;
  std::vector<double> theNSigmas;

  std::string  theNtupleFileName;
  std::string  theNtuplePath;

  // Book-keeping information

  int          eventNumber;
  unsigned int numHltLabels;
  bool         isIsolatedPath;

  // Monitor Elements (Histograms and ints)

  DQMStore* dbe_;

  bool createStandAloneHistos;
  std::string histoFileName;
  

 //  std::vector <MonitorElement*> hPassMaxPtGen;
//   std::vector <MonitorElement*> hPassEtaGen;
//   std::vector <MonitorElement*> hPassPhiGen;
  std::vector <MonitorElement*> hPassMaxPtRec;
  std::vector <MonitorElement*> hPassEtaRec;
  std::vector <MonitorElement*> hPassPhiRec;
  std::vector <MonitorElement*> hPassMatchPtRec;
  std::vector <MonitorElement*> hPassExaclyOneMuonMaxPtRec;
  //std::vector <MonitorElement*> hPtMatchVsPtRec;
  //std::vector <MonitorElement*> hEtaMatchVsEtaRec;
  //std::vector <MonitorElement*> hPhiMatchVsPhiRec;
  std::vector <MonitorElement*> hPhiVsEtaRec;
  std::vector <MonitorElement*> hPassPtRec;
  std::vector <MonitorElement*> hPassPtRecExactlyOne;
  std::vector <MonitorElement*> hResoPtAodRec;
  std::vector <MonitorElement*> hResoEtaAodRec;
  std::vector <MonitorElement*> hResoPhiAodRec;
  std::vector <MonitorElement*> hPassD0Rec;
  std::vector <MonitorElement*> hPassZ0Rec;
  std::vector <MonitorElement*> hPassCharge;
  std::vector <MonitorElement*> hPassD0BeamRec;
  std::vector <MonitorElement*> hPassZ0BeamRec;
  std::vector <MonitorElement*> hBeamSpotZ0Rec;

  // studies for cosmics
  std::vector <MonitorElement*> hMatchedDeltaPhi;
  //std::vector <MonitorElement*> hDeltaPhiVsPhi;
  //std::vector <MonitorElement*> hDeltaPhiVsZ0;
  //std::vector <MonitorElement*> hDeltaPhiVsD0;
  
  

  std::vector <MonitorElement*> hDeltaRMatched;
  std::vector <MonitorElement*> hIsolationRec;
  std::vector <MonitorElement*> hChargeFlipMatched;

  std::vector <MonitorElement*> allHltCandPt;
  std::vector <MonitorElement*> allHltCandEta;
  std::vector <MonitorElement*> allHltCandPhi;
  // can't do d0, z0 becuase only have 4 vectors
  //std::vector <MonitorElement*> allHltCandD0Rec;
  //std::vector <MonitorElement*> allHltCandZ0Rec;
    
  std::vector <MonitorElement*> fakeHltCandPt;
  std::vector <MonitorElement*> fakeHltCandEta;
  std::vector <MonitorElement*> fakeHltCandPhi;
  //std::vector <MonitorElement*> fakeHltCandD0Rec;
  //std::vector <MonitorElement*> fakeHltCandZ0Rec;

  //std::vector <MonitorElement*> fakeHltCandEtaPhi;
  //std::vector <MonitorElement*> fakeHltCandDeltaR;

  // RAW histograms
  // L1, L2, L3
  std::vector <MonitorElement*> rawMatchHltCandPt;
  std::vector <MonitorElement*> rawMatchHltCandEta;
  std::vector <MonitorElement*> rawMatchHltCandPhi;

  // 
  //std::string highPtTrackCollection;
  
  
  MonitorElement *hNumObjects;
  //MonitorElement *hNumOrphansGen;
  MonitorElement *hNumOrphansRec;
  MonitorElement *meNumberOfEvents;

};
#endif

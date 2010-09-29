#ifndef DQMOffline_Trigger_HLTMuonMatchAndPlot_H
#define DQMOffline_Trigger_HLTMuonMatchAndPlot_H

/** \class HLTMuonMatchAndPlot
 *  Get L1/HLT efficiency/rate plots
 *  Documentation available on the CMS TWiki:
 *  https://twiki.cern.ch/twiki/bin/view/CMS/MuonHLTOfflinePerformance
 *
 *  $Date: 2010/03/16 14:36:39 $
 *  $Revision: 1.12 $
 *  
 *  \author  J. Slaunwhite, based on code from Jeff Klukas
 */

// Base Class Headers

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
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
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/MuonReco/interface/Muon.h"
//#include "CommonTools/Utilities/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include <vector>
#include "TFile.h"
#include "TNtuple.h"
#include "TString.h"


typedef math::XYZTLorentzVector LorentzVector;

//-----------------------------------------
// Add in a struct that gathers together
// the selectors and string that form the
// definition of each object
//-----------------------------------------

struct MuonSelectionStruct {

  // constructor is empty, but passes in arguments
  MuonSelectionStruct(StringCutObjectSelector<reco::Muon> rsel, StringCutObjectSelector<trigger::TriggerObject> hltsel,
                      std::string cl, double pD0cut, double pZ0cut, std::string trackCol, std::vector<std::string> reqTrigs)
    :recoMuonSelector(rsel), hltMuonSelector(hltsel), customLabel(cl),
     d0cut(pD0cut), z0cut(pZ0cut), trackCollection(trackCol),
     requiredTriggers(reqTrigs) {};
  
  StringCutObjectSelector<reco::Muon> recoMuonSelector;
  StringCutObjectSelector<trigger::TriggerObject> hltMuonSelector;
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
                       std::vector<std::string> validTriggers,
                       const edm::Run & currentRun,
                       const edm::EventSetup & currentEventSetup );

  // Operations
  virtual void            begin  ( );
  virtual void            analyze( const edm::Event & iEvent );
  virtual void            finish ( );
  virtual void            endRun (const edm::Run& r, const edm::EventSetup& c);
  virtual MonitorElement* bookIt ( TString name, TString title, std::vector<double> );
  virtual MonitorElement* bookIt ( TString name, TString title, int nbins, float* xBinLowEdges);
  
  TString             calcHistoSuffix (std::string moduleName);

  // Struct and methods for matching

  // Big Change from RelVal
  // Now we store Muons and TriggerObjects
  // instead of Tracks and 4-vectors
  struct MatchStruct {

    // this is the reco muon
    const reco::Muon*         recCand;
    
    // this is the cand from the L1 filter
    // in L1 passthru triggers
    // that matches the reco muon
    // (from trigger summary aod)
    trigger::TriggerObject l1Cand;

    // this is the L1 seed candidate from
    // HLT trigger paths

    trigger::TriggerObject l1Seed;

    // this is the raw candidate from
    // the trigger summary with refs
    // that matches the reco muon    
    LorentzVector l1RawCand;

    // these are the hlt candidates from
    // trigger sum AOD that match
    // the reco muon
    std::vector<trigger::TriggerObject> hltCands;

    // these are the hlt candidates from
    // trigger sum w/ refs that match
    // the reco muon
    std::vector<LorentzVector> hltRawCands;
    
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

  std::vector< std::vector<HltFakeStruct> > hltFakeCands;

  virtual ~HLTMuonMatchAndPlot() {} ;

  //-------- The fuctions/data below used to be private, but we want to
  //-------- have other classes inherit them, so we make them public

  

  // Functions that are an internal decomposition of the stages of the
  // public "analyze" function

  // They need to be public so that other modules can use them?
  
  bool virtual selectAndMatchMuons(const edm::Event & iEvent,
                                   std::vector<MatchStruct> & myRecMatches,
                                   std::vector< std::vector<HltFakeStruct> > & myHltFakeCands
                                   );

  // does this need to be generalized to work for a given trigger too?
  bool  selectAndMatchMuons(const edm::Event & iEvent,
                            std::vector<MatchStruct> & myRecMatches,
                            std::vector< std::vector<HltFakeStruct> > & myHltFakeCands,
                            MuonSelectionStruct muonSelection
                            );

  
  void virtual fillPlots(std::vector<MatchStruct> & myRecMatches,
                         std::vector< std::vector<HltFakeStruct> > & myHltFakeCands);

protected:
  
  ////////////////////////////////////////////////////
  
  const reco::Candidate* findMother( const reco::Candidate* );
  int findGenMatch( double eta, double phi, double maxDeltaR,
		    std::vector<MatchStruct> matches );
  int findRecMatch( double eta, double phi, double maxdeltaR,
		    std::vector<MatchStruct> matches );

  bool virtual applyTrackSelection (MuonSelectionStruct mySelection, reco::Muon candMuon);
  reco::TrackRef getCandTrackRef (MuonSelectionStruct mySelection, reco::Muon candMuon);

  bool virtual applyTriggerSelection (MuonSelectionStruct mySelection, const edm::Event & event);

  // boolean to turn on/off overflow inclusion
  // function to do the overflow
  bool includeOverflow;
  void moveOverflow (MonitorElement * myElement);


  // put the code for getting trigger objects into a single module
  // fills up foundObjects with your matching trigger objects
  void getAodTriggerObjectsForModule (edm::InputTag collectionTag,
                                      edm::Handle<trigger::TriggerEvent> aodTriggerEvent,
                                      trigger::TriggerObjectCollection trigObjs,
                                      std::vector<trigger::TriggerObject> & foundObjects,
                                      MuonSelectionStruct muonSelection);
  
  // keep track of the ME's you've booked,
  // rebin them at job end.
  std::vector<MonitorElement*> booked1DMonitorElements;
  
  // Data members

  // flag to decide how you want to label output histos
  // old label scheme kept lots of the trigger name,
  // but complicated harvesting step
  bool    useOldLabels;
  bool    useFullDebugInformation;
  int     HLT_PLOT_OFFSET;
  bool    isL1Path, isL2Path, isL3Path;

  unsigned int numL1Cands;

  bool    makeNtuple;
  float   theNtuplePars[100]; 
  TNtuple *theNtuple;
  TFile   *theFile;

  reco::BeamSpot  beamSpot;
  bool foundBeamSpot;


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

  bool requireL1SeedForHLTPaths;


  std::string matchType;

  MuonSelectionStruct mySelection;

  //======= Trigger Selection info ==========
  edm::InputTag TriggerResultLabel;

  std::vector<std::string> selectedValidTriggers;

  std::string theL1SeedModuleForHLTPath;
  
  //StringCutObjectSelector<reco::Muon> myMuonSelector;
  //std::string myCustomName;
  
  // constants and a method
  // to simplify charge plots

  static const int POS_CHARGE = 1;
  static const int NEG_CHARGE = -1;
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
  std::vector<double> thePhiEtaParameters2d;

  //   std::vector<double> theMaxPtParameters2d;
  //   std::vector<double> theEtaParameters2d;
  //   std::vector<double> thePhiParameters2d;
  
  //   std::vector<double> theDeltaPhiVsPhiParameters;
  //   std::vector<double> theDeltaPhiVsZ0Parameters;
  //   std::vector<double> theDeltaPhiVsD0Parameters;
  

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

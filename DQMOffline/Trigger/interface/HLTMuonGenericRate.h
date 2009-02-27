#ifndef DQMOffline_Trigger_HLTMuonGenericRate_H
#define DQMOffline_Trigger_HLTMuonGenericRate_H

/** \class HLTMuonGenericRate
 *  Get L1/HLT efficiency/rate plots
 *  Documentation available on the CMS TWiki:
 *  https://twiki.cern.ch/twiki/bin/view/CMS/MuonHLTOfflinePerformance
 *
 *  $Date: 2009/02/11 20:21:26 $
 *  $Revision: 1.30 $
 *  \author  M. Vander Donckt, J. Klukas  (copied from J. Alcaraz)
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

#include <vector>
#include "TFile.h"
#include "TNtuple.h"



typedef math::XYZTLorentzVector LorentzVector;



class HLTMuonGenericRate {

public:

  /// Constructor
  HLTMuonGenericRate( const edm::ParameterSet& pset, std::string triggerName,
		      std::vector<std::string> moduleNames );

  // Operations
  void            begin  ( );
  void            analyze( const edm::Event & iEvent );
  void            finish ( );
  MonitorElement* bookIt ( TString name, TString title, std::vector<double> );

private:

  // Struct and methods for matching

  struct MatchStruct {
    //const reco::GenParticle*   genCand;
    const reco::Track*         recCand;
    LorentzVector              l1Cand;
    std::vector<LorentzVector> hltCands;
    std::vector<const reco::RecoChargedCandidate*> hltTracks;
  };

  const reco::Candidate* findMother( const reco::Candidate* );
  int findGenMatch( double eta, double phi, double maxDeltaR,
		    std::vector<MatchStruct> matches );
  int findRecMatch( double eta, double phi, double maxdeltaR,
		    std::vector<MatchStruct> matches );
  
  // Data members

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

  bool         useMuonFromGenerator;
  bool         useMuonFromReco;
  std::string  theGenLabel;
  std::string  theRecoLabel;

  bool         useAod;
  std::string  theAodL1Label;
  std::string  theAodL2Label;

  std::vector<double> theMaxPtParameters;
  std::vector<double> thePtParameters;
  std::vector<double> theEtaParameters;
  std::vector<double> thePhiParameters;

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
  

  std::vector <MonitorElement*> hPassMaxPtGen;
  std::vector <MonitorElement*> hPassEtaGen;
  std::vector <MonitorElement*> hPassPhiGen;
  std::vector <MonitorElement*> hPassMaxPtRec;
  std::vector <MonitorElement*> hPassEtaRec;
  std::vector <MonitorElement*> hPassPhiRec;

  MonitorElement *hNumObjects;
  MonitorElement *hNumOrphansGen;
  MonitorElement *hNumOrphansRec;
  MonitorElement *meNumberOfEvents;

};
#endif

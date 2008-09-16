#ifndef HLTriggerOffline_Muon_HLTMuonGenericRate_H
#define HLTriggerOffline_Muon_HLTMuonGenericRate_H

/** \class HLTMuonGenericRate
 *  Get L1/HLT efficiency/rate plots
 *
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
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <vector>
#include "TFile.h"
#include "TNtuple.h"


class HLTMuonGenericRate {

public:

  /// Constructor
  HLTMuonGenericRate(const edm::ParameterSet& pset, int triggerIndex);

  /// Destructor
  virtual ~HLTMuonGenericRate();

  // Operations
  void analyze          ( const edm::Event & iEvent );
  void BookHistograms   ( );
  MonitorElement* BookIt( TString name, TString title, 
			  int Nbins, float Min, float Max );

private:

  // Input from cfg file
  edm::InputTag theL1CollectionLabel;
  std::string   theGenLabel;
  std::string   theRecoLabel;
  std::vector<edm::InputTag> theHLTCollectionLabels;

  double theL1ReferenceThreshold;
  double theHLTReferenceThreshold;
  double theLuminosity;
  double thePtMin;
  double thePtMax;
  double theMinPtCut;
  double theMaxEtaCut;

  std::vector<double> theNSigmas;
  unsigned int theNumberOfObjects;
  unsigned int theNbins;
  int  thisEventWeight;
  int  theMotherParticleId;
  bool m_useMuonFromGenerator;
  bool m_useMuonFromReco;
  bool m_makeNtuple;

  // Struct for matching
  struct MatchStruct {
    const reco::GenParticle*       genCand;
    const reco::Track*             recCand;
    const l1extra::L1MuonParticle* l1Cand;
    std::vector<const reco::RecoChargedCandidate*> hltCands;
  };

  const reco::Candidate* findMother( const reco::Candidate* );
  int findGenMatch( double eta, double phi, double maxDeltaR,
		    std::vector<MatchStruct> matches );
  int findRecMatch( double eta, double phi, double maxdeltaR,
		    std::vector<MatchStruct> matches );
  
  // Monitor Elements (Histograms and ints)
  DQMStore* dbe_;

  std::vector <MonitorElement*> hPtPassGen ;
  std::vector <MonitorElement*> hEtaPassGen;
  std::vector <MonitorElement*> hPhiPassGen;
  std::vector <MonitorElement*> hPtPassRec ;
  std::vector <MonitorElement*> hEtaPassRec;
  std::vector <MonitorElement*> hPhiPassRec;

  MonitorElement *NumberOfEvents;
  MonitorElement *NumberOfL1Events;
  MonitorElement *NumberOfL1Orphans;
  MonitorElement *NumberOfHltOrphans;
  int theNumberOfEvents;
  int theNumberOfL1Events;
  int theNumberOfL1Orphans;
  int theNumberOfHltOrphans;
  std::string theRootFileName;


  // Facilities for writing a match ntuple
  bool    makeNtuple;
  TNtuple *theNtuple;
  TFile   *theFile;
  float   ntParams[18];

};
#endif

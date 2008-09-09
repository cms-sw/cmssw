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
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include <vector>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "TFile.h"
#include "TNtuple.h"


class HLTMuonGenericRate {
public:
  /// Constructor
  HLTMuonGenericRate(const edm::ParameterSet& pset, int triggerIndex);

  /// Destructor
  virtual ~HLTMuonGenericRate();

  // Operations

  void analyze(const edm::Event & event);
  void endJob();

  void BookHistograms() ;
  void WriteHistograms() ;
  void SetCurrentFolder( TString folder );
  MonitorElement* BookIt( TString name, TString title, 
			  int Nbins, float Min, float Max);

private:

  // Input from cfg file
  std::string folderName;
  edm::InputTag theL1CollectionLabel, theGenLabel, theRecoLabel;
  std::vector<edm::InputTag> theHLTCollectionLabels;
  double theL1ReferenceThreshold;
  double theHLTReferenceThreshold;
  std::vector<double> theNSigmas;
  unsigned int theNumberOfObjects;
  bool useMuonFromGenerator,useMuonFromReco;
  double theCrossSection;
  double theLuminosity;
  double thePtMin;
  double thePtMax;
  double theMinPtCut;
  double theMaxEtaCut;
  unsigned int theNbins;
  int thisEventWeight;
  int motherParticleId;

  // Struct for matching
  struct MatchStruct {
    HepMC::GenParticle* genCand;
    const reco::Track*  recCand;
    l1extra::L1MuonParticleRef                 l1Cand;
    std::vector<reco::RecoChargedCandidateRef> hltCands;
  };
  std::vector<MatchStruct> genMatches;
  std::vector<MatchStruct> recMatches;

  // Histograms
  DQMStore* dbe_;

  std::vector <MonitorElement*> hPtPassGen;
  std::vector <MonitorElement*> hEtaPassGen;
  std::vector <MonitorElement*> hPhiPassGen;
  std::vector <MonitorElement*> hPtPassRec;
  std::vector <MonitorElement*> hEtaPassRec;
  std::vector <MonitorElement*> hPhiPassRec;

  //  HepMC::GenEvent::particle_const_iterator theAssociatedGenPart;
  //  reco::TrackCollection::const_iterator theAssociatedRecoPart;
  const HepMC::GenEvent* theGenEvent;
  std::vector<const HepMC::GenParticle*> theGenMuons;
  std::vector<const reco::Track*> theRecMuons;

  int findGenMatch( double eta, double phi, double maxDeltaR );
  int findRecMatch( double eta, double phi, double maxdeltaR );

  MonitorElement *NumberOfEvents, *NumberOfL1Events;
  int theNumberOfEvents,theNumberOfL1Events;
  std::string theRootFileName;

  // ntuple
  TNtuple *nt;
  TFile *file;
  float params[18];

};
#endif

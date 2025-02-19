#ifndef CommonTools_ParticleFlow_TopProjectors
#define CommonTools_ParticleFlow_TopProjectors

#include "CommonTools/ParticleFlow/interface/TopProjectorAlgo.h"


#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidateFwd.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"

/* //TODO just for testing, remove this */
/* #include "DataFormats/TrackReco/interface/Track.h" */

/* #include "FWCore/Framework/interface/ESHandle.h" */

// #include "FWCore/MessageLogger/interface/MessageLogger.h"
/* #include "FWCore/Utilities/interface/Exception.h" */
/* #include "FWCore/Framework/interface/EventSetup.h" */



namespace ipf2pat {
  
  typedef pf2pat::TopProjectorAlgo<PFJet, PFCandidate> TPPFJetsOnPFCandidates;
  typedef pf2pat::TopProjectorAlgo<PFCandidate, PFCandidate> TPPFCandidatesOnPFCandidates;
  typedef pf2pat::TopProjectorAlgo<PileUpPFCandidate, PFCandidate> TPPileUpPFCandidatesOnPFCandidates;
  typedef pf2pat::TopProjectorAlgo<PFCandidate, PileUpPFCandidate> TPPFCandidatesOnPileUpPFCandidates;
  typedef pf2pat::TopProjectorAlgo<IsolatedPFCandidate, PFCandidate> TPIsolatedPFCandidatesOnPFCandidates;
  typedef pf2pat::TopProjectorAlgo<PFJet, PFCandidate> TPPFJetsOnPFCandidates;
  typedef pf2pat::TopProjectorAlgo<PFTau, PFJet> TPPFTausOnPFJets;
}

#endif

#include "PhysicsTools/PatAlgos/interface/PATCandMatcher.h"
#include "PhysicsTools/HepMCCandAlgos/interface/MCTruthPairSelector.h"
// #include "PhysicsTools/PatUtils/interface/PATMatchByDR.h"
#include "PhysicsTools/PatUtils/interface/PATMatchByDRDPt.h"
#include "PhysicsTools/PatUtils/interface/PATMatchLessByDPt.h"

typedef pat::PATCandMatcher<
  reco::CandidateView,
  reco::CandidateCollection,
  helpers::MCTruthPairSelector<reco::Candidate>,
  pat::PATMatchByDRDPt<reco::CandidateView::value_type,
		    reco::CandidateCollection::value_type>,
  pat::PATMatchLessByDPt<reco::CandidateView,
			 reco::CandidateCollection>
> PATMCMatcher;

// Alternative with pure deltaR matching (explicit) & sorting (default)
// typedef pat::PATCandMatcher<
//   reco::CandidateView,
//   reco::CandidateCollection,
//   helpers::MCTruthPairSelector<reco::Candidate>,
//   pat::PATMatchByDR<reco::CandidateView::value_type,
// 		    reco::CandidateCollection::value_type>
// > PATMCMatcher;

// or, using also the default (deltaR) for the match
// typedef pat::PATCandMatcher<
//   reco::CandidateView,
//   reco::CandidateCollection,
//   helpers::MCTruthPairSelector<reco::Candidate>
// > PATMCMatcher;


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( PATMCMatcher );

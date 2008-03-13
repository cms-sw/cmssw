//
// $Id: PATHLTMatcher.cc,v 1.1 2008/03/06 09:23:10 llista Exp $
//


#include "PhysicsTools/PatAlgos/plugins/PATCandMatcher.h"
#include "PhysicsTools/PatAlgos/plugins/PATHLTMatchSelector.h"
#include "PhysicsTools/PatUtils/interface/PATMatchByDRDPt.h"
#include "PhysicsTools/PatUtils/interface/PATMatchLessByDPt.h"

#include "DataFormats/Candidate/interface/Candidate.h"

// Match by deltaR and deltaPt, ranking by deltaR (default)
typedef pat::PATCandMatcher<
  reco::CandidateView,
  reco::CandidateCollection,
  pat::PATHLTMatchSelector<reco::CandidateView::value_type,
			   reco::CandidateCollection::value_type>,
  pat::PATMatchByDRDPt<reco::CandidateView::value_type,
		       reco::CandidateCollection::value_type>
> PATHLTMatcher;

// Alternative: match by deltaR and deltaPt, ranking by deltaPt
typedef pat::PATCandMatcher<
  reco::CandidateView,
  reco::CandidateCollection,
  pat::PATHLTMatchSelector<reco::CandidateView::value_type,
			   reco::CandidateCollection::value_type>,
  pat::PATMatchByDRDPt<reco::CandidateView::value_type,
		       reco::CandidateCollection::value_type>,
  pat::PATMatchLessByDPt<reco::CandidateView,
 			 reco::CandidateCollection >
> PATHLTMatcherByPt;

// Maybe, also a special treatment of jets is needed in the trigger matching!
// // JET Match by deltaR, ranking by deltaR (default)
// typedef pat::PATCandMatcher<
//   reco::CandidateView,
//   reco::GenJetCollection,
//   pat::PATMatchSelector<reco::CandidateView::value_type,
// 			reco::GenJetCollection::value_type>,
//   pat::PATMatchByDR<reco::CandidateView::value_type,
// 		    reco::CandidateView::value_type>
// > PATHLTJetMatcher;


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( PATHLTMatcher );
DEFINE_FWK_MODULE( PATHLTMatcherByPt );
// DEFINE_FWK_MODULE( PATHLTJetMatcher );


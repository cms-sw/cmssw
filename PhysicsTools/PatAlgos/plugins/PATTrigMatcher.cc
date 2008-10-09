//
// $Id: PATTrigMatcher.cc,v 1.2 2008/06/08 12:24:03 vadler Exp $
//
#include "PhysicsTools/PatAlgos/plugins/PATTrigMatchSelector.h"
#include "PhysicsTools/UtilAlgos/interface/PhysObjectMatcher.h"
#include "PhysicsTools/UtilAlgos/interface/MatchByDRDPt.h"
#include "PhysicsTools/UtilAlgos/interface/MatchLessByDPt.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/TriggerPrimitive.h"


// using namespace pat;
// using namespace reco;


/// Match by deltaR and deltaPt, ranking by deltaR (default)
typedef reco::PhysObjectMatcher<
  reco::CandidateView,
  pat::TriggerPrimitiveCollection,
  pat::PATTrigMatchSelector<reco::CandidateView::value_type,
			    pat::TriggerPrimitiveCollection::value_type>,
  reco::MatchByDRDPt<reco::CandidateView::value_type,
		     pat::TriggerPrimitiveCollection::value_type>
> PATTrigMatcher;

/// Alternative: match by deltaR and deltaPt, ranking by deltaPt
typedef reco::PhysObjectMatcher<
  reco::CandidateView,
  pat::TriggerPrimitiveCollection,
  pat::PATTrigMatchSelector<reco::CandidateView::value_type,
			    pat::TriggerPrimitiveCollection::value_type>,
  reco::MatchByDRDPt<reco::CandidateView::value_type,
		     pat::TriggerPrimitiveCollection::value_type>,
  reco::MatchLessByDPt<reco::CandidateView,
		       pat::TriggerPrimitiveCollection >
> PATTrigMatcherByPt;


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( PATTrigMatcher );
DEFINE_FWK_MODULE( PATTrigMatcherByPt );


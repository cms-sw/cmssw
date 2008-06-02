//
// $Id$
//


#include "PhysicsTools/PatAlgos/plugins/PATCandMatcher.h"
#include "PhysicsTools/PatAlgos/plugins/PATTrigMatchSelector.h"
#include "PhysicsTools/PatUtils/interface/PATMatchByDRDPt.h"
#include "PhysicsTools/PatUtils/interface/PATMatchLessByDPt.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/TriggerPrimitive.h"

// Match by deltaR and deltaPt, ranking by deltaR (default)
typedef pat::PATCandMatcher<
  reco::CandidateView,
  pat::TriggerPrimitiveCollection,
  pat::PATTrigMatchSelector<reco::CandidateView::value_type,
                           pat::TriggerPrimitiveCollection::value_type>,
  pat::PATMatchByDRDPt<reco::CandidateView::value_type,
                       pat::TriggerPrimitiveCollection::value_type>
> PATTrigMatcher;

// Alternative: match by deltaR and deltaPt, ranking by deltaPt
typedef pat::PATCandMatcher<
  reco::CandidateView,
  pat::TriggerPrimitiveCollection,
  pat::PATTrigMatchSelector<reco::CandidateView::value_type,
                           pat::TriggerPrimitiveCollection::value_type>,
  pat::PATMatchByDRDPt<reco::CandidateView::value_type,
                       pat::TriggerPrimitiveCollection::value_type>,
  pat::PATMatchLessByDPt<reco::CandidateView,
                         pat::TriggerPrimitiveCollection >
> PATTrigMatcherByPt;


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( PATTrigMatcher );
DEFINE_FWK_MODULE( PATTrigMatcherByPt );


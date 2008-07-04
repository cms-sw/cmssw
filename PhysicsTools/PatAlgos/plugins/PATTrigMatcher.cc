//
// $Id$
//


#include "PhysicsTools/PatAlgos/plugins/PATCandMatcher.h"
#include "PhysicsTools/PatAlgos/plugins/PATTrigMatchSelector.h"
#include "PhysicsTools/PatUtils/interface/PATMatchByDRDPt.h"
#include "PhysicsTools/PatUtils/interface/PATMatchLessByDPt.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/TriggerPrimitive.h"


using namespace pat;
using namespace reco;


/// Match by deltaR and deltaPt, ranking by deltaR (default)
typedef PATCandMatcher<
  CandidateView,
  TriggerPrimitiveCollection,
  PATTrigMatchSelector<CandidateView::value_type,
                       TriggerPrimitiveCollection::value_type>,
  PATMatchByDRDPt<CandidateView::value_type,
                  TriggerPrimitiveCollection::value_type>
> PATTrigMatcher;

/// Alternative: match by deltaR and deltaPt, ranking by deltaPt
typedef PATCandMatcher<
  CandidateView,
  TriggerPrimitiveCollection,
  PATTrigMatchSelector<CandidateView::value_type,
                       TriggerPrimitiveCollection::value_type>,
  PATMatchByDRDPt<CandidateView::value_type,
                  TriggerPrimitiveCollection::value_type>,
  PATMatchLessByDPt<CandidateView,
                    TriggerPrimitiveCollection >
> PATTrigMatcherByPt;


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( PATTrigMatcher );
DEFINE_FWK_MODULE( PATTrigMatcherByPt );


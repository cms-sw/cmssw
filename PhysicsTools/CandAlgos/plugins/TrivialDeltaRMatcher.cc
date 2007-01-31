#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/CandAlgos/interface/CandMatcher.h"
#include "PhysicsTools/UtilAlgos/interface/AnyPairSelector.h"

typedef reco::modules::CandMatcher<AnyPairSelector<reco::Candidate> > TrivialDeltaRMatcher;

DEFINE_FWK_MODULE( TrivialDeltaRMatcher );

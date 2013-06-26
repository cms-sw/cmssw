#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/CandAlgos/interface/CandMatcher.h"
#include "CommonTools/UtilAlgos/interface/AnyPairSelector.h"

typedef reco::modules::CandMatcher<AnyPairSelector> TrivialDeltaRMatcher;

DEFINE_FWK_MODULE( TrivialDeltaRMatcher );

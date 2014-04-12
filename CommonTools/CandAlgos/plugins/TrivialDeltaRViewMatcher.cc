#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/CandAlgos/interface/CandMatcher.h"
#include "CommonTools/UtilAlgos/interface/AnyPairSelector.h"

typedef 
  reco::modules::CandMatcher<
                   AnyPairSelector, 
                   reco::CandidateView
                 > TrivialDeltaRViewMatcher;

DEFINE_FWK_MODULE( TrivialDeltaRViewMatcher );

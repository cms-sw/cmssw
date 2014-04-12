#include "CommonTools/CandAlgos/interface/CompositeCandSelector.h"
#include "CommonTools/UtilAlgos/interface/AndPairSelector.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"

#include "FWCore/Framework/interface/MakerMacros.h"

typedef SingleObjectSelector<
          reco::CandidateView,
          CompositeCandSelector<
            AndPairSelector<
              StringCutObjectSelector<reco::Candidate>,
              StringCutObjectSelector<reco::Candidate> 
            >
          >
        > CompositeCandAndSelector;

DEFINE_FWK_MODULE(CompositeCandAndSelector);

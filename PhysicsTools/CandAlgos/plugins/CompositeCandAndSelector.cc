#include "PhysicsTools/CandAlgos/interface/CompositeCandSelector.h"
#include "PhysicsTools/UtilAlgos/interface/AndPairSelector.h"
#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"

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

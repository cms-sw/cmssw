#include "PhysicsTools/RecoAlgos/plugins/CandCommonVertexFitter.h"
#include "PhysicsTools/CandCombiners/interface/CandCombiner.h"
#include "PhysicsTools/UtilAlgos/interface/AndSelector.h"
#include "PhysicsTools/UtilAlgos/interface/MassRangeSelector.h"
#include "PhysicsTools/UtilAlgos/interface/ChargeSelector.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"

typedef reco::modules::CandCombiner<
          reco::CandidateCollection,
          AndSelector<
            ChargeSelector,
            MassRangeSelector
          >,
          reco::CandidateCollection,
          AnyPairSelector,
          combiner::helpers::ShallowClone,
          CandCommonVertexFitter<KalmanVertexFitter>
        > MassRangeAndChargeKalmanVertexFitCandShallowCloneCombiner;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( MassRangeAndChargeKalmanVertexFitCandShallowCloneCombiner );

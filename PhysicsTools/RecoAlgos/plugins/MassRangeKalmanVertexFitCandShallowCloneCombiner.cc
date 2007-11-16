#include "PhysicsTools/RecoAlgos/plugins/CandCommonVertexFitter.h"
#include "PhysicsTools/CandAlgos/interface/CandCombiner.h"
#include "PhysicsTools/UtilAlgos/interface/MassRangeSelector.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"

typedef reco::modules::CandCombiner<
          reco::CandidateCollection,
          MassRangeSelector,
          reco::CandidateCollection,
          AnyPairSelector,
          combiner::helpers::ShallowClone,
          CandCommonVertexFitter<KalmanVertexFitter>
        > MassRangeKalmanVertexFitCandShallowCloneCombiner;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( MassRangeKalmanVertexFitCandShallowCloneCombiner );

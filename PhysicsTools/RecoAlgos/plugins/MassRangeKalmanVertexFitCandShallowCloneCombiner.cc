#include "PhysicsTools/RecoAlgos/plugins/CandCommonVertexFitter.h"
#include "PhysicsTools/CandAlgos/interface/CandCombiner.h"
#include "PhysicsTools/UtilAlgos/interface/MassRangeSelector.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"

typedef reco::modules::CandCombiner<
          reco::CandidateCollection,
          MassRangeSelector,
          reco::VertexCompositeCandidateCollection,
          AnyPairSelector,
          combiner::helpers::ShallowClone,
          CandCommonVertexFitter<KalmanVertexFitter>
        > MassRangeKalmanVertexFitCandShallowCloneCombiner;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( MassRangeKalmanVertexFitCandShallowCloneCombiner );

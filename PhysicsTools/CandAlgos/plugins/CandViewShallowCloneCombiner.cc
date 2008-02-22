/* \class CandViewShallowCloneCombiner
 * 
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "PhysicsTools/CandAlgos/interface/CandCombiner.h"
#include "DataFormats/Candidate/interface/Candidate.h"

namespace reco {
  namespace modules {
    typedef CandCombiner<
              reco::CandidateView,
              StringCutObjectSelector<reco::Candidate>,
              reco::CompositeCandidateCollection,
              AnyPairSelector,
              combiner::helpers::ShallowClone
            > CandViewShallowCloneCombiner;

DEFINE_FWK_MODULE( CandViewShallowCloneCombiner );

  }
}

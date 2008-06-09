/* \class CandViewShallowCloneCombiner
 * 
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "PhysicsTools/CandAlgos/interface/CandCombiner.h"

namespace reco {
  namespace modules {
    typedef CandCombiner<
              StringCutObjectSelector<reco::Candidate>,
              AnyPairSelector,
              combiner::helpers::ShallowClonePtr
            > CandViewShallowClonePtrCombiner;

DEFINE_FWK_MODULE( CandViewShallowClonePtrCombiner );

  }
}

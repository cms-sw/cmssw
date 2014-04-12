/* \class CandViewShallowCloneCombiner
 * 
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "CommonTools/CandAlgos/interface/CandCombiner.h"

namespace reco {
  namespace modules {
    typedef CandCombiner<
              StringCutObjectSelector<reco::Candidate, true>,
              AnyPairSelector,
              combiner::helpers::ShallowClonePtr
            > CandViewShallowClonePtrCombiner;

DEFINE_FWK_MODULE( CandViewShallowClonePtrCombiner );

  }
}

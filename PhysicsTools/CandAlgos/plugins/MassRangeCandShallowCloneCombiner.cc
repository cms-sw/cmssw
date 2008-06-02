/* \class MassRangeCandShallowCloneCombiner
 * 
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/CandAlgos/interface/CandCombiner.h"
#include "PhysicsTools/UtilAlgos/interface/AndSelector.h"
#include "PhysicsTools/UtilAlgos/interface/MassRangeSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

namespace reco {
  namespace modules {

    typedef CandCombiner< 
              reco::CandidateCollection,
              MassRangeSelector,
              reco::CandidateCollection,
              AnyPairSelector,
              combiner::helpers::ShallowClone
            > MassRangeCandShallowCloneCombiner;

DEFINE_FWK_MODULE( MassRangeCandShallowCloneCombiner );

  }
}

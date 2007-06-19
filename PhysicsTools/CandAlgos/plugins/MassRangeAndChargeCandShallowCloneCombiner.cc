/* \class MassRangeAndChargeCandShallowCloneCombiner
 * 
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/CandAlgos/interface/CandCombiner.h"
#include "PhysicsTools/UtilAlgos/interface/AndSelector.h"
#include "PhysicsTools/UtilAlgos/interface/MassRangeSelector.h"
#include "PhysicsTools/UtilAlgos/interface/ChargeSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"


namespace reco {
  namespace modules {

    typedef CandCombiner<
              AndSelector<
                ChargeSelector,
                MassRangeSelector
              >,
              AnyPairSelector,
              combiner::helpers::ShallowClone
            > MassRangeAndChargeCandShallowCloneCombiner;

DEFINE_FWK_MODULE( MassRangeAndChargeCandShallowCloneCombiner );

  }
}

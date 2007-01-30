/* \class MassRangeAndChargeCandShallowCloneCombiner
 * 
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/CandAlgos/src/CandCombiner.h"
#include "PhysicsTools/Utilities/interface/AndSelector.h"
#include "PhysicsTools/Utilities/interface/MassRangeSelector.h"
#include "PhysicsTools/Utilities/interface/ChargeSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef CandCombiner<
          AndSelector<
            ChargeSelector<reco::Candidate>,
            MassRangeSelector<reco::Candidate>
          >,
          combiner::helpers::ShallowClone
        > MassRangeAndChargeCandShallowCloneCombiner;


DEFINE_FWK_MODULE( MassRangeAndChargeCandShallowCloneCombiner );

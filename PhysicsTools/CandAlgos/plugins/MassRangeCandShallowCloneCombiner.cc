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

typedef CandCombiner<
          MassRangeSelector<reco::Candidate>,
          combiner::helpers::ShallowClone
        > MassRangeCandShallowCloneCombiner;


DEFINE_FWK_MODULE( MassRangeCandShallowCloneCombiner );

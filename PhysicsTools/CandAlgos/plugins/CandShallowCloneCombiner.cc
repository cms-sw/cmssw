/* \class CandShallowCloneCombiner
 * 
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/CandAlgos/src/CandCombiner.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef CandCombiner<
          SingleObjectSelector<reco::Candidate>,
          combiner::helpers::ShallowClone
        > CandShallowCloneCombiner;

DEFINE_FWK_MODULE( CandShallowCloneCombiner );

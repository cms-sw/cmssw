/* \class CandShallowCloneCombiner
 * 
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "PhysicsTools/CandAlgos/interface/CandCombiner.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef CandCombiner<
          StringCutObjectSelector<reco::Candidate>,
          combiner::helpers::ShallowClone
        > CandShallowCloneCombiner;

DEFINE_FWK_MODULE( CandShallowCloneCombiner );

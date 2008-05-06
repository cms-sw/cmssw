/* \class DeltaRMinCandCombiner
 * 
 * Configurable Candidate Selector
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/DeltaRMinPairSelector.h"
#include "PhysicsTools/CandAlgos/interface/CandCombiner.h"

typedef reco::modules::CandCombiner<
          StringCutObjectSelector<reco::Candidate>,
          DeltaRMinPairSelector
        > DeltaRMinCandCombiner;

DEFINE_FWK_MODULE(DeltaRMinCandCombiner);

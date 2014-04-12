/* \class DeltaPhiMinCandCombiner
 * 
 * Configurable Candidate Selector
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/DeltaPhiMinPairSelector.h"
#include "CommonTools/CandAlgos/interface/CandCombiner.h"

typedef reco::modules::CandCombiner<
          StringCutObjectSelector<reco::Candidate, true>,
          DeltaPhiMinPairSelector
        > DeltaPhiMinCandCombiner;

DEFINE_FWK_MODULE(DeltaPhiMinCandCombiner);

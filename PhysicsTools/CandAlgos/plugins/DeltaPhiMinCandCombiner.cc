/* \class DeltaPhiMinCandCombiner
 * 
 * Configurable Candidate Selector
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/DeltaPhiMinPairSelector.h"
#include "PhysicsTools/CandAlgos/interface/CandCombiner.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"

typedef reco::modules::CandCombiner<
          StringCutObjectSelector<reco::Candidate>,
          reco::CompositeCandidateCollection,
          DeltaPhiMinPairSelector
        > DeltaPhiMinCandCombiner;

DEFINE_FWK_MODULE(DeltaPhiMinCandCombiner);

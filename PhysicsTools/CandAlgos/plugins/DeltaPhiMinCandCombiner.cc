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
#include "DataFormats/Candidate/interface/Candidate.h"

typedef reco::modules::CandCombiner<
  reco::CandidateCollection,
  StringCutObjectSelector<reco::Candidate>,
  reco::CandidateCollection,
  DeltaPhiMinPairSelector
> DeltaPhiMinCandCombiner;

DEFINE_FWK_MODULE( DeltaPhiMinCandCombiner );

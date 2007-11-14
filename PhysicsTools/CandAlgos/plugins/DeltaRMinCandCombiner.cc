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
#include "DataFormats/Candidate/interface/Candidate.h"

typedef reco::modules::CandCombiner<
  reco::CandidateCollection,
  StringCutObjectSelector<reco::Candidate>,
  reco::CandidateCollection, 
  DeltaRMinPairSelector
> DeltaRMinCandCombiner;

DEFINE_FWK_MODULE( DeltaRMinCandCombiner );

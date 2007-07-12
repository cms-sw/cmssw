/* \class PtMinCandRefToBaseCloneSelector
 * 
 * Candidate Selector based on a minimum pt cut.
 * Reads a edm::View<Candidate> as input
 * and saves a vector of clones.
 *
 * Usage:
 * 
 * module selectedCands = PtMinCandRefToBaseCloneSelector {
 *   InputTag src = myCollection
 *   double ptMin = 15.0
 * };
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/PtMinSelector.h"
#include "PhysicsTools/UtilAlgos/interface/RefSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef SingleObjectSelector<
          reco::CandidateBaseRefVector,
          RefSelector<PtMinSelector>,
          reco::CandidateCollection
        > PtMinCandRefToBaseCloneSelector;

DEFINE_FWK_MODULE( PtMinCandRefToBaseCloneSelector );

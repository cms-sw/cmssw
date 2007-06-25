/* \class PtMinCandViewCloneSelector
 * 
 * Candidate Selector based on a minimum pt cut.
 * Reads a edm::View<Candidate> as input
 * and saves a vector of clones.
 *
 * Usage:
 * 
 * module selectedCands = PtMinCandViewCloneSelector {
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
#include "DataFormats/Candidate/interface/Candidate.h"

typedef SingleObjectSelector<
          edm::View<reco::Candidate>,
          PtMinSelector<reco::Candidate>,
          reco::CandidateCollection
        > PtMinCandViewCloneSelector;

DEFINE_FWK_MODULE( PtMinCandViewCloneSelector );

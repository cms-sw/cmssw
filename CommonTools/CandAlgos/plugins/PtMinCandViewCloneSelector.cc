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
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/PtMinSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef SingleObjectSelector<
          edm::View<reco::Candidate>,
          PtMinSelector,
          reco::CandidateCollection
        > PtMinCandViewCloneSelector;

DEFINE_FWK_MODULE( PtMinCandViewCloneSelector );

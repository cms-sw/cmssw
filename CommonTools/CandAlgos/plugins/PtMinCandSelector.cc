/* \class PtMinCandSelector
 * 
 * Candidate Selector based on a minimum pt cut.
 * Usage:
 * 
 * module selectedCands = PtMinCandSelector {
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
          reco::CandidateCollection,
          PtMinSelector
        > PtMinCandSelector;

DEFINE_FWK_MODULE( PtMinCandSelector );

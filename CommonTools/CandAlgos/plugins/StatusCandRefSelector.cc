/* \class StatusCandRefSelector
 * 
 * Candidate Selector based on the status.
 * Save a vector of references to selected candidates.
 * Usage:
 * 
 * module leptons = StatusCandRefSelector {
 *   InputTag src = myCollection
 *   vint32 status =  { 1 }
 * };
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/StatusSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef SingleObjectSelector<
          reco::CandidateCollection,
          StatusSelector,
          reco::CandidateRefVector
        > StatusCandRefSelector;

DEFINE_FWK_MODULE( StatusCandRefSelector );

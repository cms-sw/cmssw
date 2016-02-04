/* \class StatusCandViewSelector
 * 
 * Candidate Selector based on the status.
 * The input collection is a View<Candidate>
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
          reco::CandidateView,
          StatusSelector
        > StatusCandViewSelector;

DEFINE_FWK_MODULE( StatusCandViewSelector );

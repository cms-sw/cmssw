/* \class CandViewCountFilter
 *
 * Filters events based on number of particle candidates
 * (or derrived classes)
 *
 * \author: Luca Lista, INFN
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "CommonTools/UtilAlgos/interface/ObjectCountFilter.h"

typedef ObjectCountFilter<reco::CandidateView>::type CandViewCountFilter;

DEFINE_FWK_MODULE( CandViewCountFilter );

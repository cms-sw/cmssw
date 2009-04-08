/* \class CandCountFilter
 *
 * Filters events based on number of particle candidates
 *
 * \author: Luca Lista, INFN
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "CommonTools/UtilAlgos/interface/ObjectCountFilter.h"

typedef ObjectCountFilter<reco::CandidateCollection>::type CandCountFilter;

DEFINE_FWK_MODULE( CandCountFilter );

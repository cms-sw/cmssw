/* \class CandCountFilter
 *
 * Filters events if at least N muons
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "CommonTools/UtilAlgos/interface/ObjectCountFilter.h"

 typedef ObjectCountFilter<
           reco::CandidateCollection
         > CandCountFilter;

DEFINE_FWK_MODULE( CandCountFilter );

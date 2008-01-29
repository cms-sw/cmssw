/* \class CandViewCountFilter
 *
 * Filters events if at least N candidates
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectCountFilter.h"

 typedef ObjectCountFilter<
           reco::CandidateView
         > CandViewCountFilter;

DEFINE_FWK_MODULE( CandViewCountFilter );

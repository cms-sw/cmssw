/* \class reco::modules::plugins::CandViewCombiner
 * 
 * Configurable Candidate Selector reading
 * a View<Candidate> as input
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "PhysicsTools/CandAlgos/interface/CandCombiner.h"
#include "DataFormats/Candidate/interface/Candidate.h"

DEFINE_SEAL_MODULE();

typedef reco::modules::CandCombiner<
                         reco::CandidateView,
                         StringCutObjectSelector<reco::Candidate>
                       > CandViewCombiner;
      
DEFINE_ANOTHER_FWK_MODULE( CandViewCombiner );

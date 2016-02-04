/* \class reco::modules::plugins::CandViewCombiner
 * 
 * Configurable Candidate Selector reading
 * a View<Candidate> as input
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "CommonTools/CandAlgos/interface/CandCombiner.h"

typedef reco::modules::CandCombiner<
                         StringCutObjectSelector<reco::Candidate, true>
                       > CandViewCombiner;
      
DEFINE_FWK_MODULE( CandViewCombiner );

/* \class reco::modules::CandSelector
 * 
 * Configurable Candidate Selector
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "PhysicsTools/CandAlgos/interface/CandCombiner.h"
#include "DataFormats/Candidate/interface/Candidate.h"

DEFINE_SEAL_MODULE();

namespace reco {
  namespace modules {
    namespace plugin {
      typedef reco::modules::CandCombiner<
	StringCutObjectSelector<reco::Candidate>
      > CandCombiner;
      
      DEFINE_ANOTHER_FWK_MODULE( CandCombiner );
    }
  }
}

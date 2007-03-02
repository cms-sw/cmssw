/* \class reco::modules::CandSelector
 * 
 * Configurable Candidate Selector
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

DEFINE_SEAL_MODULE();

namespace reco {
  namespace modules {
    typedef SingleObjectSelector<
              reco::CandidateCollection,
              StringCutObjectSelector<reco::Candidate>
            > CandSelector;

DEFINE_ANOTHER_FWK_MODULE( CandSelector );

  }
}

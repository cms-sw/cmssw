/* \class reco::modules::CandSelector
 * 
 * Configurable Candidate Selector
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

DEFINE_SEAL_MODULE();

namespace reco {
  namespace modules {
    typedef ObjectSelector<
              SingleElementCollectionSelector<
                reco::CandidateCollection,
                SingleObjectSelector<reco::Candidate>
              >
            > CandSelector;

DEFINE_ANOTHER_FWK_MODULE( CandSelector );

  }
}

/* \class MCTruthCompositeMatcher
 *
 * \author Luca Lista, INFN
 *
 */
#include "PhysicsTools/HepMCCandAlgos/interface/MCTruthCompositeMatcher.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"

namespace reco {
  namespace modules {
    typedef ::MCTruthCompositeMatcher<reco::CandidateCollection> MCTruthCompositeMatcher;

DEFINE_FWK_MODULE( MCTruthCompositeMatcher );

  }
}

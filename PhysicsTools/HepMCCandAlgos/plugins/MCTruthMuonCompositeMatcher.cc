/* \class MCTruthMuonCompositeMatcher
 *
 * \author Luca Lista, INFN
 *
 */
#include "PhysicsTools/HepMCCandAlgos/interface/MCTruthCompositeMatcher.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

namespace reco {
  namespace modules {
    typedef MCTruthCompositeMatcher<reco::MuonCollection> MCTruthMuonCompositeMatcher;

DEFINE_FWK_MODULE( MCTruthMuonCompositeMatcher );

  }
}

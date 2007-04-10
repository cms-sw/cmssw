/* \class MCTruthMuonCompositeMatcher
 *
 * \author Luca Lista, INFN
 *
 */
#include "PhysicsTools/HepMCCandAlgos/interface/MCTruthCompositeMatcher.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "FWCore/Framework/interface/MakerMacros.h"

namespace reco {
  namespace modules {
    typedef MCTruthCompositeMatcher<reco::PixelMatchGsfElectronCollection> MCTruthPixelMatchGsfElectronCompositeMatcher;

DEFINE_FWK_MODULE( MCTruthPixelMatchGsfElectronCompositeMatcher );

  }
}

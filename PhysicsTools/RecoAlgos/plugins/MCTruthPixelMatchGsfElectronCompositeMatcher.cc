/* \class MCTruthMuonCompositeMatcher
 *
 * \author Luca Lista, INFN
 *
 */
#include "PhysicsTools/HepMCCandAlgos/interface/MCTruthCompositeMatcher.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

namespace reco {
  namespace modules {
    typedef MCTruthCompositeMatcher<reco::GsfElectronCollection> MCTruthPixelMatchGsfElectronCompositeMatcher;

DEFINE_FWK_MODULE( MCTruthPixelMatchGsfElectronCompositeMatcher );

  }
}

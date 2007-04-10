/* \class MCTruthCaloJetCompositeMatcher
 *
 * \author Luca Lista, INFN
 *
 */
#include "PhysicsTools/HepMCCandAlgos/interface/MCTruthCompositeMatcher.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "FWCore/Framework/interface/MakerMacros.h"

namespace reco {
  namespace modules {
    typedef MCTruthCompositeMatcher<reco::CaloJetCollection> MCTruthCaloJetCompositeMatcher;

DEFINE_FWK_MODULE( MCTruthCaloJetCompositeMatcher );

  }
}

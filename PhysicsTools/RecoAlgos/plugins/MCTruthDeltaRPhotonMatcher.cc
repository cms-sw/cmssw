/* \class MCTruthDeltaRPhotonMatcher
 *
 * Producer fo simple MC truth match map
 * based on DeltaR
 *
 */
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "PhysicsTools/UtilAlgos/interface/Matcher.h"
#include "PhysicsTools/HepMCCandAlgos/interface/MCTruthPairSelector.h"

typedef reco::modules::Matcher<
          reco::PhotonCollection, 
          reco::CandidateCollection, 
          helpers::MCTruthPairSelector<reco::Photon> 
        > MCTruthDeltaRPhotonMatcher;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( MCTruthDeltaRPhotonMatcher );

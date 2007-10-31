/* \class MCTruthDeltaRPixelMatchGsfElectronMatcher
 *
 * Producer fo simple MC truth match map
 * based on DeltaR
 *
 */
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"
#include "PhysicsTools/UtilAlgos/interface/Matcher.h"
#include "PhysicsTools/HepMCCandAlgos/interface/MCTruthPairSelector.h"

typedef reco::modules::Matcher<
          reco::PixelMatchGsfElectronCollection, 
          reco::CandidateCollection, 
          helpers::MCTruthPairSelector<reco::PixelMatchGsfElectron> 
        > MCTruthDeltaRPixelMatchGsfElectronMatcher;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( MCTruthDeltaRPixelMatchGsfElectronMatcher );

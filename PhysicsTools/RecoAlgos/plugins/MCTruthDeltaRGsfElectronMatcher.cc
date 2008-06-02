/* \class MCTruthDeltaRGsfElectronMatcher
 *
 * Producer fo simple MC truth match map
 * based on DeltaR
 *
 */
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "PhysicsTools/UtilAlgos/interface/Matcher.h"
#include "PhysicsTools/HepMCCandAlgos/interface/MCTruthPairSelector.h"

typedef reco::modules::Matcher<
          reco::GsfElectronCollection, 
          reco::CandidateCollection, 
          helpers::MCTruthPairSelector<reco::GsfElectron> 
        > MCTruthDeltaRGsfElectronMatcher;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( MCTruthDeltaRGsfElectronMatcher );

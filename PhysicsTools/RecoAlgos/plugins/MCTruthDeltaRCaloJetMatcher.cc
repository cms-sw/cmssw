/* \class MCTruthDeltaRCaloJetMatcher
 *
 * Producer fo simple MC truth match map
 * based on DeltaR
 *
 */
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "PhysicsTools/UtilAlgos/interface/Matcher.h"
#include "PhysicsTools/HepMCCandAlgos/interface/MCTruthPairSelector.h"

typedef reco::modules::Matcher<
          reco::CaloJetCollection, 
          reco::CandidateCollection, 
          helpers::MCTruthPairSelector<reco::CaloJet> 
        > MCTruthDeltaRCaloJetMatcher;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( MCTruthDeltaRCaloJetMatcher );

/* \class MCTruthDeltaRViewMatcher
 *
 * Producer fo simple MC truth match map
 * based on DeltaR 
 *
 */
#include "CommonTools/CandAlgos/interface/CandMatcher.h"
#include "PhysicsTools/HepMCCandAlgos/interface/MCTruthPairSelector.h"

typedef reco::modules::CandMatcher<
          helpers::MCTruthPairSelector<reco::Candidate>,
          reco::CandidateView
        > MCTruthDeltaRViewMatcher;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( MCTruthDeltaRViewMatcher );


/* \class MCTruthDeltaRViewMatcher
 *
 * Producer fo simple MC truth match map
 * based on DeltaR 
 *
 */
#include "CommonTools/CandAlgos/interface/NewCandMatcher.h"
#include "PhysicsTools/HepMCCandAlgos/interface/MCTruthPairSelector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

typedef reco::modulesNew::CandMatcher<
          helpers::MCTruthPairSelector<reco::Candidate>,
          reco::CandidateView,
          reco::GenParticleCollection
        > MCTruthDeltaRMatcherNew;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( MCTruthDeltaRMatcherNew );


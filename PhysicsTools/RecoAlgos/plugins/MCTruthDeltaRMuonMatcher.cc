/* \class MCTruthDeltaRMuonMatcher
 *
 * Producer fo simple MC truth match map
 * based on DeltaR
 *
 */
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "PhysicsTools/UtilAlgos/interface/Matcher.h"
#include "PhysicsTools/HepMCCandAlgos/interface/MCTruthPairSelector.h"

typedef reco::modules::Matcher<
          reco::MuonCollection, 
          reco::CandidateCollection, 
          helpers::MCTruthPairSelector<reco::Muon> 
        > MCTruthDeltaRMuonMatcher;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( MCTruthDeltaRMuonMatcher );

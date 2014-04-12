/*
 * Produce Association<Tau> matches by Delta R using PhysObjMatcher
 * for PF and CaloTaus.
 *
 * Based on PhysicsTools/HepMCCandAlgos/plugins/MCTruthMatchers.cc
 *
 * Author: Evan K. Friis, UC Davis
 *
 */
#include "CommonTools/UtilAlgos/interface/PhysObjectMatcher.h"
#include "CommonTools/UtilAlgos/interface/MCMatchSelector.h"
#include "CommonTools/UtilAlgos/interface/MatchByDRDPt.h"
#include "CommonTools/UtilAlgos/interface/MatchLessByDPt.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauFwd.h"

// Tau Match by deltaR, ranking by deltaR
typedef reco::PhysObjectMatcher<
  reco::CandidateView,
  reco::PFTauCollection,
  reco::MCMatchSelector<reco::CandidateView::value_type,
			reco::PFTauCollection::value_type>,
  reco::MatchByDR<reco::CandidateView::value_type,
		  reco::CandidateView::value_type>
> PFTauMatcher;

typedef reco::PhysObjectMatcher<
  reco::CandidateView,
  reco::CaloTauCollection,
  reco::MCMatchSelector<reco::CandidateView::value_type,
			reco::CaloTauCollection::value_type>,
  reco::MatchByDR<reco::CandidateView::value_type,
		  reco::CandidateView::value_type>
> CaloTauMatcher;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFTauMatcher);
DEFINE_FWK_MODULE(CaloTauMatcher);

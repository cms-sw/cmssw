#include "CommonTools/UtilAlgos/interface/PhysObjectMatcher.h"
#include "CommonTools/UtilAlgos/interface/MCMatchSelector.h"
#include "CommonTools/UtilAlgos/interface/MatchByDRDPt.h"
#include "CommonTools/UtilAlgos/interface/MatchLessByDPt.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"

// Match by deltaR and deltaPt, ranking by deltaR (default)
typedef reco::PhysObjectMatcher<
  reco::CandidateView,
  reco::GenParticleCollection,
  reco::MCMatchSelector<reco::CandidateView::value_type,
			reco::GenParticleCollection::value_type>,
  reco::MatchByDRDPt<reco::CandidateView::value_type,
		     reco::GenParticleCollection::value_type>
> MCMatcher;

// Alternative: match by deltaR and deltaPt, ranking by deltaPt
typedef reco::PhysObjectMatcher<
  reco::CandidateView,
  reco::GenParticleCollection,
  reco::MCMatchSelector<reco::CandidateView::value_type,
			reco::GenParticleCollection::value_type>,
  reco::MatchByDRDPt<reco::CandidateView::value_type,
		     reco::GenParticleCollection::value_type>,
  reco::MatchLessByDPt<reco::CandidateView,
			 reco::GenParticleCollection>
> MCMatcherByPt;

// JET Match by deltaR, ranking by deltaR (default)
typedef reco::PhysObjectMatcher<
  reco::CandidateView,
  reco::GenJetCollection,
  reco::MCMatchSelector<reco::CandidateView::value_type,
			reco::GenJetCollection::value_type>,
  reco::MatchByDR<reco::CandidateView::value_type,
		  reco::CandidateView::value_type>
> GenJetMatcher;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MCMatcher);
DEFINE_FWK_MODULE(MCMatcherByPt);
DEFINE_FWK_MODULE(GenJetMatcher);

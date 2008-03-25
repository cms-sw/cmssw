#include "PhysicsTools/PatAlgos/plugins/PATCandMatcher.h"
#include "PhysicsTools/PatAlgos/interface/PATMatchSelector.h"
#include "PhysicsTools/PatUtils/interface/PATMatchByDRDPt.h"
#include "PhysicsTools/PatUtils/interface/PATMatchLessByDPt.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJetfwd.h"
#include "DataFormats/JetReco/interface/GenJet.h"

// Match by deltaR and deltaPt, ranking by deltaR (default)
typedef pat::PATCandMatcher<
  reco::CandidateView,
  reco::GenParticleCollection,
  pat::PATMatchSelector<reco::CandidateView::value_type,
			reco::GenParticleCollection::value_type>,
  pat::PATMatchByDRDPt<reco::CandidateView::value_type,
		       reco::GenParticleCollection::value_type>
> PATMCMatcher;

// Alternative: match by deltaR and deltaPt, ranking by deltaPt
typedef pat::PATCandMatcher<
  reco::CandidateView,
  reco::GenParticleCollection,
  pat::PATMatchSelector<reco::CandidateView::value_type,
			reco::GenParticleCollection::value_type>,
  pat::PATMatchByDRDPt<reco::CandidateView::value_type,
		       reco::GenParticleCollection::value_type>,
  pat::PATMatchLessByDPt<reco::CandidateView,
 			 reco::GenParticleCollection>
> PATMCMatcherByPt;

// JET Match by deltaR, ranking by deltaR (default)
typedef pat::PATCandMatcher<
  reco::CandidateView,
  reco::GenJetCollection,
  pat::PATMatchSelector<reco::CandidateView::value_type,
			reco::GenJetCollection::value_type>,
  pat::PATMatchByDR<reco::CandidateView::value_type,
		    reco::CandidateView::value_type>
> PATGenJetMatcher;




#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( PATMCMatcher );
DEFINE_FWK_MODULE( PATMCMatcherByPt );
DEFINE_FWK_MODULE( PATGenJetMatcher );


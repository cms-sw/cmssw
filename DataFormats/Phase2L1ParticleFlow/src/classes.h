#include "Rtypes.h"

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Phase2L1ParticleFlow/interface/PFCluster.h"
#include "DataFormats/Phase2L1ParticleFlow/interface/PFTrack.h"
#include "DataFormats/Phase2L1ParticleFlow/interface/PFCandidate.h"
#include "DataFormats/Phase2L1ParticleFlow/interface/PFJet.h"
#include "DataFormats/Phase2L1ParticleFlow/interface/PFTau.h"


namespace DataFormats_Phase2L1ParticleFlow {
  struct dictionary {
    l1t::PFCluster l1clus;
    l1t::PFTrack l1trk;
    l1t::PFCandidate l1pfc;
    l1t::PFJet l1pfj;
    l1t::PFTau l1pft;

    l1t::PFClusterCollection l1PFClusterCollection;
    l1t::PFTrackCollection l1PFTrackCollection;
    l1t::PFCandidateCollection l1PFCandidateCollection;
    l1t::PFJetCollection l1PFJetCollection;
    l1t::PFTauCollection l1PFTauCollection;

    edm::Wrapper<l1t::PFClusterCollection>   wl1PFClusterCollection;
    edm::Wrapper<l1t::PFTrackCollection>   wl1PFTrackCollection;
    edm::Wrapper<l1t::PFCandidateCollection>   wl1PFCandidateCollection;
    edm::Wrapper<l1t::PFJetCollection>   wl1PFJetCollection;
    edm::Wrapper<l1t::PFTauCollection>   wl1PFTauCollection;

    edm::Ref<l1t::PFClusterCollection> l1PFClusterRef;
    edm::Ref<l1t::PFTrackCollection> l1PFTrackRef;
    edm::Ref<l1t::PFCandidateCollection> l1PFCandidateRef;
    edm::Ref<l1t::PFJetCollection> l1PFJetRef;
    edm::Ref<l1t::PFTauCollection> l1PFTauRef;

    edm::RefVector<l1t::PFClusterCollection> l1PFClusterRefVector;
    edm::RefVector<l1t::PFTrackCollection> l1PFTrackRefVector;
    edm::RefVector<l1t::PFCandidateCollection> l1PFCandidateRefVector;
    edm::RefVector<l1t::PFJetCollection> l1PFJetRefVector;
    edm::RefVector<l1t::PFTauCollection> l1PFTauRefVector;
  };
}


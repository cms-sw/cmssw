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

    edm::Wrapper<l1t::PFClusterCollection>   wl1PFClusterCollection;
    edm::Wrapper<l1t::PFTrackCollection>   wl1PFTrackCollection;
    edm::Wrapper<l1t::PFCandidateCollection>   wl1PFCandidateCollection;
    edm::Wrapper<l1t::PFJetCollection>   wl1PFJetCollection;
    edm::Wrapper<l1t::PFTauCollection>   wl1PFTauCollection;

  };
}


#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"

namespace DataFormats_PFClusterMatch {
  struct dictionary {
    edm::ValueMap<edm::Ref<std::vector<reco::GenParticle>,reco::GenParticle,edm::refhelper::FindUsingAdvance<std::vector<reco::GenParticle>,reco::GenParticle> > > vmg2;
    edm::Wrapper<edm::ValueMap<edm::Ref<std::vector<reco::GenParticle>,reco::GenParticle,edm::refhelper::FindUsingAdvance<std::vector<reco::GenParticle>,reco::GenParticle> > > > wvmg2;
  };
}

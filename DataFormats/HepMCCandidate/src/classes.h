#include "DataFormats/HepMCCandidate/interface/HepMCCandidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {
    // this is needed to fix a "missing dictionary" problem.
    // see the following thread:
    //   https://hypernews.cern.ch/HyperNews/CMS/get/physTools/63.html
    edm::Wrapper<reco::HepMCCandidate> w1;

    edm::Wrapper<reco::GenParticleCandidate> w2;
    reco::GenParticleCandidateCollection c3;
    reco::GenParticleCandidateRef r1;
    reco::GenParticleCandidateRefVector rv1;
    reco::GenParticleCandidateRefProd rp1;
    edm::Wrapper<reco::GenParticleCandidateCollection> w3;
    
  }
}

#include "DataFormats/V0Candidate/interface/V0Candidate.h"

namespace {
  struct dictionary {
    std::vector<reco::V0Candidate> v01;
    edm::Wrapper<std::vector<reco::V0Candidate> > wv01;
  };
}

#include "DataFormats/HepMCCandidate/interface/HepMCCandidate.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {
    // this is needed to fix a "missing dictionary" problem.
    // see the following thread:
    //   https://hypernews.cern.ch/HyperNews/CMS/get/physTools/63.html
    edm::Wrapper<reco::HepMCCandidate> w1;
  }
}

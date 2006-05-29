#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include <vector>

namespace {
  namespace {
    std::vector<reco::Muon> v1;
    edm::Wrapper<std::vector<reco::Muon> > c1;
    edm::Ref<std::vector<reco::Muon> > r1;
    edm::RefProd<std::vector<reco::Muon> > rp1;
    edm::RefVector<std::vector<reco::Muon> > rv1;

    edm::reftobase::Holder<reco::Candidate, reco::MuonRef> rb1;
  }
}

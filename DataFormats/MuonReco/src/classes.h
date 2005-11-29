#include "FWCore/EDProduct/interface/Wrapper.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include <vector>

namespace {
  namespace {
    std::vector<reco::Muon> v1;
    edm::Wrapper<std::vector<reco::Muon> > c1;
  }
}

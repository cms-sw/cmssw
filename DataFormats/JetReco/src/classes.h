#include "DataFormats/JetReco/interface/CaloJet.h"
#include "FWCore/EDProduct/interface/Wrapper.h"

namespace {
  namespace {
    std::vector<CaloTower> vc1;
    edm::SortedCollection<CaloTower> c1;

    edm::RefVector<edm::SortedCollection<CaloTower> > r1;

    std::vector<reco::CaloJet> v1;
    edm::Wrapper<std::vector<reco::CaloJet> > w1;
  }
}

#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "FWCore/EDProduct/interface/Wrapper.h"

namespace {
  namespace {
    std::vector<CaloTower> a;
    edm::SortedCollection<CaloTower> aa;

    edm::Wrapper<edm::SortedCollection<CaloTower> > aaa;
  }
}

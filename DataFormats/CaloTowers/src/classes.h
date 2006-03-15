#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {
    std::vector<CaloTower> a;
    edm::SortedCollection<CaloTower> aa;

    edm::Wrapper<edm::SortedCollection<CaloTower> > aaa;
  }
}

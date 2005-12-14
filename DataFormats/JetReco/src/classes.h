#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "FWCore/EDProduct/interface/Wrapper.h"
#include <vector>

namespace {
  namespace {
    ROOT::Math::PxPyPzE4D<Double32_t> p4;
    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<Double32_t> > lv;

    std::vector<CaloTower> vc1;
    edm::SortedCollection<CaloTower> c1;

    edm::RefVector<edm::SortedCollection<CaloTower> > r1;

    std::vector<reco::CaloJet> v1;
    edm::Wrapper<std::vector<reco::CaloJet> > w1;
  }
}

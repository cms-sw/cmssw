/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// ////////////////////////////////////////

#include "DataFormats/Common/interface/Wrapper.h"


/*********************/
/** L1 CALO TRIGGER **/
/*********************/

#include "DataFormats/Phase2L1CaloTrig/interface/L1EGCrystalCluster.h"
#include "DataFormats/Phase2L1CaloTrig/interface/L1CaloTower.h"


namespace {
  namespace {


    l1slhc::L1EGCrystalCluster                       egcrystalcluster;
    std::vector<l1slhc::L1EGCrystalCluster>         l1egcrystalclustervec;
    l1slhc::L1EGCrystalClusterCollection            l1egcrystalclustercoll;
    edm::Wrapper<l1slhc::L1EGCrystalClusterCollection>   wl1egcrystalclustercoll;

    L1CaloTower                                     l1CaloTower;
    std::vector<L1CaloTower>                        l1CaloTowervec;
    L1CaloTowerCollection                           l1CaloTowercoll;
    edm::Wrapper<L1CaloTowerCollection>             wl1CaloTowercoll;

  }
}


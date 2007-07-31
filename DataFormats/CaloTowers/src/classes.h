#include <boost/cstdint.hpp> 
#include "Math/Cartesian3D.h" 
#include "Math/Polar3D.h" 
#include "Math/CylindricalEta3D.h" 
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {
    std::vector<CaloTower> v1;
    CaloTowerCollection c1;
    CaloTowerRef r1;
    CaloTowersRef rr1;
    CaloTowerRefs rrr1;
    edm::Wrapper<CaloTowerCollection> w1;
  }
}

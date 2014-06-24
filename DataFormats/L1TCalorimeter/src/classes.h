#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "Rtypes.h"
#include "Math/Cartesian3D.h"
#include "Math/Polar3D.h"
#include "Math/CylindricalEta3D.h"
#include "Math/PxPyPzE4D.h"
#include <boost/cstdint.hpp>
#include "DataFormats/L1Trigger/interface/BXVector.h"

#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"
#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
#include "DataFormats/L1TCalorimeter/interface/CaloCluster.h"

namespace DataFormats_L1TCalorimeter {
  struct dictionary {

    l1t::CaloRegionBxCollection  caloRegionBxColl;
    l1t::CaloEmCandBxCollection  caloEmCandBxColl;
    l1t::CaloTowerBxCollection   caloTowerBxColl;
    l1t::CaloClusterBxCollection caloClusterBxColl;

    edm::Wrapper<l1t::CaloRegionBxCollection>  w_caloRegionBxColl;
    edm::Wrapper<l1t::CaloEmCandBxCollection>  w_caloEmCandBxColl;
    edm::Wrapper<l1t::CaloTowerBxCollection>   w_caloTowerBxColl;
    edm::Wrapper<l1t::CaloClusterBxCollection> w_caloClusterBxColl;

  };
}

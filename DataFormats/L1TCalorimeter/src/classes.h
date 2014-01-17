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

namespace {
  struct dictionary {

    l1t::CaloRegionBxCollection caloRegionBxColl;
    l1t::CaloEmCandBxCollection caloEmCandBxColl;

    edm::Wrapper<l1t::CaloRegionBxCollection> w_caloRegionBxColl;
    edm::Wrapper<l1t::CaloEmCandBxCollection> w_caloEmCandBxColl;
  };
}

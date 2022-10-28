#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

BTLDetId BTLDetId::geographicalId(CrysLayout lay) const {
  // For tracking geometry navigation

  if (lay == CrysLayout::barphiflat) {
    // barphiflat: count modules in a rod, combining all types
    return BTLDetId(mtdSide(), mtdRR(), module() + kModulePerTypeBarPhiFlat * (modType() - 1), 0, 1);
  } else if (lay == CrysLayout::v2) {
    // v2: set number of crystals to 17 to distinguish from crystal BTLDetId
    return BTLDetId(mtdSide(), mtdRR(), runit(), module(), modType(), kCrystalsPerModuleV2 + 1);
  }

  return 0;
}

#include <iomanip>

std::ostream& operator<<(std::ostream& os, const BTLDetId& id) {
  os << (MTDDetId&)id;
  os << " BTL " << std::endl
     << " Side        : " << id.mtdSide() << std::endl
     << " Rod         : " << id.mtdRR() << std::endl
     << " Crystal type: " << id.modType() << std::endl
     << " Readout unit: " << id.runit() << std::endl
     << " Module      : " << id.module() << std::endl
     << " Crystal     : " << id.crystal() << std::endl;
  return os;
}

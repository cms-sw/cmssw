#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

BTLDetId BTLDetId::geographicalId(CrysLayout lay) const {
  // For tracking geometry navigation

  if (lay == CrysLayout::v2 || lay == CrysLayout::v3) {
    // v2: set number of crystals to 17 to distinguish from crystal BTLDetId
    // v3: set number of crystals to 17 to distinguish from crystal BTLDetId, build V2-like type and RU number as in BTLNumberingScheme
    return BTLDetId(mtdSide(), mtdRR(), runitByType(), module(), modType(), kCrystalsPerModuleV2 + 1, true);
  }
  if (lay == CrysLayout::v4) {
    // v4: identical to v3, needed to update BTLDetId format and corresponding numbering scheme
    return BTLDetId(mtdSide(), mtdRR(), runit(), dmodule(), smodule(), kCrystalsPerModuleV2);
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
     << " Readout unit by type: " << id.runitByType() << std::endl
     << " Detector Module: " << id.dmodule() << std::endl
     << " Sensor Module: " << id.smodule() << std::endl
     << " Module      : " << id.module() << std::endl
     << " Crystal     : " << id.crystal() << std::endl
     << " Crystal in DB: " << id.crystalConsDB() << std::endl;
  return os;
}

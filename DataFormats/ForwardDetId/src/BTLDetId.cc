#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

BTLDetId BTLDetId::geographicalId(CrysLayout lay) const {
  // For tracking geometry navigation
  // v2,v3: set number of crystals to 17 to distinguish from crystal BTLDetId
  // v1: obsolete and not supported

  if (lay == CrysLayout::v2 || lay == CrysLayout::v3) {
    return BTLDetId(mtdSide(), mtdRR(), runit(), dmodule(), smodule(), kCrystalsPerModuleV2);
  } else {
    edm::LogWarning("MTDGeom") << "CrysLayout could only be v2 or v3";
  }

  return 0;
}

#include <iomanip>

std::ostream& operator<<(std::ostream& os, const BTLDetId& id) {
  os << (MTDDetId&)id;
  os << " BTL " << std::endl
     << " Side           : " << id.mtdSide() << std::endl
     << " Rod            : " << id.mtdRR() << std::endl
     << " Crystal type   : " << id.modType() << std::endl  // crystal type in v1 geometry scheme
     << " Runit by Type  : " << id.runitByType() << std::endl
     << " Readout unit   : " << id.runit() << std::endl
     << " Detector module: " << id.dmodule() << std::endl
     << " Sensor module  : " << id.smodule() << std::endl
     << " Module         : " << id.module() << std::endl
     << " Crystal        : " << id.crystal() << std::endl
     << " Crystal in ConsDB: " << id.crystalConsDB() << std::endl;
  return os;
}

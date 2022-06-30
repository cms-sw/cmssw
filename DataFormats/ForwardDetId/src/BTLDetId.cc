#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

int BTLDetId::modulesPerType(CrysLayout lay) const {
  if (lay == CrysLayout::barphiflat) {
    return kModulePerTypeBarPhiFlat;
  }
  return 0;
}

BTLDetId BTLDetId::geographicalId(CrysLayout lay) const {
  // reorganize the modules to count from 0 to 54
  //    (0 to 42 in the case of BarZflat geometry)
  // remove module type
  // remove crystal index

  int boundRef = modulesPerType(lay);

  return BTLDetId(mtdSide(), mtdRR(), module() + boundRef * (modType() - 1), 0, 1);
}

#include <iomanip>

std::ostream& operator<<(std::ostream& os, const BTLDetId& id) {
  os << (MTDDetId&)id;
  os << " BTL " << std::endl
     << " Side        : " << id.mtdSide() << std::endl
     << " Rod         : " << id.mtdRR() << std::endl
     << " Module      : " << id.module() << std::endl
     << " Crystal type: " << id.modType() << std::endl
     << " Crystal     : " << id.crystal() << std::endl;
  return os;
}

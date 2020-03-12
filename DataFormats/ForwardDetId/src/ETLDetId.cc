#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include <iomanip>

ETLDetId ETLDetId::geographicalId() const {
  // strip off module type info
  return ETLDetId(mtdSide(), mtdRR(), module(), 0);
}

std::ostream& operator<<(std::ostream& os, const ETLDetId& id) {
  os << (MTDDetId&)id;
  os << " ETL " << std::endl
     << " Side        : " << id.mtdSide() << std::endl
     << " Ring        : " << id.mtdRR() << "    "
     << " Disc/Side/Quarter = " << id.nDisc() << " " << id.discSide() << " " << id.quarter() << std::endl
     << " Module      : " << id.module() << std::endl
     << " Module type : " << id.modType() << std::endl;
  return os;
}

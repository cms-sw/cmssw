#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include <iomanip>

std::ostream& operator<<(std::ostream& os, const ETLDetId& id) {
  os << (MTDDetId&)id;
  os << " ETL " << std::endl
     << " Side        : " << id.mtdSide() << std::endl
     << " Ring        : " << id.mtdRR() << "    "
     << " Disc/Side/Sector = " << id.nDisc() << " " << id.discSide() << " " << id.sector() << std::endl
     << " Module      : " << id.module() << std::endl
     << " Module type : " << id.modType() << std::endl;
  return os;
}

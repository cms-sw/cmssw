#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include <iomanip>

std::ostream& operator<<(std::ostream& os, const ETLDetId& id) {
  os << (MTDDetId&)id;
  os << " ETL " << std::endl
     << " Side        : " << id.mtdSide() << std::endl
     << " Disc        : " << id.nDisc() << std::endl
     << " Side        : " << id.discSide() << std::endl
     << " Sector      : " << id.sector() << std::endl
     << " Module      : " << id.module() << std::endl
     << " Module type : " << id.modType() << std::endl
     << " Sensor      : " << id.sensor() << std::endl;
  return os;
}

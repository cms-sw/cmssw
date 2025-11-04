#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include <iomanip>

std::ostream& operator<<(std::ostream& os, const ETLDetId& id) {
  os << (MTDDetId&)id;
  os << " ETL " << std::endl
     << " Side                  : " << id.mtdSide() << std::endl
     << " Disc                  : " << id.nDisc() << std::endl
     << " Disc Side             : " << id.discSide() << std::endl
     << " Sector                : " << id.sector() << std::endl
     << " Service Hybrid Type   : " << id.servType() << std::endl
     << " Service Hybrid Number : " << id.servCopy() << std::endl
     << " Module Number         : " << id.module() << std::endl
     << " Module Type           : " << id.modType() << std::endl
     << " Sensor                : " << id.sensor() << std::endl;
  return os;
}

std::stringstream printETLDetId(uint32_t detId) {
  std::stringstream ss;
  ETLDetId thisId(detId);
  if (thisId.det() != DetId::Forward || thisId.subdetId() != MTDDetId::FastTime ||
      thisId.mtdSubDetector() != MTDDetId::ETL) {
    ss << "DetId " << detId << " not an ETLDetId!";
  }

  ss << " ETLDetId " << thisId.rawId() << " side = " << std::setw(2) << thisId.mtdSide()
     << " disc/face/sec = " << std::setw(4) << thisId.nDisc() << " " << std::setw(4) << thisId.discSide() << " "
     << std::setw(4) << thisId.sector() << " shtyp/sh = " << std::setw(4) << thisId.servType() << " " << std::setw(4)
     << thisId.servCopy() << " mod/typ/sens = " << std::setw(4) << thisId.module() << " " << std::setw(4)
     << thisId.modType() << " " << std::setw(4) << thisId.sensor();
  return ss;
}

#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include <ostream>
#include <iostream>

const HGCSiliconDetId HGCSiliconDetId::Undefined(DetId::HGCalEE, 0, 0, 0, 0, 0, 0, 0);

std::ostream& operator<<(std::ostream& s, const HGCSiliconDetId& id) {
  return s << " HGCSiliconDetId::EE:HE= " << id.isEE() << ":" << id.isHE() << " type= " << id.type()
           << " z= " << id.zside() << " layer= " << id.layer() << " wafer(u,v:x,y)= (" << id.waferU() << ","
           << id.waferV() << ":" << id.waferX() << "," << id.waferY() << ")"
           << " cell(u,v:x,y)= (" << id.cellU() << "," << id.cellV() << ":" << id.cellX() << "," << id.cellY() << ")";
}

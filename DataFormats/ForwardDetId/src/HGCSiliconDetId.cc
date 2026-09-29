#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include <ostream>
#include <iostream>

const HGCSiliconDetId HGCSiliconDetId::Undefined(DetId::HGCalEE, 0, 0, 0, 0, 0, 0, 0);

std::ostream& operator<<(std::ostream& s, const HGCSiliconDetId& id) {
  return s << " HGCSiliconDetId:: " << id.detType() << " type= " << id.waferTypeX() << " z= " << id.zside()
           << " layer= " << id.layer() << " wafer(u,v)= (" << id.waferU() << "," << id.waferV() << ")"
           << " cell(u,v)= (" << id.cellU() << "," << id.cellV() << ")";
}

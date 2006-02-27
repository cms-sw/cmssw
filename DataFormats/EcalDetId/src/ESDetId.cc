#include "DataFormats/EcalDetId/interface/ESDetId.h"


ESDetId::ESDetId() : DetId() {
}
  
ESDetId::ESDetId(uint32_t rawid) : DetId(rawid) {
}
  
ESDetId::ESDetId(int strip, int ixs, int iys, int plane, int iz) : DetId(Ecal,EcalPreshower) {
  id_ |=
    (strip&0x3F) |
    ((ixs&0x3F)<<6) |
    ((iys&0x3F)<<12) |
    (((plane-1)&0x1)<<18) |
    ((iz>0)?(1<<19):(0));
}
  
ESDetId::ESDetId(const DetId& gen) {
  if (gen.det()!=Ecal || gen.subdetId()!=EcalPreshower) {
    throw new std::exception();
  }
  id_=gen.rawId();
}
  
ESDetId& ESDetId::operator=(const DetId& gen) {
  if (gen.det()!=Ecal || gen.subdetId()!=EcalPreshower) {
    throw new std::exception();
  }
  id_=gen.rawId();
  return *this;
}
  
int ESDetId::hashedIndex() const {
  // TODO: more efficient index!
  return id_&0xFFFFFF;
}
  
std::ostream& operator<<(std::ostream& s,const ESDetId& id) {
  return s << "(ES z=" << id.zside() << "  plane " << id.plane() << " " <<
    id.six() << ':' << id.siy() << " " << id.strip() << ')';
}

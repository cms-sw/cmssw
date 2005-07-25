#include "DataFormats/EcalDetId/interface/ESDetId.h"

namespace cms
{
  ESDetId::ESDetId() : DetId() {
  }
  
  ESDetId::ESDetId(uint32_t rawid) : DetId(rawid) {
  }
  
  ESDetId::ESDetId(int strip, int ixs, int iys, int plane, int iz) : DetId(Ecal,EcalPreshower) {
    id_ |=
      (strip&0x1F) |
      ((ixs&0x3F)<<5) |
      ((iys&0x3F)<<11) |
      ((plane&0x1)<<17) |
      ((iz>0)?(1<<18):(0));
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
    return s << "(ES z=" << id.zside() << "  plane " << id.plane() << 
      id.six() << ',' << id.siy() << ':' << id.strip() << ')';
  }
  
}

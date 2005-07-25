#include "DataFormats/EcalDetId/interface/EEDetId.h"

namespace cms
{
  EEDetId::EEDetId() : DetId() {
  }
  EEDetId::EEDetId(uint32_t rawid) : DetId(rawid) {
  }
  EEDetId::EEDetId(int ix, int iy, int iz) : DetId(Ecal,EcalEndcap) {
    // checking?
    id_|=(iy&0x7f)|((ix&0x7f)<<7)|((iz>0)?(0x4000):(0));
  }
  
  EEDetId::EEDetId(const DetId& gen) {
    if (gen.det()!=Ecal || gen.subdetId()!=EcalEndcap) {
      throw new std::exception();
    }
    id_=gen.rawId();
  }
  
  EEDetId& EEDetId::operator=(const DetId& gen) {
    if (gen.det()!=Ecal || gen.subdetId()!=EcalEndcap) {
      throw new std::exception();
    }
    id_=gen.rawId();
    return *this;
  }
  
  int EEDetId::hashedIndex() const {
    return ((zside()>0)?(IX_MAX*IY_MAX):(0))+(iy()-1)*IX_MAX+(ix()-1);
  }
  
  std::ostream& operator<<(std::ostream& s,const EEDetId& id) {
    return s << "(EE" << ((id.zside()>0)?("+ "):("- ")) << id.ix() << ',' << id.iy() << ')';
  }
  
}

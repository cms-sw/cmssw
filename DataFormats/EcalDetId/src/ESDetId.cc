#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "FWCore/Utilities/interface/Exception.h"

ESDetId::ESDetId() : DetId() {
}
  
ESDetId::ESDetId(uint32_t rawid) : DetId(rawid) {
}
  
ESDetId::ESDetId(int strip, int ixs, int iys, int plane, int iz) : DetId(Ecal,EcalPreshower) {
  if ( !validDetId( strip, ixs, iys, plane, iz) )
    throw cms::Exception("InvalidDetId") << "ESDetId:  Cannot create object.  Indexes out of bounds \n" 
                                         << " strip = " << strip << " x = " << ixs << " y = " << iys << "\n" 
                                         << " plane = " << plane << " z = " << iz;

   id_ |=
    (strip&0x3F) |
    ((ixs&0x3F)<<6) |
    ((iys&0x3F)<<12) |
    (((plane-1)&0x1)<<18) |
    ((iz>0)?(1<<19):(0));
}
  
ESDetId::ESDetId(const DetId& gen) {
  if (!gen.null() && ( gen.det()!=Ecal || gen.subdetId()!=EcalPreshower )) {
    throw cms::Exception("InvalidDetId");
  }
  id_=gen.rawId();
}

bool ESDetId::validDetId(int istrip, int ixs, int iys, int iplane, int iz) {

  bool valid = true;
  if ((istrip<ISTRIP_MIN) || (istrip > ISTRIP_MAX) ||
      (ixs<IX_MIN) || (ixs > IX_MAX) ||
      (iys<IY_MIN) || (iys > IY_MAX) ||
      (abs(iz)) != 1 ||
      (iplane != 1 && iplane != 2)) { valid = false; }
  return valid;

}
  
ESDetId& ESDetId::operator=(const DetId& gen) {
  if (!gen.null() && ( gen.det()!=Ecal || gen.subdetId()!=EcalPreshower )) {
    throw cms::Exception("InvalidDetId");
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

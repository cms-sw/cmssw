#include "DataFormats/EcalDetId/interface/EKDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
    
#include <iostream>
#include <algorithm>

EKDetId::EKDetId(int module_ix, int module_iy, int fiber, int ro, 
		 int iz) : DetId( Ecal, EcalShashlik) {
  id_ |= (module_iy&0xff) | ((module_ix&0xff)<<8) |
    ((fiber&0x7)<<16) | ((ro&0x3)<<19) | ((iz>0)?(0x200000):(0));
}

EKDetId::EKDetId(int module_ix, int module_iy, int iz) 
  : DetId( Ecal, EcalShashlik) {
  id_ |= (module_iy&0xff) | ((module_ix&0xff)<<8) | ((iz>0)?(0x200000):(0));
}

void EKDetId::setFiber(int fib, int ro) {
  uint32_t idc = (id_ & 0xffe0ffff);
  id_ = (idc) | ((fib&0x7)<<16) | ((ro&0x3)<<19);
}

int EKDetId::distanceX(const EKDetId& a,const EKDetId& b) {
  return abs(a.ix()-b.ix());
}

int EKDetId::distanceY(const EKDetId& a,const EKDetId& b) {
  return abs(a.iy() - b.iy()); 
}

uint32_t EKDetId::denseIndex() const {
  throw cms::Exception("No support") << "EKDetId::denseIndex is not implemented";
  return 0;
}

EKDetId EKDetId::detIdFromDenseIndex( uint32_t din ) {
  throw cms::Exception("No support") << "EKDetId::detIdFromDenseIndex is not implemented";  
  return EKDetId ();
}

#include <ostream>
std::ostream& operator<<(std::ostream& s,const EKDetId& id) {
  return s << "(EK iz " << ((id.zside()>0)?("+ "):("- ")) << " fiber "
	   << id.fiber() << ", RO " << id.readout() << ", ix " << id.ix() 
	   << ", iy " << id.iy() << ')';
}


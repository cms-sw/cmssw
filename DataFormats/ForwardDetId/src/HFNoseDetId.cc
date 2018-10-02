#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>
#include <iostream>

const HFNoseDetId HFNoseDetId::Undefined(0,0,0,0,0,0,0);

HFNoseDetId::HFNoseDetId() : DetId() {
}

HFNoseDetId::HFNoseDetId(uint32_t rawid) : DetId(rawid) {
}

HFNoseDetId::HFNoseDetId(int zp, int type, int layer, int waferU, int waferV,
			 int cellU, int cellV) : DetId(Forward,HFNose) {  

  int waferUabs(std::abs(waferU)), waferVabs(std::abs(waferV));
  int waferUsign = (waferU >= 0) ? 0 : 1;
  int waferVsign = (waferV >= 0) ? 0 : 1;
  int zside      = (zp < 0) ? 1 : 0;
  id_ |= (((cellU     & kHFNoseCellUMask) << kHFNoseCellUOffset) |
	  ((cellV     & kHFNoseCellVMask) << kHFNoseCellVOffset) |
	  ((waferUabs & kHFNoseWaferUMask) << kHFNoseWaferUOffset) |
	  ((waferUsign& kHFNoseWaferUSignMask) << kHFNoseWaferUSignOffset) |
	  ((waferVabs & kHFNoseWaferVMask) << kHFNoseWaferVOffset) |
	  ((waferVsign& kHFNoseWaferVSignMask) << kHFNoseWaferVSignOffset) |
	  ((layer     & kHFNoseLayerMask) << kHFNoseLayerOffset) |
	  ((zside     & kHFNoseZsideMask) << kHFNoseZsideOffset) |
	  ((type      & kHFNoseTypeMask)  << kHFNoseTypeOffset));
}

HFNoseDetId::HFNoseDetId(const DetId& gen) {
  if (!gen.null()) {
    if (!((gen.det()==Forward) && (gen.subdetId()==(int)(HFNose)))) {
      throw cms::Exception("Invalid DetId") << "Cannot initialize HFNoseDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_ = gen.rawId();
}

HFNoseDetId& HFNoseDetId::operator=(const DetId& gen) {
  if (!gen.null()) {
    if (!((gen.det()==Forward) && (gen.subdetId()==(int)(HFNose)))) {
      throw cms::Exception("Invalid DetId") << "Cannot assign HFNoseDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_ = gen.rawId();
  return (*this);
}

std::ostream& operator<<(std::ostream& s,const HFNoseDetId& id) {
  return s << " HFNoseDetId::EE:HE= " << id.isEE() << ":" << id.isHE()
	   << " type= " << id.type()  << " z= " << id.zside() 
	   << " layer= " << id.layer() 
	   << " wafer(u,v:x,y)= (" << id.waferU() << "," << id.waferV() << ":"
	   << id.waferX() << "," << id.waferY() << ")"
	   << " cell(u,v:x,y)= (" << id.cellU() << "," << id.cellV() << ":"
	   << id.cellX() << "," << id.cellY() << ")";
}



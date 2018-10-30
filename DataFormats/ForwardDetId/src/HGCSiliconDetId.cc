#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>
#include <iostream>

const HGCSiliconDetId HGCSiliconDetId::Undefined(HGCalEE,0,0,0,0,0,0,0);

HGCSiliconDetId::HGCSiliconDetId() : DetId() {
}

HGCSiliconDetId::HGCSiliconDetId(uint32_t rawid) : DetId(rawid) {
}

HGCSiliconDetId::HGCSiliconDetId(DetId::Detector det, int zp, int type, 
				 int layer, int waferU, int waferV, int cellU,
				 int cellV) : DetId(det,ForwardEmpty) {  

  int waferUabs(std::abs(waferU)), waferVabs(std::abs(waferV));
  int waferUsign = (waferU >= 0) ? 0 : 1;
  int waferVsign = (waferV >= 0) ? 0 : 1;
  int zside      = (zp < 0) ? 1 : 0;
  id_ |= (((cellU     & kHGCalCellUMask) << kHGCalCellUOffset) |
	  ((cellV     & kHGCalCellVMask) << kHGCalCellVOffset) |
	  ((waferUabs & kHGCalWaferUMask) << kHGCalWaferUOffset) |
	  ((waferUsign& kHGCalWaferUSignMask) << kHGCalWaferUSignOffset) |
	  ((waferVabs & kHGCalWaferVMask) << kHGCalWaferVOffset) |
	  ((waferVsign& kHGCalWaferVSignMask) << kHGCalWaferVSignOffset) |
	  ((layer     & kHGCalLayerMask) << kHGCalLayerOffset) |
	  ((zside     & kHGCalZsideMask) << kHGCalZsideOffset) |
	  ((type      & kHGCalTypeMask)  << kHGCalTypeOffset));
}

HGCSiliconDetId::HGCSiliconDetId(const DetId& gen) {
  if (!gen.null()) {
    if ((gen.det()!=HGCalEE) && (gen.det()!=HGCalHSi)) {
      throw cms::Exception("Invalid DetId") << "Cannot initialize HGCSiliconDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_ = gen.rawId();
}

HGCSiliconDetId& HGCSiliconDetId::operator=(const DetId& gen) {
  if (!gen.null()) {
    if ((gen.det()!=HGCalEE) && (gen.det()!=HGCalHSi)) {
      throw cms::Exception("Invalid DetId") << "Cannot assign HGCSiliconDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_ = gen.rawId();
  return (*this);
}

std::ostream& operator<<(std::ostream& s,const HGCSiliconDetId& id) {
  return s << " HGCSiliconDetId::EE:HE= " << id.isEE() << ":" << id.isHE()
	   << " type= " << id.type()  << " z= " << id.zside() 
	   << " layer= " << id.layer() 
	   << " wafer(u,v:x,y)= (" << id.waferU() << "," << id.waferV() << ":"
	   << id.waferX() << "," << id.waferY() << ")"
	   << " cell(u,v:x,y)= (" << id.cellU() << "," << id.cellV() << ":"
	   << id.cellX() << "," << id.cellY() << ")";
}



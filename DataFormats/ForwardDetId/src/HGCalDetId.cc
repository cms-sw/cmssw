#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include <iostream>

//#define DebugLog

HGCalDetId::HGCalDetId() : DetId() {
}

HGCalDetId::HGCalDetId(uint32_t rawid) : DetId(rawid) {
}

HGCalDetId::HGCalDetId(ForwardSubdetector subdet, int zp, int lay, int wafertype, int wafer, int cell) : DetId(Forward,subdet) {  

  if (wafertype < 0) wafertype = 0;
  if (cell>kHGCalCellMask || cell<0 || wafer>kHGCalWaferMask || wafer<0 || wafertype>kHGCalWaferTypeMask || lay>kHGCalLayerMask || lay<0) {
#ifdef DebugLog
    std::cout << "[HGCalDetId] request for new id for"
	      << " layer=" << lay << ":" << kHGCalLayerMask
	      << " @ zp=" << zp 
	      << " wafer=" << wafer << ":" << kHGCalWaferMask
	      << " waferType=" << wafertype << ":" << kHGCalWaferTypeMask
	      << " cell=" << cell << ":" << kHGCalCellMask
	      << " for subdet=" << subdet 
	      << " has one or more fields out of bounds and will be reset" 
	      << std::endl;
#endif
    zp = lay = wafertype = wafer = cell = 0;
  }
  id_ |= ((cell   & kHGCalCellMask)         << kHGCalCellOffset);
  id_ |= ((wafer  & kHGCalWaferMask)        << kHGCalWaferOffset);
  id_ |= ((wafertype & kHGCalWaferTypeMask) << kHGCalWaferTypeOffset);
  id_ |= ((lay    & kHGCalLayerMask)        << kHGCalLayerOffset);
  if (zp>0) id_ |= ((zp & kHGCalZsideMask)  << kHGCalZsideOffset);
}

HGCalDetId::HGCalDetId(const DetId& gen) {
  id_ = gen.rawId();
}

HGCalDetId& HGCalDetId::operator=(const DetId& gen) {
  id_ = gen.rawId();
  return (*this);
}

bool HGCalDetId::isValid(ForwardSubdetector subdet, int zp, int lay, int wafertype, int wafer, int cell) {

  bool ok = ((subdet == HGCEE || subdet == HGCHEF || subdet == HGCHEB) &&
	     (cell >= 0 && cell <= kHGCalCellMask) && 
	     (wafer >= 0 && wafer <= kHGCalWaferMask) &&
	     (wafertype <= kHGCalWaferTypeMask) && 
	     (lay >= 0 && lay <= kHGCalLayerMask) &&
	     (zp == -1 || zp == 1));
#ifdef DebugLog
  if (!ok) 
    std::cout << "HGCalDetId: subdet " << subdet << ":" 
	      << (subdet == HGCEE || subdet == HGCHEF || subdet == HGCHEB) 
	      << " Cell " << cell << ":" << (cell >= 0 && cell <= kHGCalCellMask) 
	      << " Wafer " << wafer << ":" << (wafer >= 0 && wafer <= kHGCalWaferMask) 
	      << " WaferType " << wafertype << ":" << (wafertype <= kHGCalWaferTypeMask)
	      << " Layer " << lay << ":" << (lay >= 0 && lay <= kHGCalLayerMask) 
	      << " zp " << zp << ":" << (zp == -1 || zp == 1) << std::endl;
#endif
  return ok;
}



std::ostream& operator<<(std::ostream& s,const HGCalDetId& id) {
  return s << "HGCalDetId::isHGCal=" << id.isHGCal() << " subdet= " 
	   << id.subdetId() << " zpos=" << id.zside() << " layer=" 
	   << id.layer() << " wafer type=" << id.waferType() << " wafer="
	   << id.wafer() << " cell=" << id.cell();
}

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include <iostream>

//#define DebugLog

HGCalDetId::HGCalDetId() : DetId() {
}

HGCalDetId::HGCalDetId(uint32_t rawid) : DetId(rawid) {
}

HGCalDetId::HGCalDetId(ForwardSubdetector subdet, int zp, int lay, int sec, int subsec, int cell) : DetId(Forward,subdet) {  

  if (cell > kHGCalCellMask || sec>kHGCalSectorMask || 
      subsec > kHGCalSubSectorMask || lay>kHGCalLayerMask ) {
#ifdef DebugLog
    std::cout << "[HGCalDetId] request for new id for layer=" << lay
	      << " @ zp=" << zp 
	      << " sector=" << sec 
	      << " subsec=" << subsec 
	      << " cell=" << cell 
	      << " for subdet=" << subdet 
	      << " has one or more fields out of bounds and will be reset" 
	      << std::endl;
#endif
    cell=0;  sec=0;  subsec=0; lay=0;
  }
  uint32_t rawid=0;
  rawid |= ((cell   & kHGCalCellMask)        << kHGCalCellOffset);
  rawid |= ((sec    & kHGCalSectorMask)      << kHGCalSectorOffset);
  if (subsec<0) subsec=0;
  rawid |= ((subsec & kHGCalSubSectorMask)   << kHGCalSubSectorOffset);
  rawid |= ((lay    & kHGCalLayerMask)       << kHGCalLayerOffset);
  if (zp>0) rawid |= ((zp & kHGCalZsideMask) << kHGCalZsideOffset);
  id_ = rawid;
}

HGCalDetId::HGCalDetId(const DetId& gen) {
  id_ = gen.rawId();
}

HGCalDetId& HGCalDetId::operator=(const DetId& gen) {
  id_ = gen.rawId();
  return (*this);
}

bool HGCalDetId::isValid(ForwardSubdetector subdet, int zp, int lay, 
			 int mod, int subsec, int cell) {
  bool ok = ((subdet == HGCEE || subdet == HGCHEF || subdet == HGCHEB) &&
	     (cell >= 0 && cell <= kHGCalCellMask) && 
	     (mod >= 1 && mod <= kHGCalSectorMask) &&
	     (subsec == 0 || subsec == 1) && 
	     (lay >= 0 && lay <= kHGCalLayerMask) &&
	     (zp == -1 || zp == 1));
#ifdef DebugLog
  if (!ok) 
    std::cout << "HGCalDetId: subdet " << subdet << ":" 
	      << (subdet == HGCEE || subdet == HGCHEF || subdet == HGCHEB) 
	      << " Cell " << cell << ":" << (cell >= 0 && cell <= 0xffff) 
	      << " Module " << mod << ":" << (mod >= 1 && mod <= 0x7f) 
	      << " SubSector " << subsec << ":" << (subsec == 0 || subsec == 1)
	      << " Layer " << lay << ":" << (lay >= 0 && lay <= 0x7f) 
	      << " zp " << zp << ":" << (zp == -1 || zp == 1) << std::endl;
#endif
  return ok;
}

std::ostream& operator<<(std::ostream& s,const HGCalDetId& id) {
  return s << "isHGCal=" << id.isHGCal() << " zpos=" << id.zside() 
	   << " layer=" << id.layer()  << " phi subSector=" << id.subsector()
	   << " sector=" << id.sector() << " cell=" << id.cell();
}

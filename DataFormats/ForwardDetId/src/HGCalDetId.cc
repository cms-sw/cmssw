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





// Function to work on raw HGCAlDetIds
int HGCalDetIdUtil::subdetIdOf(uint32_t id) {
  return getMaskedId(id, HGCalDetId::kSubdetOffset, 0x7);
}
int HGCalDetIdUtil::cellOf(uint32_t id) {
  return getMaskedId(id, HGCalDetId::kHGCalCellOffset,HGCalDetId::kHGCalCellMask);
}
int HGCalDetIdUtil::waferOf(uint32_t id) {
  return getMaskedId(id, HGCalDetId::kHGCalWaferOffset,HGCalDetId::kHGCalWaferMask);
}
int HGCalDetIdUtil::waferTypeOf(uint32_t id) {
  return (getMaskedId(id, HGCalDetId::kHGCalWaferTypeOffset,HGCalDetId::kHGCalWaferTypeMask) ? 1 : -1);
}
int HGCalDetIdUtil::layerOf(uint32_t id) {
  return getMaskedId(id, HGCalDetId::kHGCalLayerOffset,HGCalDetId::kHGCalLayerMask);
}
int HGCalDetIdUtil::zsideOf(uint32_t id) {
  return (getMaskedId(id, HGCalDetId::kHGCalZsideOffset,HGCalDetId::kHGCalZsideMask) ? 1 : -1);
}

void HGCalDetIdUtil::setCellOf(uint32_t& id, int cell) {
  HGCalDetIdUtil::resetMaskedId(id, HGCalDetId::kHGCalCellOffset,HGCalDetId::kHGCalCellMask);
  HGCalDetIdUtil::setMaskedId(id, cell, HGCalDetId::kHGCalCellOffset,HGCalDetId::kHGCalCellMask);
}
void HGCalDetIdUtil::setWaferOf(uint32_t& id, int mod) {
  HGCalDetIdUtil::resetMaskedId(id, HGCalDetId::kHGCalWaferOffset,HGCalDetId::kHGCalWaferMask);
  HGCalDetIdUtil::setMaskedId(id, mod, HGCalDetId::kHGCalWaferOffset,HGCalDetId::kHGCalWaferMask);
}
void HGCalDetIdUtil::setWaferTypeOf(uint32_t& id, int wafertype) {
  HGCalDetIdUtil::resetMaskedId(id, HGCalDetId::kHGCalWaferTypeOffset, HGCalDetId::kHGCalWaferTypeMask);
  HGCalDetIdUtil::setMaskedId(id, wafertype, HGCalDetId::kHGCalWaferTypeOffset,HGCalDetId::kHGCalWaferTypeMask);
}
void HGCalDetIdUtil::setLayerOf(uint32_t& id, int lay) {
  HGCalDetIdUtil::resetMaskedId(id, HGCalDetId::kHGCalLayerOffset,HGCalDetId::kHGCalLayerMask);
  HGCalDetIdUtil::setMaskedId(id, lay, HGCalDetId::kHGCalLayerOffset,HGCalDetId::kHGCalLayerMask);
}
void HGCalDetIdUtil::setZsideOf(uint32_t& id, int zside) {
  HGCalDetIdUtil::resetMaskedId(id, HGCalDetId::kHGCalZsideOffset,HGCalDetId::kHGCalZsideMask);
  HGCalDetIdUtil::setMaskedId(id, zside, HGCalDetId::kHGCalZsideOffset,HGCalDetId::kHGCalZsideMask);
}

int HGCalDetIdUtil::getMaskedId(uint32_t id, uint32_t shift, uint32_t mask) {
  return (id >> shift) & mask ; 
}
void HGCalDetIdUtil::setMaskedId(uint32_t& id, uint32_t value, uint32_t shift, uint32_t mask ){
  id|= ((value & mask ) <<shift ); 
}
void HGCalDetIdUtil::resetMaskedId(uint32_t& id, uint32_t shift, uint32_t mask ){
  id &= ~(mask<<shift); 
} 


std::ostream& operator<<(std::ostream& s,const HGCalDetId& id) {
  return s << "isHGCal=" << id.isHGCal() << " zpos=" << id.zside() 
	   << " layer=" << id.layer()  << " wafer type=" << id.waferType()
	   << " wafer=" << id.wafer() << " cell=" << id.cell();
}

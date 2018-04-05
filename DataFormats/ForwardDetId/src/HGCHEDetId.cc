#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>

const HGCHEDetId HGCHEDetId::Undefined(ForwardEmpty,0,0,0,0,0);

HGCHEDetId::HGCHEDetId() : DetId() {
}

HGCHEDetId::HGCHEDetId(uint32_t rawid) : DetId(rawid) {
}

HGCHEDetId::HGCHEDetId(ForwardSubdetector subdet, int zp, int lay, int sec, int subsec, int cell) : DetId(Forward,subdet) {  

  id_ |= ((cell   & kHGCHECellMask)       << kHGCHECellOffset);
  id_ |= ((sec    & kHGCHESectorMask)     << kHGCHESectorOffset);
  if (subsec<0) subsec=0;
  id_ |= ((subsec & kHGCHESubSectorMask)  << kHGCHESubSectorOffset);
  id_ |= ((lay    & kHGCHELayerMask)      << kHGCHELayerOffset);
  if(zp>0) id_ |= ((zp & kHGCHEZsideMask) << kHGCHEZsideOffset);
}

HGCHEDetId::HGCHEDetId(const DetId& gen) {
  if (!gen.null()) {
    ForwardSubdetector subdet=(ForwardSubdetector(gen.subdetId()));
    if ((gen.det()!=Forward) ||
	(subdet!=HGCHEF && subdet!=HGCHEB && subdet!=HGCHET)) {
      throw cms::Exception("Invalid DetId") << "Cannot initialize HGCHEDetId from " << std::hex << gen.rawId() << std::dec << " Det|SubDet " << gen.det() << "|" << subdet; 
    }  
  }
  id_ = gen.rawId();
}

HGCHEDetId& HGCHEDetId::operator=(const DetId& gen) {
  if (!gen.null()) {
    ForwardSubdetector subdet=(ForwardSubdetector(gen.subdetId()));
    if ((gen.det()!=Forward) ||
	(subdet!=HGCHEF && subdet!=HGCHEB && subdet!=HGCHET)) {
      throw cms::Exception("Invalid DetId") << "Cannot assign HGCHEDetId from " << std::hex << gen.rawId() << std::dec << " Det|SubDet " << gen.det() << "|" << subdet; 
    }  
  }
  id_ = gen.rawId();
  return (*this);
}

HGCHEDetId HGCHEDetId::geometryCell () const {
  int sub = ((subdet() == HGCHEF) ? 0 : ((id_>> kHGCHESubSectorOffset)& kHGCHESubSectorMask));
  return HGCHEDetId(subdet(), zside(), layer(), sector(), sub, 0);
}

std::ostream& operator<<(std::ostream& s,const HGCHEDetId& id) {
  if  (id.subdet() == HGCHEF || id.subdet() == HGCHEB ||
       id.subdet() == HGCHET) {
    return s << "isHE=" << id.isHE() << " zpos=" << id.zside() 
	     << " layer=" << id.layer() << " phi subSector=" << id.subsector()
	     << " sector=" << id.sector() << " cell=" << id.cell();
  } else {
    return s << std::hex << id.rawId() << std::dec;
  }
}



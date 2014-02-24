#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>

HcalZDCDetId::HcalZDCDetId() : DetId() {
}


HcalZDCDetId::HcalZDCDetId(uint32_t rawid) : DetId(rawid) {
}

HcalZDCDetId::HcalZDCDetId(Section section, bool true_for_positive_eta, int channel) : DetId(DetId::Calo,SubdetectorId) {
  id_|=(section&0x7)<<5; // 3-bits 5:7 (range 0:7)
  if (true_for_positive_eta) id_|=0x100; // 1-bit 8:8 (range 0:1)
  id_|=channel&0x1F;     // 5-bits 0:4 (range 0:31)
  id_|=0x10000;          // 1-bit 16:16 (change of format)
//std::cout << "HcalZDCDetID::HcalZDCDetId: id_ "<<std::hex << id_ << std::dec << ", section: " << section << ", channel: " << channel << " z " << true_for_positive_eta << std::endl;
}

HcalZDCDetId::HcalZDCDetId(const DetId& gen) {
  if (!gen.null() && (gen.det()!=Calo || gen.subdetId()!=SubdetectorId)) {
    throw cms::Exception("Invalid DetId") << "Cannot initialize ZDCDetId from " << std::hex << gen.rawId() << std::dec; 
  }
  id_=gen.rawId();
}

HcalZDCDetId& HcalZDCDetId::operator=(const DetId& gen) {
  if (!gen.null() && (gen.det()!=Calo || gen.subdetId()!=SubdetectorId)) {
    throw cms::Exception("Invalid DetId") << "Cannot assign ZDCDetId from " << std::hex << gen.rawId() << std::dec; 
  }
  id_=gen.rawId();
  return *this;
}

bool HcalZDCDetId::operator==(DetId gen) const {
  uint32_t rawid = gen.rawId();
  if (rawid == id_) return true;
  int zsid, sec, chn;
  if ((rawid&0x10000)==0) {
    zsid = (rawid&0x40)?(1):(-1);
    sec  = (rawid>>4)&0x3;
    chn  = rawid&0xF;
  } else {
    zsid = (rawid&0x100)?(1):(-1);
    sec  = (rawid>>5)&0x7;
    chn  = rawid&0x1F;
  }
  bool result = (zsid==zside() && sec==section() && chn==channel());
  return result;
}

bool HcalZDCDetId::operator!=(DetId gen) const {
  uint32_t rawid = gen.rawId();
  if (rawid == id_) return false;
  int zsid, sec, chn;
  if ((rawid&0x10000)==0) {
    zsid = (rawid&0x40)?(1):(-1);
    sec  = (rawid>>4)&0x3;
    chn  = rawid&0xF;
  } else {
    zsid = (rawid&0x100)?(1):(-1);
    sec  = (rawid>>5)&0x7;
    chn  = rawid&0x1F;
  }
  bool result = (zsid!=zside() || sec!=section() || chn!=channel());
  return result;
}

bool HcalZDCDetId::operator<(DetId gen) const {
  uint32_t rawid = gen.rawId();
  if ((rawid&0x10000)==(id_&0x10000)) {
    return id_<rawid;
  } else {
    int zsid, sec, chn;
    if ((rawid&0x10000)==0) {
      zsid = (rawid&0x40)?(1):(-1);
      sec  = (rawid>>4)&0x3;
      chn  = rawid&0xF;
    } else {
      zsid = (rawid&0x100)?(1):(-1);
      sec  = (rawid>>5)&0x7;
      chn  = rawid&0x1F;
    }
    rawid = 0;
    if ((id_&0x10000) == 0) {
      rawid |= (sec&0x3)<<4; 
      if (zsid > 0) rawid |= 0x40;
      rawid |= chn&0xF;
    } else {
      rawid |= (sec&0x7)<<5; 
      if (zsid > 0) rawid |= 0x100;
      rawid |= chn&0x1F;
      rawid |= 0x10000;
    }
    return (id_&0x1FFFF)<rawid;
  }
}

int HcalZDCDetId::zside() const {
  if (id_&0x10000) return (id_&0x100)?(1):(-1);
  else             return (id_&0x40)?(1):(-1);
}

HcalZDCDetId::Section HcalZDCDetId::section() const { 
  if (id_&0x10000) return (Section)((id_>>5)&0x7);
  else             return (Section)((id_>>4)&0x3);
}

int HcalZDCDetId::depth() const { 
  if      (section() == EM)  return 1;
  else if (section() == HAD) return (1+channel());
  else                       return channel();
}

int HcalZDCDetId::channel() const { 
  if (id_&0x10000) return  id_&0x1F;
  else             return  id_&0xF;
}

uint32_t HcalZDCDetId::otherForm() const {
  uint32_t rawId = (id_&0xFE000000);
  if (id_&0x10000 == 0) {
    rawId |= (section()&0x7)<<5; 
    if (zside() > 0) rawId |= 0x100;
    rawId |= (channel()&0x1F) | 0x10000;
  } else {
    rawId |= (section()&0x3)<<4; 
    if (zside() > 0) rawId |= 0x40;
    rawId |= channel()&0xF;
  }
  return rawId;
}

uint32_t HcalZDCDetId::denseIndex() const {
   const int se ( section() ) ;
   uint32_t indx = channel() - 1;
   if (se == RPD) {
     indx += (2*kDepTot + zside()<0 ? 0 : kDepRPD);
   } else {
     indx += ( (zside()<0 ? 0 : kDepTot) + 
	       ( se == HAD  ? kDepEM :
		 ( se == LUM ? kDepEM + kDepHAD : 0) ) );
   }
   return indx;
}

HcalZDCDetId HcalZDCDetId::detIdFromDenseIndex( uint32_t di ) {
  if( validDenseIndex( di ) ) {
    if (di >= 2*kDepTot) {
      const bool lz ( di >= kDepTot ) ;
      const uint32_t in ( di%kDepTot ) ;
      const Section se ( in<kDepEM ? EM :
			 ( in<kDepEM+kDepHAD ? HAD : LUM ) );
      const uint32_t dp ( EM == se ? in+1 :
			  ( HAD == se ? in - kDepEM + 1 : in - kDepEM - kDepHAD + 1 ) );
      return HcalZDCDetId( se, lz, dp ) ;
//    std::cout<<"HcalZDCDetID:: section: "<<se<<", lz: "<<lz<<", dp: "<<dp<<", from denseIndex: di: "<<di<<std::endl;
    } else {
      const bool lz ( di >= 2*kDepTot+kDepRPD ) ;
      const uint32_t in ((di-2*kDepTot)%kDepRPD );
      const Section se (RPD);
      const uint32_t dp ( in+1 );
//    std::cout<<"HcalZDCDetID:: section: "<<se<<", lz: "<<lz<<", dp: "<<dp<<", from denseIndex: di: "<<di<<std::endl;
      return HcalZDCDetId( se, lz, dp ) ;
    }
  } else {
    return HcalZDCDetId() ;
  }
}

bool HcalZDCDetId::validDetId( Section se , int     dp   ) {
  return ( dp >= 1 && (
		       ( ( se == EM      ) &&
			 ( dp <= kDepEM  )    ) ||
		       ( ( se == HAD     ) &&
			 ( dp <= kDepHAD )    ) ||
		       ( ( se == LUM     ) &&
			 ( dp <= kDepLUM )    ) ||
		       ( ( se == RPD     ) &&
			 ( dp <= kDepRPD )    ) 
		       )
	   ) ;
}

std::ostream& operator<<(std::ostream& s,const HcalZDCDetId& id) {
  s << "(ZDC" << ((id.zside()==1)?("+"):("-"));
  switch (id.section()) {
  case(HcalZDCDetId::EM) : s << " EM "; break;
  case(HcalZDCDetId::HAD) : s << " HAD "; break;
  case(HcalZDCDetId::LUM) : s << " LUM "; break;
  case(HcalZDCDetId::RPD) : s << " RPD "; break;
  default : s <<" UNKNOWN ";
  }
  return s << id.channel() << "," << id.depth() << ')';
}


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
  std::cout << "HcalZDCDetID::HcalZDCDetId: id_ "<<std::hex << id_ << std::dec << ", section: " << section << ", channel: " << channel << " z " << true_for_positive_eta << std::endl;
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

uint32_t HcalZDCDetId::denseIndex() const {
   const int se ( section() ) ;
   return ( ( zside()<0 ? 0 : kDepTot ) + channel() - 1 +
	    ( se == HAD  ? kDepEM :
	      ( se == LUM ? kDepEM + kDepHAD :
		( se == RPD ? kDepEM+kDepHAD+kDepLUM : 0) ) ) ) ;
}

HcalZDCDetId HcalZDCDetId::detIdFromDenseIndex( uint32_t di ) {
  if( validDenseIndex( di ) ) {
    const bool lz ( di >= kDepTot ) ;
    const uint32_t in ( di%kDepTot ) ;
    const Section se ( in<kDepEM ? EM :
		       ( in<kDepEM+kDepHAD ? HAD : 
			 ( in<kDepEM+kDepHAD+kDepLUM ? LUM : RPD ) ) ) ;
    const uint32_t dp ( EM == se ? in+1 :
			( HAD == se ? in - kDepEM + 1 : 
			  (LUM == se ? in - kDepEM - kDepHAD + 1 : 
			   (RPD == se ? in - kDepEM - kDepHAD - kDepLUM + 1 : in - kDepEM-kDepHAD-kDepLUM+kDepRPD+1) )  ) ) ;
    std::cout<<"HcalZDCDetID:: section: "<<se<<", lz: "<<lz<<", dp: "<<dp<<", from denseIndex: di: "<<di<<std::endl;
    return HcalZDCDetId( se, lz, dp ) ;
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


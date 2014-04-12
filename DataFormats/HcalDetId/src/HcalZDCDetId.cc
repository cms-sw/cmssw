#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "FWCore/Utilities/interface/Exception.h"

HcalZDCDetId::HcalZDCDetId() : DetId() {
}


HcalZDCDetId::HcalZDCDetId(uint32_t rawid) : DetId(rawid) {
}

HcalZDCDetId::HcalZDCDetId(Section section, bool true_for_positive_eta, int channel) : DetId(DetId::Calo,SubdetectorId) {
  id_|=(section&0x3)<<4;
  if (true_for_positive_eta) id_|=0x40;
  id_|=channel&0xF;
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

uint32_t 
HcalZDCDetId::denseIndex() const 
{
   const int se ( section() ) ;
   return ( ( zside()<0 ? 0 : kDepTot ) + channel() - 1 +
	    ( se == HAD  ? kDepEM :
	      ( se == LUM ? kDepEM + kDepHAD : 0 ) ) ) ;
}

HcalZDCDetId 
HcalZDCDetId::detIdFromDenseIndex( uint32_t di ) 
{
   if( validDenseIndex( di ) )
   {
      const bool lz ( di >= kDepTot ) ;
      const uint32_t in ( di%kDepTot ) ;
      const Section se ( in<kDepEM ? EM :
			 ( in<kDepEM+kDepHAD ? HAD : LUM ) ) ;
      const uint32_t dp ( EM == se ? in+1 :
			  ( HAD == se ? in - kDepEM + 1 : in - kDepEM - kDepHAD + 1 ) ) ;
      return HcalZDCDetId( se, lz, dp ) ;
   }
   else
   {
      return HcalZDCDetId() ;
   }
}

bool 
HcalZDCDetId::validDetId( Section se ,
			  int     dp   )
{
  return ( dp >= 1 && (
		       ( ( se == EM      ) &&
			 ( dp <= kDepEM  )    ) ||
		       ( ( se == HAD     ) &&
			 ( dp <= kDepHAD )    ) ||
		       ( ( se == LUM     ) &&
			 ( dp <= kDepLUM )    )   
		       )
	   ) ;
}

std::ostream& operator<<(std::ostream& s,const HcalZDCDetId& id) {
  s << "(ZDC" << ((id.zside()==1)?("+"):("-"));
  switch (id.section()) {
  case(HcalZDCDetId::EM) : s << " EM "; break;
  case(HcalZDCDetId::HAD) : s << " HAD "; break;
  case(HcalZDCDetId::LUM) : s << " LUM "; break;
  default : s <<" UNKNOWN ";
  }
  return s << id.channel() << "," << id.depth() << ')';
}


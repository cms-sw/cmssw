#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>

CaloTowerDetId::CaloTowerDetId() : DetId() {
}
  
CaloTowerDetId::CaloTowerDetId(uint32_t rawid) : DetId(rawid&0xFFF0FFFFu) {
  
}
  
CaloTowerDetId::CaloTowerDetId(int ieta, int iphi) : DetId(Calo,SubdetId) {
  id_|= 
    ((ieta>0)?(0x2000|((ieta&0x3F)<<7)):(((-ieta)&0x3f)<<7)) |
    (iphi&0x7F);
}
  
CaloTowerDetId::CaloTowerDetId(const DetId& gen) {
  if (!gen.null() && (gen.det()!=Calo || gen.subdetId()!=SubdetId)) {
    throw cms::Exception("Invalid DetId") << "Cannot initialize CaloTowerDetId from " << std::hex << gen.rawId() << std::dec;
  }
  id_=gen.rawId(); 
}
  
CaloTowerDetId& CaloTowerDetId::operator=(const DetId& gen) {
  if (!gen.null() && (gen.det()!=Calo || gen.subdetId()!=SubdetId)) {
    throw cms::Exception("Invalid DetId") << "Cannot assign CaloTowerDetId from " << std::hex << gen.rawId() << std::dec;
  }
  id_=gen.rawId();
  return *this;
}

int CaloTowerDetId::iphi() const {
  int retval=id_&0x7F;
  return retval;
}  

bool 
CaloTowerDetId::validDetId( int ie , int ip ) 
{
   const int ia ( abs( ie ) ) ;
   return ( ( ia >= 1 )        &&
	    ( ip >= 1 )        &&
	    ( ia <= kMaxIEta ) &&
	    ( ip <= kMaxIPhi ) &&
	    ( ( ( ia <= kBarIEta ) &&
		( ip <= kBarNPhi )    ) ||
	     ( ( ia >  kBarIEta ) &&
		( ia <= kEndIEta ) &&
		( (ip-1)%2 == 0  )    ) ||
	      ( ( ia >  kEndIEta ) &&
		( ia <= kForIEta ) &&
		( (ip-3)%4 == 0 )    )    )  ) ;
}

uint32_t 
CaloTowerDetId::denseIndex() const 
{
   const uint32_t ie ( ietaAbs()     ) ;
   const uint32_t ip ( iphi()    - 1 ) ;
   
   return ( ( 0 > zside() ? 0 : kAllNTot ) +
	    ( ( kBarIEta >= ie ? ( ie - 1 )*kBarNPhi + ip :
		( kEndIEta >= ie ?  kBarNTot + ( ie - 1 - kBarIEta )*kEndNPhi + ip/2 :
		  kBarNTot + kEndNTot + ( ie - 1 - kEndIEta )*kForNPhi + ip/4 ) ) ) ) ;
}

CaloTowerDetId 
CaloTowerDetId::detIdFromDenseIndex( uint32_t din ) 
{
   const int iz ( din < kAllNTot ? -1 : 1 ) ;
   din %= kAllNTot ;
   const uint32_t ie ( ( kBarNTot + kEndNTot ) - 1 < din ?
		       kEndIEta + ( din - kBarNTot - kEndNTot )/kForNPhi + 1 :
		       ( kBarNTot - 1 < din ?
			 kBarIEta + ( din - kBarNTot )/kEndNPhi + 1 :
			 din/kBarNPhi + 1 ) ) ;

   const uint32_t ip ( ( kBarNTot + kEndNTot ) - 1 < din ?
		       ( ( din - kBarNTot - kEndNTot )%kForNPhi )*4 + 3 :
		       ( kBarNTot - 1 < din ?
			 ( ( din - kBarNTot )%kEndNPhi )*2 + 1 :
			 din%kBarNPhi + 1 ) ) ;

   return ( validDenseIndex( din ) ? CaloTowerDetId( iz*ie, ip ) : CaloTowerDetId() ) ;
}

std::ostream& operator<<(std::ostream& s, const CaloTowerDetId& id) {
  return s << "Tower (" << id.ieta() << "," << id.iphi() << ")";
}

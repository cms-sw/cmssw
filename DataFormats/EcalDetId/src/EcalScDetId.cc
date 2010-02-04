#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>

EcalScDetId::EcalScDetId() : DetId() {
}
EcalScDetId::EcalScDetId(uint32_t rawid) : DetId(rawid) {
}
EcalScDetId::EcalScDetId(int ix, int iy, int iz) : DetId(Ecal,EcalEndcap) 
{
  if(!validDetId(ix,iy,iz))
    {
      throw cms::Exception("InvalidDetId") << "EcalScDetId:  Cannot create object.  Indexes out of bounds \n" 
                                           << "x = " << ix << " y = " << iy << " z = " << iz;
    }
  const int scBit = 1<<15; //bit set to 1 to distinguish from crystal id (EEDetId)
  //                         and for a reasonale behaviour of DetId ccomparison operators.
  id_|=(iy&0x7f)|((ix&0x7f)<<7)|((iz>0)?(1<<14):(0))|scBit;
}
  
EcalScDetId::EcalScDetId(const DetId& gen) {
  if (!gen.null() && (gen.det()!=Ecal || gen.subdetId()!=EcalEndcap)) {
    throw cms::Exception("InvalidDetId"); 
  }
  id_=gen.rawId();
}
  
EcalScDetId& EcalScDetId::operator=(const DetId& gen) {
  if (!gen.null() && ( gen.det()!=Ecal || gen.subdetId()!=EcalEndcap )) {
    throw cms::Exception("InvalidDetId"); 
  }
  id_=gen.rawId();
  return *this;
}

int 
EcalScDetId::iquadrant() const 
{
   const int jx ( ix() ) ;
   const int jy ( iy() ) ;
   static const int MX ( IX_MAX/2 ) ;
   static const int MY ( IY_MAX/2 ) ;
   return ( MX <= jx && MY <= jy ? 1 :
	    ( MX >= jx && MY <= jy ? 2 :
	      ( MX >= jx && MY >= jy ? 3 : 4 ) ) ) ;
}

int 
EcalScDetId::isc() const 
{ 
   return EEDetId::isc( ix(), iy() ) ; 
}

bool 
EcalScDetId::validDetId(int iX, int iY, int iZ) 
{
   static const char endcapMap[401] = {
    "       XXXXXX       "
    "    XXXXXXXXXXXX    "
    "   XXXXXXXXXXXXXX   "
    "  XXXXXXXXXXXXXXXX  "
    " XXXXXXXXXXXXXXXXXX "
    " XXXXXXXXXXXXXXXXXX "             //    Z
    " XXXXXXXXXXXXXXXXXX "             //     x-----> X
    "XXXXXXXXXXXXXXXXXXXX"             //     |
    "XXXXXXXXX  XXXXXXXXX"             //     |
    "XXXXXXXX    XXXXXXXX"//_          //     |
    "XXXXXXXX    XXXXXXXX"             //     V Y
    "XXXXXXXXX  XXXXXXXXX"
    "XXXXXXXXXXXXXXXXXXXX"
    " XXXXXXXXXXXXXXXXXX "
    " XXXXXXXXXXXXXXXXXX "
    " XXXXXXXXXXXXXXXXXX "
    "  XXXXXXXXXXXXXXXX  "
    "   XXXXXXXXXXXXXX   "
    "    XXXXXXXXXXXX    "
    "       XXXXXX       "};

   return ( 1      == abs(iZ) && 
	    IX_MIN <= iX      &&
	    IX_MAX >= iX      &&
	    IY_MIN <= iY      &&
	    IY_MAX >= iY      &&
	    endcapMap[iX-1+(iY-1)*20]!=' ' ) ;
}

std::ostream& operator<<(std::ostream& s,const EcalScDetId& id) {
  return s << "(EE iz " << ((id.zside()>0)?("+ "):("- ")) << " ix " << id.ix() << " , iy " << id.iy() << ')';
}

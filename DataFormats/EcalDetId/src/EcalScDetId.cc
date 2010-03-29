#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
const int EcalScDetId::QuadColLimits[EcalScDetId::nCols+1] = { 0, 8,17,27,36,45,54,62,70,76,79 };
const int EcalScDetId::iYoffset[EcalScDetId::nCols+1]      = { 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0 };

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
  
int EcalScDetId::hashedIndex() const {
  return ((zside()>0)?(IX_MAX*IY_MAX):(0))+(iy()-1)*IX_MAX+(ix()-1);
}

int EcalScDetId::ixQuadrantOne() const
{ 
  int iQuadrant = iquadrant();
  if ( iQuadrant == 1 || iQuadrant == 4)
    return (ix() - 10);
  else if ( iQuadrant == 2 || iQuadrant == 3)
    return (11 - ix());
  //Should never be reached
  return -1;
}

int EcalScDetId::iyQuadrantOne() const
{ 
  int iQuadrant = iquadrant();
  if ( iQuadrant == 1 || iQuadrant == 2)
    return (iy() - 10);
  else if ( iQuadrant == 3 || iQuadrant == 4)
    return 11 - iy();
  //Should never be reached
  return -1;
}

int EcalScDetId::iquadrant() const {
  if (ix()>10)
    {
      if(iy()>10)
	return 1;
      else
	return 4;
    }
  else
    {
      if(iy()>10)
	return 2;
      else
	return 3;
    }
  //Should never be reached
  return -1;
}  

int EcalScDetId::isc() const 
{
  /*
   *  Return SC number from (x,y) coordinates.
   *
   *  Copied from code originally written by B W Kennedy
   */
  
  int iCol = ix();
  int iRow = iy();
  int nSCinQuadrant = QuadColLimits[nCols];
  int iSC;
  
  if (iRow <= iYoffset[iCol]) 
    return -1;
  else 
    iSC = QuadColLimits[iCol-1] + iRow - iYoffset[iCol];

  if (iSC > QuadColLimits[iCol]) 
    return -2;
  
  if (iSC>0) 
      iSC += nSCinQuadrant*(iquadrant()-1);
  
  return iSC;
}  

bool EcalScDetId::validDetId(int iX, int iY, int iZ) {
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

  return abs(iZ)==1 && endcapMap[iX-1+(iY-1)*20]!=' ';
}

std::ostream& operator<<(std::ostream& s,const EcalScDetId& id) {
  return s << "(EE iz " << ((id.zside()>0)?("+ "):("- ")) << " ix " << id.ix() << " , iy " << id.iy() << ')';
}

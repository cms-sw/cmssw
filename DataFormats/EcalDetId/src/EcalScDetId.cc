#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <ostream>
#include <cassert>

short EcalScDetId::xyz2HashedIndex[EcalScDetId::IX_MAX][EcalScDetId::IY_MAX][EcalScDetId::nEndcaps];

EcalScDetId EcalScDetId::hashedIndex2DetId[kSizeForDenseIndexing];


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
  
int EcalScDetId::iquadrant() const {
  const int xMiddle = IX_MAX/2; //y = 0 between xMiddle and xMiddle+1
  const int yMiddle = IY_MAX/2; //x = 0 between yMiddle and yMiddle+1
  if (iy()>yMiddle){// y>0
    if(ix()>xMiddle)   //             A y	     
      return 1;        //             |		     
    else	       //      Q2     |    Q1	     
      return 2;        //             |		     
  } else{// y<0	       //   ----------o---------> x   
    if(ix()>xMiddle)   //             |		      
      return 4;        //      Q3     |    Q4	    
    else	       //             |               
      return 3;
  }
  //Should never be reached
  return -1;
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

void EcalScDetId::checkHashedIndexMap(){
  static bool initialized = false;
  if(initialized) return;
  int hashedIndex = -1;
  for(int iZ = -1; iZ <= +1; iZ+=2){
    for(int iY = IY_MIN; iY <= IY_MAX; ++iY){
      for(int iX = IX_MIN; iX <= IX_MAX; ++iX){
	if(validDetId(iX,iY,iZ)){
	  xyz2HashedIndex[iX-IX_MIN][iY-IY_MIN][iZ>0?1:0] = ++hashedIndex;
	  assert((unsigned)hashedIndex < sizeof(hashedIndex2DetId)/sizeof(hashedIndex2DetId[0]));
	     hashedIndex2DetId[hashedIndex] = EcalScDetId(iX, iY, iZ);
	}
      }
    }
  }
  initialized = true;
}

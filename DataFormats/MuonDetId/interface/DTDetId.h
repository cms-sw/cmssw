/**
   \file
   Declaration of class DTDetId

   \author Stefano ARGIRO
   \version $Id: DTDetId.h,v 1.2 2005/08/23 09:11:28 argiro Exp $
   \date 27 Jul 2005
*/

#ifndef __DTDetId_h_
#define __DTDetId_h_

#include <DataFormats/MuonDetId/interface/MuonSubdetId.h>
#include <DataFormats/DetId/interface/DetId.h>

#include <iosfwd>

static const char CVSId__DTDetId[] = 
"$Id: DTDetId.h,v 1.2 2005/08/23 09:11:28 argiro Exp $";

  /**
     \class DTDetId DTDetId.h "/DTDetId.h"

     \brief DetUnit identifier for DT chambers
     
    
     \author Stefano ARGIRO
     \date 27 Jul 2005

  */
  class DTDetId :public cms::DetId {

  public:
      
    DTDetId();

    /// Construct from fully qualified identifier but wire
    DTDetId(int wheel, 
	    unsigned int station, 
	    unsigned int sector,
	    unsigned int superlayer,
	    unsigned int layer) :
      DetId(cms::DetId::Muon, MuonSubdetId::DT ){
 
      unsigned int tmpwheelid = (unsigned int)(wheel- minWheelId +1);
      id_ |= (tmpwheelid& wheelMask_)  << wheelStartBit_     |
 	     (station & stationMask_)  << stationStartBit_   |
	     (sector  &sectorMask_ )   << sectorStartBit_    |
             (superlayer & slMask_)    << slayerStartBit_    |
	     (layer & lMask_)          << layerStartBit_     ;
    }
    
  

    /// wheel id
    int wheel() const{
      return int((id_>>wheelStartBit_) & wheelMask_)+ minWheelId -1;
    }

    /// station id
    unsigned int station() const
      { return ((id_>>stationStartBit_) & stationMask_) ;}

    /// sector id
    unsigned int sector() const 
      { return ((id_>>sectorStartBit_)& sectorMask_) ;}

    /// sector id
    unsigned int superlayer() const 
      {return ((id_>>slayerStartBit_)&slMask_) ;}

    /// layer id
    unsigned int layer() const 
      { return ((id_>>layerStartBit_)&lMask_) ;}


    /// lowest wheel number
    static const int minWheelId=              -2;
    /// highest wheel number
    static const int maxWheelId=               2;
    /// lowest station id
    static const unsigned int minStationId=    1;
    /// highest station id
    static const unsigned int maxStationId=    4;
    /// lowest sector id
    static const unsigned int minSectorId=     1;
    /// highest sector id
    static const unsigned int maxSectorId=    12;
    /// loweset super layer id
    static const unsigned int minSuperLayerId= 1;
    /// highest superlayer id
    static const unsigned int maxSuperLayerId= 3;
    /// lowest layer id
    static const unsigned int minLayerId=      1;
    /// highest layer id
    static const unsigned int maxLayerId=      4;
 

  private:
    /// two bits would be enough, but  we could use the number "0" as a wildcard
    static const unsigned int layerNumBits_=   3;
    static const unsigned int layerStartBit_=  10;
    static const unsigned int slayerNumBits_=  2;
    static const unsigned int slayerStartBit_= layerStartBit_+ layerNumBits_;
    static const unsigned int sectorNumBits_=  4;
    static const unsigned int sectorStartBit_= slayerStartBit_+slayerNumBits_;
    /// two bits would be enough, but  we could use the number "0" as a wildcard
    static const unsigned int stationNumBits_= 3;
    static const unsigned int stationStartBit_=sectorStartBit_+sectorNumBits_;
    static const unsigned int wheelNumBits_  = 3;
    static const unsigned int wheelStartBit_=  stationStartBit_+stationNumBits_;

    static const unsigned int wheelMask_=    0x7;
    static const unsigned int stationMask_=  0x7;
    static const unsigned int sectorMask_=   0xf;
    static const unsigned int slMask_=       0x3;
    static const unsigned int lMask_=        0x7;

 

  }; // DTDetId

std::ostream& operator<<( std::ostream& os, const DTDetId& id );



#endif

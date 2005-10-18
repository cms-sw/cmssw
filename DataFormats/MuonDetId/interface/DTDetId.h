#ifndef MuonDetId_DTDetId_h
#define MuonDetId_DTDetId_h

/** \class DTDetId
 *  DetUnit identifier for DT chambers
 *
 *  $Date: $
 *  $Revision: $
 *  \author Stefano ARGIRO
 */

#include <DataFormats/MuonDetId/interface/MuonSubdetId.h>
#include <DataFormats/DetId/interface/DetId.h>

#include <iosfwd>

class DTDetId :public DetId {
  
 public:
      
  DTDetId();

  /// Construct from fully qualified identifier. Wire is optional since it
  /// is not reqired to identify a DetUnit, however it is part of the interface
  /// since it is required for the numbering schema.
  DTDetId(int wheel, 
	  unsigned int station, 
	  unsigned int sector,
	  unsigned int superlayer,
	  unsigned int layer,
	  unsigned int wire=0) :
    DetId(DetId::Muon, MuonSubdetId::DT ){ 
      unsigned int tmpwheelid = (unsigned int)(wheel- minWheelId +1);
      id_ |= (tmpwheelid& wheelMask_)  << wheelStartBit_     |
	(station & stationMask_)  << stationStartBit_   |
	(sector  &sectorMask_ )   << sectorStartBit_    |
	(superlayer & slMask_)    << slayerStartBit_    |
	(layer & lMask_)          << layerStartBit_     |
	(wire & wireMask_)          << wireStartBit_;
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

  /// wire id
  unsigned int wire() const 
  { return ((id_>>wireStartBit_)&wireMask_) ;}


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
  /// lowest layer id (numbering starts from 1 or 2)
  static const unsigned int minWireId=      1;
  /// highest wire id (chambers have 48 to 96 wires)
  static const unsigned int maxWireId=      97;
 

 private:
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  static const unsigned int wireNumBits_=   7;  
  static const unsigned int wireStartBit_=  3;
  static const unsigned int layerNumBits_=   3;
  static const unsigned int layerStartBit_=  wireStartBit_ + wireNumBits_;
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
  static const unsigned int wireMask_=    0x7f;
 

}; // DTDetId

std::ostream& operator<<( std::ostream& os, const DTDetId& id );

#endif

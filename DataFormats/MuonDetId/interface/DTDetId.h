#ifndef MuonDetId_DTDetId_h
#define MuonDetId_DTDetId_h

/** \class DTDetId
 *  DetUnit identifier for DT chambers
 *
 *  $Date: 2005/10/20 09:49:23 $
 *  $Revision: 1.6 $
 *  \author Stefano ARGIRO
 */

#include <DataFormats/MuonDetId/interface/MuonSubdetId.h>
#include <DataFormats/DetId/interface/DetId.h>

#include <iosfwd>

class DTDetId :public DetId {
  
 public:
      
  /// Default constructor; fills the common part in the base
  /// and leaves 0 in all other fields
  DTDetId();

  /// Construct from a packed id. It is required that the Detector part of
  /// id is Muon and the SubDet part is DT, otherwise an exception is thrown.
  explicit DTDetId(uint32_t id);

  /// Construct from fully qualified identifier. Wire is optional since it
  /// is not reqired to identify a DetUnit, however it is part of the interface
  /// since it is required for the numbering schema.
  /// Input values are required to be within legal ranges, otherwise an
  /// exception is thrown.
  DTDetId(int wheel, 
	  int station, 
	  int sector,
	  int superlayer,
	  int layer,
	  int wire=0);
  
  /// wheel id
  int wheel() const{
    return int((id_>>wheelStartBit_) & wheelMask_)+ minWheelId -1;
  }

  /// station id
  int station() const
  { return ((id_>>stationStartBit_) & stationMask_) ;}

  /// sector id
  int sector() const 
  { return ((id_>>sectorStartBit_)& sectorMask_) ;}

  /// sector id
  int superlayer() const 
  {return ((id_>>slayerStartBit_)&slMask_) ;}

  /// layer id
  int layer() const 
  { return ((id_>>layerStartBit_)&lMask_) ;}

  /// wire id
  int wire() const 
  { return ((id_>>wireStartBit_)&wireMask_) ;}


  /// lowest wheel number
  static const int minWheelId=              -2;
  /// highest wheel number
  static const int maxWheelId=               2;
  /// lowest station id
  static const int minStationId=    1;
  /// highest station id
  static const int maxStationId=    4;
  /// lowest sector id
  static const int minSectorId=     1;
  /// highest sector id
  static const int maxSectorId=    14;
  /// loweset super layer id. 0 indicates a full chamber
  static const int minSuperLayerId= 0;
  /// highest superlayer id
  static const int maxSuperLayerId= 3;
  /// lowest layer id. 0 indicates a full SL
  static const int minLayerId=      0;
  /// highest layer id
  static const int maxLayerId=      4;
  /// lowest wire id (numbering starts from 1 or 2). 0 indicates a full layer
  static const int minWireId=      0;
  /// highest wire id (chambers have 48 to 96 wires)
  static const int maxWireId=      97;
 

 private:
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  static const int wireNumBits_=   7;  
  static const int wireStartBit_=  3;
  static const int layerNumBits_=   3;
  static const int layerStartBit_=  wireStartBit_ + wireNumBits_;
  static const int slayerNumBits_=  2;
  static const int slayerStartBit_= layerStartBit_+ layerNumBits_;
  static const int sectorNumBits_=  4;
  static const int sectorStartBit_= slayerStartBit_+slayerNumBits_;
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  static const int stationNumBits_= 3;
  static const int stationStartBit_=sectorStartBit_+sectorNumBits_;
  static const int wheelNumBits_  = 3;
  static const int wheelStartBit_=  stationStartBit_+stationNumBits_;

  static const unsigned int wheelMask_=    0x7;
  static const unsigned int stationMask_=  0x7;
  static const unsigned int sectorMask_=   0xf;
  static const unsigned int slMask_=       0x3;
  static const unsigned int lMask_=        0x7;
  static const unsigned int wireMask_=    0x7f;
 

}; // DTDetId

std::ostream& operator<<( std::ostream& os, const DTDetId& id );

#endif

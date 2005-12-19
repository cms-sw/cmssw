#ifndef MuonDetId_DTChamberId_H
#define MuonDetId_DTChamberId_H

/** \class DTChamberId
 *  DetUnit identifier for DT chambers
 *
 *  $Date: $
 *  $Revision: $
 *  \author Stefano ARGIRO & G. Cerminara
 */

#include <DataFormats/MuonDetId/interface/MuonSubdetId.h>
#include <DataFormats/DetId/interface/DetId.h>

#include <iosfwd>

class DTChamberId : public DetId {
public:
  /// Default constructor. It fills the common part in the base
  /// and leaves 0 in all other fields
  DTChamberId();
  

  /// Construct from a packed id. It is required that the Detector part of
  /// id is Muon and the SubDet part is DT, otherwise an exception is thrown.
  explicit DTChamberId(uint32_t id);


  /// Construct from fully qualified identifier.
  /// Input values are required to be within legal ranges, otherwise an
  /// exception is thrown.
  DTChamberId(int wheel, 
	      int station, 
	      int sector);


  /// wheel id
  int wheel() const {
    return int((id_>>wheelStartBit_) & wheelMask_)+ minWheelId -1;
  }


  /// station id
  int station() const {
    return ((id_>>stationStartBit_) & stationMask_);
  }


  /// sector id
  int sector() const {
    return ((id_>>sectorStartBit_)& sectorMask_);
  }



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
 

 protected:
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

  static const uint32_t wheelMask_=    0x7;
  static const uint32_t stationMask_=  0x7;
  static const uint32_t sectorMask_=   0xf;
  static const uint32_t slMask_=       0x3;
  static const uint32_t lMask_=        0x7;
  static const uint32_t wireMask_=    0x7f;

  static const uint32_t layerIdMask_= ~(wireMask_<<wireStartBit_);
  static const uint32_t slIdMask_ = ~((wireMask_<<wireStartBit_) |
				      (lMask_<<layerStartBit_));
  static const uint32_t chamberIdMask_ = ~((wireMask_<<wireStartBit_) |
					   (lMask_<<layerStartBit_) |
					   (slMask_<<slayerStartBit_));
  
};


std::ostream& operator<<( std::ostream& os, const DTChamberId& id );

#endif


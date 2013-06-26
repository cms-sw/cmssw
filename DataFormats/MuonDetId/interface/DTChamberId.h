#ifndef MuonDetId_DTChamberId_H
#define MuonDetId_DTChamberId_H

/** \class DTChamberId
 *  DetUnit identifier for DT chambers.
 *  
 *  $Date: 2008/11/06 10:34:55 $
 *  $Revision: 1.10 $
 *  \author Stefano ARGIRO & G. Cerminara
 */

#include <DataFormats/DetId/interface/DetId.h>

#include <iosfwd>

class DTChamberId : public DetId {
public:
  /// Default constructor. 
  /// Fills the common part in the base and leaves 0 in all other fields
  DTChamberId();
  

  /// Construct from a packed id.
  /// It is required that the packed id represents a valid DT DetId
  /// (proper Detector and  SubDet fields), otherwise an exception is thrown.
  /// Any bits outside the DTChamberId fields are zeroed; apart for
  /// this, no check is done on the vaildity of the values.
  DTChamberId(uint32_t id);
  DTChamberId(DetId id);


  /// Construct from indexes.
  /// Input values are required to be within legal ranges, otherwise an
  /// exception is thrown.
  DTChamberId(int wheel, 
	      int station, 
	      int sector);


  /// Copy Constructor.
  /// Any bits outside the DTChamberId fields are zeroed; apart for
  /// this, no check is done on the vaildity of the values.
  DTChamberId(const DTChamberId& chId);


  /// Return the wheel number
  int wheel() const {
    return int((id_>>wheelStartBit_) & wheelMask_)+ minWheelId -1;
  }


  /// Return the station number
  int station() const {
    return ((id_>>stationStartBit_) & stationMask_);
  }


  /// Return the sector number. Sectors are numbered from 1 to 12,
  /// starting at phi=0 and increasing with phi.
  /// In station 4, where the top and bottom setcors are made of two chambers,
  /// two additional sector numbers are used, 13 (after sector 4, top)
  /// and 14 (after sector 10, bottom).
  int sector() const {
    return ((id_>>sectorStartBit_)& sectorMask_);
  }


  /// lowest station id
  static const int minStationId=    1;
  /// highest station id
  static const int maxStationId=    4;
  /// lowest sector id. 0 indicates all sectors (a station) 
  static const int minSectorId=     0;
  /// highest sector id.
  static const int maxSectorId=    14;
  /// lowest wheel number
  static const int minWheelId=     -2;
  /// highest wheel number
  static const int maxWheelId=      2;
  /// loweset super layer id. 0 indicates a full chamber
  static const int minSuperLayerId= 0;
  /// highest superlayer id
  static const int maxSuperLayerId= 3;
  /// lowest layer id. 0 indicates a full SL
  static const int minLayerId=      0;
  /// highest layer id
  static const int maxLayerId=      4;
  /// lowest wire id (numbering starts from 1 or 2). 0 indicates a full layer
  static const int minWireId=       0;
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
  static const int wheelNumBits_  = 3;
  static const int wheelStartBit_=  slayerStartBit_ + slayerNumBits_;
  static const int sectorNumBits_=  4;
  static const int sectorStartBit_= wheelStartBit_ + wheelNumBits_;
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  static const int stationNumBits_= 3;
  static const int stationStartBit_ = sectorStartBit_ + sectorNumBits_;


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

  // Perform a consistency check of the id with a DT Id
  // It thorows an exception if this is not the case
  void checkMuonId();
  
};


std::ostream& operator<<( std::ostream& os, const DTChamberId& id );

#endif


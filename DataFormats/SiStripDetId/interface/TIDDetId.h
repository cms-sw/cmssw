#ifndef DataFormats_SiStripDetId_TIDDetId_H
#define DataFormats_SiStripDetId_TIDDetId_H

#include <ostream>
#include <vector>
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

/** 
 * Det identifier class for the TIB
 */
class TIDDetId;

std::ostream& operator<<(std::ostream& os,const TIDDetId& id);

class TIDDetId : public SiStripDetId {
 public:
  /** Constructor of a null id */
  TIDDetId();
  /** Constructor from a raw value */
  TIDDetId(uint32_t rawid);
  /**Construct from generic DetId */
  TIDDetId(const DetId& id); 
  
  TIDDetId(uint32_t side,
	   uint32_t wheel,
	   uint32_t ring,
	   uint32_t module_fw_bw,
	   uint32_t module,
	   uint32_t ster) : SiStripDetId(DetId::Tracker,StripSubdetector::TID){
    id_ |= (side& sideMask_)      << sideStartBit_    |
      (wheel& wheelMask_)          << wheelStartBit_      |
      (ring& ringMask_)            << ringStartBit_       |
      (module_fw_bw& module_fw_bwMask_)  << module_fw_bwStartBit_  |
      (module& moduleMask_)              << moduleStartBit_        |
      (ster& sterMask_)            << sterStartBit_ ;
  }
  
  
  /// positive or negative id
  /**
   * side() = 1 The DetId identify a module in the negative part
   * side() = 2 The DetId identify a module in the positive part
   */
  unsigned int side() const{
    return int((id_>>sideStartBit_) & sideMask_);
  }
  
  /// wheel id
  unsigned int wheel() const{
    return int((id_>>wheelStartBit_) & wheelMask_);
  }
  
  ///ring id
  unsigned int ring() const
    { return ((id_>>ringStartBit_) & ringMask_) ;}
  
  /// det id
  /**
   * vector[0] = 1 -> back ring
   * vector[0] = 2 -> front ring
   * vector[1] -> module
   */
  std::vector<unsigned int> module() const
    { std::vector<unsigned int> num;
      num.push_back( order() );
      num.push_back( moduleNumber() );
      return num ;}
  
  unsigned int order() const 
  { return ((id_>>module_fw_bwStartBit_) & module_fw_bwMask_);}

  /** Returns true if the module is a double side = rphi + stereo */
  bool isDoubleSide() const;
  
  /** Returns true if the module is in TID+ (z>0 side) */
  bool isZPlusSide() const
  { return (!isZMinusSide());}
  
  /** Returns true if the module is in TID- (z<0 side) */
  bool isZMinusSide() const
  { return (side()==1);}
  
  /** Returns true if the ring is mounted on the disk back (not facing impact point) */
  bool isBackRing() const
  { return (order()==1);}
  
  /** Returns true if the ring is mounted on the disk front (facing impact point) */
  bool isFrontRing() const
  { return (!isBackRing());}
  
  /** Returns the disk number */
  unsigned int diskNumber() const
  { return wheel();}
  
  /** Returns the ring number */
  unsigned int ringNumber() const
  { return ring();}
  
  /** Returns the module number */
  unsigned int moduleNumber() const
  { return ((id_>>moduleStartBit_) & moduleMask_);}
  
  /** Returns true if the module is rphi */
  bool isRPhi()
  { return (stereo() == 0 && !isDoubleSide());}
  
  /** Returns true if the module is stereo */
  bool isStereo()
  { return (stereo() != 0 && !isDoubleSide());}
  
private:
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  static const unsigned int sideStartBit_=          13;
  static const unsigned int wheelStartBit_=         11;
  static const unsigned int ringStartBit_=          9;
  static const unsigned int module_fw_bwStartBit_=  7;
  static const unsigned int moduleStartBit_=        2;
  static const unsigned int sterStartBit_=          0;
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  static const unsigned int sideMask_=           0x3;
  static const unsigned int wheelMask_=          0x3;
  static const unsigned int ringMask_=           0x3;
  static const unsigned int module_fw_bwMask_=   0x3;
  static const unsigned int moduleMask_=         0x1F;
  static const unsigned int sterMask_=           0x3;
};


inline
TIDDetId::TIDDetId() : SiStripDetId() {
}
inline
TIDDetId::TIDDetId(uint32_t rawid) : SiStripDetId(rawid) {
}
inline
TIDDetId::TIDDetId(const DetId& id) : SiStripDetId(id.rawId()) {
}
inline
bool TIDDetId::isDoubleSide() const {
  // Double Side: only rings 1 and 2
  return this->glued() == 0 && ( this->ring() == 1 || this->ring() == 2 );
}




#endif



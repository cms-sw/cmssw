#ifndef DataFormats_SiStripDetId_TECDetId_H
#define DataFormats_SiStripDetId_TECDetId_H

#include <ostream>
#include <vector>
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

/** 
 *  Det identifier class for the TEC
 */
class TECDetId;

std::ostream& operator<<(std::ostream& s,const TECDetId& id);

class TECDetId : public SiStripDetId {
 public:
  /** Constructor of a null id */
  TECDetId();
  /** Constructor from a raw value */
  TECDetId(uint32_t rawid);
  /**Construct from generic DetId */
  TECDetId(const DetId& id); 
  
  TECDetId(uint32_t side,
	   uint32_t wheel,
	   uint32_t petal_fw_bw,
	   uint32_t petal,
	   uint32_t ring,
	   uint32_t module,
	   uint32_t ster) : SiStripDetId(DetId::Tracker,StripSubdetector::TEC){
    id_ |= (side& sideMask_)         << sideStartBit_ |
      (wheel& wheelMask_)             << wheelStartBit_ |
      (petal_fw_bw& petal_fw_bwMask_) << petal_fw_bwStartBit_ |
      (petal& petalMask_)             << petalStartBit_ |
      (ring& ringMask_)               << ringStartBit_ |
      (module& moduleMask_)                 << moduleStartBit_ |
      (ster& sterMask_)               << sterStartBit_ ;
  }
  
  
  /// positive or negative id
  /**
   * side() = 1 The DetId identify a module in the negative part (TEC-)
   * side() = 2 The DetId identify a module in the positive part (TEC+)
   */
  unsigned int side() const{
    return int((id_>>sideStartBit_) & sideMask_);
  }
  
  /// wheel id
  unsigned int wheel() const
    { return ((id_>>wheelStartBit_) & wheelMask_) ;}
  
  /// petal id
  /**
   * vector[0] = 1 -> back petal
   * vector[0] = 2 -> front petal
   * vector[1] -> petal
   */
  std::vector<unsigned int> petal() const
    { std::vector<unsigned int> num;
      num.push_back(order());
      num.push_back(petalNumber());
      return num ;}
  
  unsigned int order() const
  { return ((id_>>petal_fw_bwStartBit_) & petal_fw_bwMask_);}

  /// ring id
  unsigned int ring() const
    { return ((id_>>ringStartBit_) & ringMask_) ;}
  
  /// det id
  unsigned int module() const
    { return ((id_>>moduleStartBit_) & moduleMask_);}
  
  /** Returns true if the module is a double side = rphi + stereo */
  bool isDoubleSide() const;
  
  /** Returns true if the module is in TEC+ (z>0 side) */
  bool isZPlusSide() const
  { return (!isZMinusSide());}
  
  /** Returns true if the module is in TEC- (z<0 side) */
  bool isZMinusSide() const
  { return (side()==1);}
  
  /** Returns the wheel number */
  unsigned int wheelNumber() const
  { return wheel();}
  
  /** Returns the petal number */
  unsigned int petalNumber() const
  { return ((id_>>petalStartBit_) & petalMask_);}
  
  /** Returns the ring number */
  unsigned int ringNumber() const
  { return ring();}
  
  /** Returns the module number */
  unsigned int moduleNumber() const
  { return module();}
  
  /** Returns true if the petal is mounted on the wheel back (not facing impact point) */
  bool isBackPetal() const
  { return (order()==1);}
  
  /** Returns true if the petal is mounted on the wheel front (facing impact point) */
  bool isFrontPetal() const
  { return (!isBackPetal());}
  
  /** Returns true if the module is rphi */
  bool isRPhi()
  { return (stereo() == 0 && !isDoubleSide());}
  
  /** Returns true if the module is stereo */
  bool isStereo()
  { return (stereo() != 0 && !isDoubleSide());}
  
private:
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  static const unsigned int sideStartBit_=           18;
  static const unsigned int wheelStartBit_=          14;  
  static const unsigned int petal_fw_bwStartBit_=    12;
  static const unsigned int petalStartBit_=          8;
  static const unsigned int ringStartBit_=           5;
  static const unsigned int moduleStartBit_=         2;
  static const unsigned int sterStartBit_=           0;
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  static const unsigned int sideMask_=          0x3;
  static const unsigned int wheelMask_=         0xF;
  static const unsigned int petal_fw_bwMask_=   0x3;
  static const unsigned int petalMask_=         0xF;
  static const unsigned int ringMask_=          0x7;
  static const unsigned int moduleMask_=        0x7;
  static const unsigned int sterMask_=          0x3;
};


inline
TECDetId::TECDetId() : SiStripDetId() {
}
inline
TECDetId::TECDetId(uint32_t rawid) : SiStripDetId(rawid) {
}
inline
TECDetId::TECDetId(const DetId& id) : SiStripDetId(id.rawId()){
}

inline
bool TECDetId::isDoubleSide() const {
  // Double Side: only rings 1, 2 and 5
  return this->glued() == 0 && ( this->ring() == 1 || this->ring() == 2 || this->ring() == 5 ) ;
}


#endif

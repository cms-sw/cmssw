#ifndef DataFormats_SiStripDetId_TOBDetId_H
#define DataFormats_SiStripDetId_TOBDetId_H

#include <ostream>
#include <vector>
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

/** 
 *  Det identifier class for the TOB
 */
class TOBDetId;

std::ostream& operator<<(std::ostream& os,const TOBDetId& id);

class TOBDetId : public SiStripDetId {
 public:
  /** Constructor of a null id */
  TOBDetId();
  /** Constructor from a raw value */
  TOBDetId(uint32_t rawid);
  /**Construct from generic DetId */
  TOBDetId(const DetId& id); 
  
  TOBDetId(uint32_t layer,
	   uint32_t rod_fw_bw,
	   uint32_t rod,
	   uint32_t module,
	   uint32_t ster) : SiStripDetId(DetId::Tracker,StripSubdetector::TOB){
    id_ |= (layer& layerMask_) << layerStartBit_ |
      (rod_fw_bw& rod_fw_bwMask_) << rod_fw_bwStartBit_ |
      (rod& rodMask_) << rodStartBit_ |
      (module& moduleMask_) << moduleStartBit_ |
      (ster& sterMask_) << sterStartBit_ ;
  }
  
  
  /// layer id
  unsigned int layer() const{
    return int((id_>>layerStartBit_) & layerMask_);
  }
  
  /// rod id
  /**
   * vector[0] = 1 -> bw rod (TOB-)
   * vector[0] = 2 -> fw rod (TOB+)
   * vector[1] -> rod
   */
  std::vector<unsigned int> rod() const
    { std::vector<unsigned int> num;
      num.push_back( side() );
      num.push_back( rodNumber() );
      return num ;}
  
  unsigned int side() const
  { return ((id_>>rod_fw_bwStartBit_) & rod_fw_bwMask_);}
  /// detector id
  unsigned int module() const 
    { return ((id_>>moduleStartBit_)& moduleMask_) ;}
  
  /** Returns true if the module is a double side = rphi + stereo */
  bool isDoubleSide() const;
  
  /** Returns true if the module is in TOB+ (z>0 side) */
  bool isZPlusSide() const
  { return (!isZMinusSide());}
  
  /** Returns true if the module is in TOB- (z<0 side) */
  bool isZMinusSide() const
  { return (side() == 1);}
  
  /** Returns the layer number */
  unsigned int layerNumber() const
  { return layer();}
  
  /** Returns the rod number */
  unsigned int rodNumber() const
  { return ((id_>>rodStartBit_) & rodMask_);}
  
  /** Returns the module number */
  unsigned int moduleNumber() const
  { return module();}
  
  /** Returns true if the module is rphi */
  bool isRPhi()
  { return (stereo() == 0 && !isDoubleSide());}
  
  /** Returns true if the module is stereo */
  bool isStereo()
  { return (stereo() != 0 && !isDoubleSide());}
  
private:
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  static const unsigned int layerStartBit_=     14;
  static const unsigned int rod_fw_bwStartBit_= 12;
  static const unsigned int rodStartBit_=       5;
  static const unsigned int moduleStartBit_=    2;
  static const unsigned int sterStartBit_=      0;
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  
  static const unsigned int layerMask_=       0x7;
  static const unsigned int rod_fw_bwMask_=   0x3;
  static const unsigned int rodMask_=         0x7F;
  static const unsigned int moduleMask_=      0x7;
  static const unsigned int sterMask_=        0x3;
};


inline
TOBDetId::TOBDetId() : SiStripDetId() {
}
inline
TOBDetId::TOBDetId(uint32_t rawid) : SiStripDetId(rawid) {
}
inline
TOBDetId::TOBDetId(const DetId& id) : SiStripDetId(id.rawId()) {
}
inline
bool TOBDetId::isDoubleSide() const {
  // Double Side: only layers 1 and 2
  return this->glued() == 0 && ( this->layer() == 1 || this->layer() == 2 );
}


#endif

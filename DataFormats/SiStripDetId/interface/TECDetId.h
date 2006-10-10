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
   * side() = 1 The DetId identify a module in the negative part
   * side() = 2 The DetId identify a module in the positive part
   */
  unsigned int side() const{
    return int((id_>>sideStartBit_) & sideMask_);
  }
  
  /// wheel id
  unsigned int wheel() const
    { return ((id_>>wheelStartBit_) & wheelMask_) ;}
  
  /// petal id
  /**
   * vector[0] = 0 -> bw petal
   * vector[0] = 1 -> fw petal
   * vector[1] -> petal
   */
  std::vector<unsigned int> petal() const
    { std::vector<unsigned int> num;
    num.push_back(((id_>>petal_fw_bwStartBit_) & petal_fw_bwMask_));
    num.push_back(((id_>>petalStartBit_) & petalMask_));
    return num ;}
  
  /// ring id
  unsigned int ring() const
    { return ((id_>>ringStartBit_) & ringMask_) ;}
  
  /// det id
  unsigned int module() const
    { return ((id_>>moduleStartBit_) & moduleMask_);}
  
    /// glued
  /**
   * glued() = 0 it's not a glued module
   * glued() != 0 it's a glued module
   */
  unsigned int glued() const
    {
      if(((id_>>sterStartBit_)& sterMask_) == 1){
	return (id_ -1);
      }else if(((id_>>sterStartBit_)& sterMask_) == 2){
	return (id_ -2);
      }else{
	return 0;
      }
    }

  
  /// stereo 
  /**
   * stereo() = 0 it's not a stereo module
   * stereo() = 1 it's a stereo module
   */
  unsigned int stereo() const 
    {
      if(((id_>>sterStartBit_)& sterMask_)==1){
	return ((id_>>sterStartBit_)& sterMask_);
      }else{
	return 0;
      }
    }

  /**
   * If the DetId identify a glued module return 
   * the DetId of your partner otherwise return 0
   */
  unsigned int partnerDetId() const
    {
      if(((id_>>sterStartBit_)& sterMask_)==1){
	return (id_ + 1);
      }else if(((id_>>sterStartBit_)& sterMask_)==2){
	return (id_ - 1);
      }else{
	return 0;
      }
    }

 private:
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  static const unsigned int sideStartBit_=           23;
  static const unsigned int wheelStartBit_=          16;  
  static const unsigned int petal_fw_bwStartBit_=    15;
  static const unsigned int petalStartBit_=          8;
  static const unsigned int ringStartBit_=           5;
  //  static const unsigned int module_fw_bwStartBit_=   4;
  static const unsigned int moduleStartBit_=         2;
  static const unsigned int sterStartBit_=           0;
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  static const unsigned int sideMask_=          0x3;
  static const unsigned int wheelMask_=         0xF;
  static const unsigned int petal_fw_bwMask_=   0x1;
  static const unsigned int petalMask_=         0x7F;
  static const unsigned int ringMask_=          0x7;
  //  static const unsigned int module_fw_bwMask_=  0x1;
  static const unsigned int moduleMask_=        0x7;
  static const unsigned int sterMask_=          0x3;
};


#endif

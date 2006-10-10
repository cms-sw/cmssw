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
   * vector[0] = 0 -> bw module
   * vector[0] = 1 -> fw module
   * vector[1] -> module
   */
  std::vector<unsigned int> module() const
    { std::vector<unsigned int> num;
    num.push_back(((id_>>module_fw_bwStartBit_) & module_fw_bwMask_));
    num.push_back(((id_>>moduleStartBit_) & moduleMask_));
    return num ;}
  
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
  static const unsigned int sideStartBit_=          23;
  static const unsigned int wheelStartBit_=         16;
  static const unsigned int ringStartBit_=          8;
  static const unsigned int module_fw_bwStartBit_=  7;
  static const unsigned int moduleStartBit_=        2;
  static const unsigned int sterStartBit_=          0;
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  
  static const unsigned int sideMask_=           0x3;
  static const unsigned int wheelMask_=          0xF;
  static const unsigned int ringMask_=           0xFF;
  static const unsigned int module_fw_bwMask_=   0x1;
  static const unsigned int moduleMask_=         0x1F;
  static const unsigned int sterMask_=           0x3;
};


#endif



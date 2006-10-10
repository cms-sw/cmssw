#ifndef DataFormats_SiStripDetId_TIBDetId_H
#define DataFormats_SiStripDetId_TIBDetId_H

#include <ostream>
#include <vector>
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

/** 
 *  Det identifier class for the TIB
 */
class TIBDetId;

std::ostream& operator<<(std::ostream& os,const TIBDetId& id);

class TIBDetId : public SiStripDetId {
 public:
  /** Constructor of a null id */
  TIBDetId();
  /** Constructor from a raw value */
  TIBDetId(uint32_t rawid);
  /**Construct from generic DetId */
  TIBDetId(const DetId& id); 
  
  TIBDetId(uint32_t layer,
	   uint32_t str_fw_bw,
	   uint32_t str_int_ext,
	   uint32_t str,
	   uint32_t module,
	   uint32_t ster) : SiStripDetId(DetId::Tracker,StripSubdetector::TIB){
    id_ |= (layer& layerMask_) << layerStartBit_ |
      (str_fw_bw& str_fw_bwMask_) << str_fw_bwStartBit_ |
      (str_int_ext& str_int_extMask_) << str_int_extStartBit_ |
      (str& strMask_) << strStartBit_ |
      (module& moduleMask_) << moduleStartBit_ |
      (ster& sterMask_) << sterStartBit_ ;
  }
  
  
  /// layer id
  unsigned int layer() const{
    return int((id_>>layerStartBit_) & layerMask_);
  }
  
  /// string  id
  /**
   * vector[0] = 0 -> bw string
   * vector[0] = 1 -> fw string
   * vector[1] = 0 -> int string
   * vector[1] = 1 -> ext string
   * vector[2] -> string
   */
  std::vector<unsigned int> string() const
    { std::vector<unsigned int> num;
    num.push_back(((id_>>str_fw_bwStartBit_) & str_fw_bwMask_));
    num.push_back(((id_>>str_int_extStartBit_) & str_int_extMask_));
    num.push_back(((id_>>strStartBit_) & strMask_));
    return num ;}
  
  /// detector id
  unsigned int module() const 
    { return ((id_>>moduleStartBit_)& moduleMask_) ;}
  /// glued
  /**
   * glued() = 0 it's not a glued module
   * glued() != 0 it's a glued module
   */
  unsigned int glued() const
    {
      if(((id_>>sterStartBit_)& sterMask_) == 1 ){
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
  static const unsigned int layerStartBit_=           16;
  static const unsigned int str_fw_bwStartBit_=       15;
  static const unsigned int str_int_extStartBit_=     14;
  static const unsigned int strStartBit_=             8;
  static const unsigned int moduleStartBit_=          2;
  static const unsigned int sterStartBit_=            0;
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  
  static const unsigned int layerMask_=       0xF;
  static const unsigned int str_fw_bwMask_=   0x1;
  static const unsigned int str_int_extMask_= 0x1;
  static const unsigned int strMask_=         0x3F;
  static const unsigned int moduleMask_=      0x3F;
  static const unsigned int sterMask_=        0x3;
};


#endif

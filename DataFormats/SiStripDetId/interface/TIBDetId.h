#ifndef DataFormats_SiStripDetId_TIBDetId_H
#define DataFormats_SiStripDetId_TIBDetId_H

#include <ostream>
#include <vector>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

/** 
 *  Det identifier class for the TIB
 */

namespace cms
{

  class TIBDetId : public DetId {
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
	     uint32_t det,
	     uint32_t ster) : DetId(cms::DetId::Tracker,StripSubdetector::TIB){
      id_ |= (layer& layerMask_) << layerStartBit_ |
	     (str_fw_bw& str_fw_bwMask_) << str_fw_bwStartBit_ |
	     (str_int_ext& str_int_extMask_) << str_int_extStartBit_ |
	     (str& strMask_) << strStartBit_ |
	     (det& detMask_) << detStartBit_ |
	     (ster& sterMask_) << sterStartBit_ ;
    }
 

    /// layer id
    unsigned int layer() const{
      return int((id_>>layerStartBit_) & layerMask_);
    }

    /// string  id
    /**
     * vector[0] = 0 -> fw string
     * vector[0] = 1 -> bw string
     * vector[1] = 0 -> int string
     * vector[1] = 1 -> ext string
     * vector[2] -> string
     */
    std::vector<unsigned int> string() const
      { std::vector<unsigned int> num;
      num[0]=((id_>>str_fw_bwStartBit_) & str_fw_bwMask_);
      num[1]=((id_>>str_int_extStartBit_) & str_int_extMask_);
      num[2]=((id_>>strStartBit_) & strMask_);
      return num ;}
    
    /// detector id
    unsigned int det() const 
      { return ((id_>>detStartBit_)& detMask_) ;}

    /// stereo id
    unsigned int stereo() const 
      { return ((id_>>sterStartBit_)& sterMask_) ;}

  private:
    /// two bits would be enough, but  we could use the number "0" as a wildcard
    static const unsigned int layerStartBit_=           16;
    static const unsigned int str_fw_bwStartBit_=       15;
    static const unsigned int str_int_extStartBit_=     14;
    static const unsigned int strStartBit_=             8;
    static const unsigned int detStartBit_=             2;
    static const unsigned int sterStartBit_=            0;
    /// two bits would be enough, but  we could use the number "0" as a wildcard

    static const unsigned int layerMask_=       0xF;
    static const unsigned int str_fw_bwMask_=   0x1;
    static const unsigned int str_int_extMask_= 0x1;
    static const unsigned int strMask_=         0x3F;
    static const unsigned int detMask_=         0x3F;
    static const unsigned int sterMask_=        0x3;
  };

  std::ostream& operator<<(std::ostream& s,const TIBDetId& id);
  
}


#endif

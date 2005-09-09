#ifndef DataFormats_SiStripDetId_TOBDetId_H
#define DataFormats_SiStripDetId_TOBDetId_H

#include <ostream>
#include <vector>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

/** 
 *  Det identifier class for the TOB
 */

namespace cms
{

  class TOBDetId : public DetId {
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
	     uint32_t det,
	     uint32_t ster) : DetId(cms::DetId::Tracker,StripSubdetector::TOB){
      id_ |= (layer& layerMask_) << layerStartBit_ |
	     (rod_fw_bw& rod_fw_bwMask_) << rod_fw_bwStartBit_ |
	     (rod& rodMask_) << rodStartBit_ |
	     (det& detMask_) << detStartBit_ |
	     (ster& sterMask_) << sterStartBit_ ;
    }
 

    /// layer id
    unsigned int layer() const{
      return int((id_>>layerStartBit_) & layerMask_);
    }

    /// rod id
    /**
     * vector[0] = 0 -> fw rod
     * vector[0] = 1 -> bw rod
     * vector[1] -> rod
     */
    std::vector<unsigned int> rod() const
      { std::vector<unsigned int> num;
      num.push_back(((id_>>rod_fw_bwStartBit_) & rod_fw_bwMask_));
      num.push_back(((id_>>rodStartBit_) & rodMask_));
      return num ;}

    /// det id
    unsigned int det() const 
      { return ((id_>>detStartBit_)& detMask_) ;}

    /// stereo id
    unsigned int stereo() const 
      { return ((id_>>sterStartBit_)& sterMask_) ;}

  private:
    /// two bits would be enough, but  we could use the number "0" as a wildcard
    static const unsigned int layerStartBit_=     16;
    static const unsigned int rod_fw_bwStartBit_= 15;
    static const unsigned int rodStartBit_=        8;
    static const unsigned int detStartBit_=        2;
    static const unsigned int sterStartBit_=       0;
    /// two bits would be enough, but  we could use the number "0" as a wildcard

    static const unsigned int layerMask_=       0xF;
    static const unsigned int rod_fw_bwMask_=   0x1;
    static const unsigned int rodMask_=         0x7F;
    static const unsigned int detMask_=         0x3F;
    static const unsigned int sterMask_=        0x3;
  };

  std::ostream& operator<<(std::ostream& s,const TOBDetId& id);
  
}


#endif

#ifndef DataFormats_SiStripDetId_TIDDetId_H
#define DataFormats_SiStripDetId_TIDDetId_H

#include <ostream>
#include <vector>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

/** 
 * Det identifier class for the TIB
 */

namespace cms
{

  class TIDDetId : public DetId {
  public:
    /** Constructor of a null id */
    TIDDetId();
    /** Constructor from a raw value */
    TIDDetId(uint32_t rawid);
    /**Construct from generic DetId */
    TIDDetId(const DetId& id); 
 
    TIDDetId(uint32_t pos_neg,
	     uint32_t whell,
	     uint32_t ring,
	     uint32_t det_fw_bw,
	     uint32_t det,
	     uint32_t ster) : DetId(cms::DetId::Tracker,StripSubdetector::TID){
      id_ |= (pos_neg& pos_negMask_)      << pos_negStartBit_    |
	     (whell& whellMask_)          << whellStartBit_      |
	     (det_fw_bw& det_fw_bwMask_)  << det_fw_bwStartBit_  |
	     (det& detMask_)              << detStartBit_        |
	     (ster& sterMask_)            << sterStartBit_ ;
    }
 

    /// positive or negative id
    /**
     * posNeg() = 1 The DetId identify a module in the negative part
     * posNeg() = 2 The DetId identify a module in the positive part
     */
    unsigned int posNeg() const{
      return int((id_>>pos_negStartBit_) & pos_negMask_);
    }

    /// whell id
    unsigned int whell() const{
      return int((id_>>whellStartBit_) & whellMask_);
    }

    ///ring id
    unsigned int ring() const
      { return ((id_>>ringStartBit_) & ringMask_) ;}

    /// det id
    /**
     * vector[0] = 0 -> fw det
     * vector[0] = 1 -> bw det
     * vector[1] -> det
     */
    std::vector<unsigned int> det() const
      { std::vector<unsigned int> num;
      num.push_back(((id_>>det_fw_bwStartBit_) & det_fw_bwMask_));
      num.push_back(((id_>>detStartBit_) & detMask_));
      return num ;}

    /// glued
    /**
     * glued() = 0 it's not a glued module
     * glued() = 1 it's a glued module
     */
    unsigned int glued() const
      {
	if(((id_>>sterStartBit_)& sterMask_) == 1 ||
	   ((id_>>sterStartBit_)& sterMask_) == 2){
	  return 1;
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
    static const unsigned int pos_negStartBit_=   23;
    static const unsigned int whellStartBit_=     16;
    static const unsigned int ringStartBit_=       8;
    static const unsigned int det_fw_bwStartBit_=  7;
    static const unsigned int detStartBit_=        2;
    static const unsigned int sterStartBit_=       0;
    /// two bits would be enough, but  we could use the number "0" as a wildcard

    static const unsigned int pos_negMask_=     0x3;
    static const unsigned int whellMask_=       0xF;
    static const unsigned int ringMask_=        0xFF;
    static const unsigned int det_fw_bwMask_=   0x1;
    static const unsigned int detMask_=         0x1F;
    static const unsigned int sterMask_=        0x3;
  };

  std::ostream& operator<<(std::ostream& s,const TIDDetId& id);
  
}


#endif

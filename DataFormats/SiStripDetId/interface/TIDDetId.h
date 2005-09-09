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

    /// stereo id
    unsigned int stereo() const 
      { return ((id_>>sterStartBit_)& sterMask_) ;}

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

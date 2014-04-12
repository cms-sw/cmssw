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
   * vector[0] = 1 -> bw string (TIB-)
   * vector[0] = 2 -> fw string (TIB+)
   * vector[1] = 1 -> int string
   * vector[1] = 2 -> ext string
   * vector[2] -> string
   */
  std::vector<unsigned int> string() const
    { std::vector<unsigned int> num;
      num.push_back( side() );
      num.push_back( order() );
      num.push_back(stringNumber());
      return num ;}
  
  /// detector id
  unsigned int module() const 
    { return ((id_>>moduleStartBit_)& moduleMask_) ;}
  
  unsigned int order()const
  { return ((id_>>str_int_extStartBit_) & str_int_extMask_);}

  unsigned int side() const
  {return ((id_>>str_fw_bwStartBit_) & str_fw_bwMask_);}


  /** Returns true if the module is a double side = rphi + stereo */
  bool isDoubleSide() const;
  
  /** Returns true if the module is in TIB+ (z>0 side) */
  bool isZPlusSide() const
  { return (!isZMinusSide());}
  
  /** Returns true if the module is in TIB- (z<0 side) */
  bool isZMinusSide() const
  { return (side() == 1);}
  
  /** Returns the layer number */
  unsigned int layerNumber() const
  { return layer();}
  
  /** Returns the string number */
  unsigned int stringNumber() const
  { return ((id_>>strStartBit_) & strMask_);}
  
  /** Returns the module number */
  unsigned int moduleNumber() const
  { return module();}
  
  /** Returns true if the module is in internal part of the layer (smaller radius) */
  bool isInternalString() const
  { return (order() == 1);}
  
  /** Returns true if the module is in external part of the layer (bigger radius) */
  bool isExternalString() const
  { return (!isInternalString());}
  
  /** Returns true if the module is rphi */
  bool isRPhi()
  { return (stereo() == 0 && !isDoubleSide());}
  
  /** Returns true if the module is stereo */
  bool isStereo()
  { return (stereo() != 0 && !isDoubleSide());}
  
  
private:
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  static const unsigned int layerStartBit_=           14;
  static const unsigned int str_fw_bwStartBit_=       12;
  static const unsigned int str_int_extStartBit_=     10;
  static const unsigned int strStartBit_=             4;
  static const unsigned int moduleStartBit_=          2;
  static const unsigned int sterStartBit_=            0;
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  
  static const unsigned int layerMask_=       0x7;
  static const unsigned int str_fw_bwMask_=   0x3;
  static const unsigned int str_int_extMask_= 0x3;
  static const unsigned int strMask_=         0x3F;
  static const unsigned int moduleMask_=      0x3;
  static const unsigned int sterMask_=        0x3;
};


inline
TIBDetId::TIBDetId() : SiStripDetId(){
}
inline
TIBDetId::TIBDetId(uint32_t rawid) : SiStripDetId(rawid){
}
inline
TIBDetId::TIBDetId(const DetId& id) : SiStripDetId(id.rawId()){
}
inline
bool TIBDetId::isDoubleSide() const {
  // Double Side: only layers 1 and 2
  return this->glued() == 0 && ( this->layer() == 1 || this->layer() == 2 );
}


#endif

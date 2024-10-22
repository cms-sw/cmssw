#ifndef DataFormats_SiStripDetId_PXBDetId_H
#define DataFormats_SiStripDetId_PXBDetId_H

#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

/** 
 *  Det identifier class for the PixelBarrel
 */

class PXBDetId;

std::ostream& operator<<(std::ostream& os, const PXBDetId& id);

class PXBDetId : public DetId {
public:
  /** Constructor of a null id */
  PXBDetId();
  /** Constructor from a raw value */
  PXBDetId(uint32_t rawid);
  /**Construct from generic DetId */
  PXBDetId(const DetId& id);

  PXBDetId(uint32_t layer, uint32_t ladder, uint32_t module) : DetId(DetId::Tracker, PixelSubdetector::PixelBarrel) {
    id_ |= (layer & layerMask_) << layerStartBit_ | (ladder & ladderMask_) << ladderStartBit_ |
           (module & moduleMask_) << moduleStartBit_;
  }

  /// layer id
  unsigned int layer() const { return int((id_ >> layerStartBit_) & layerMask_); }

  /// ladder  id
  unsigned int ladder() const { return ((id_ >> ladderStartBit_) & ladderMask_); }

  /// det id
  unsigned int module() const { return ((id_ >> moduleStartBit_) & moduleMask_); }

private:
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  static const unsigned int layerStartBit_ = 16;
  static const unsigned int ladderStartBit_ = 8;
  static const unsigned int moduleStartBit_ = 2;
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  static const unsigned int layerMask_ = 0xF;
  static const unsigned int ladderMask_ = 0xFF;
  static const unsigned int moduleMask_ = 0x3F;
};

#endif

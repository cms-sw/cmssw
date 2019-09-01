#ifndef DataFormats_SiStripDetId_PXFDetId_H
#define DataFormats_SiStripDetId_PXFDetId_H

#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

/** 
 *  Det identifier class for the PixelEndcap
 */
class PXFDetId;

std::ostream& operator<<(std::ostream& os, const PXFDetId& id);

class PXFDetId : public DetId {
public:
  /** Constructor of a null id */
  PXFDetId();
  /** Constructor from a raw value */
  PXFDetId(uint32_t rawid);
  /**Construct from generic DetId */
  PXFDetId(const DetId& id);

  PXFDetId(uint32_t side, uint32_t disk, uint32_t blade, uint32_t panel, uint32_t module)
      : DetId(DetId::Tracker, PixelSubdetector::PixelEndcap) {
    id_ |= (side & sideMask_) << sideStartBit_ | (disk & diskMask_) << diskStartBit_ |
           (blade & bladeMask_) << bladeStartBit_ | (panel & panelMask_) << panelStartBit_ |
           (module & moduleMask_) << moduleStartBit_;
  }

  /// positive or negative id
  unsigned int side() const { return int((id_ >> sideStartBit_) & sideMask_); }

  /// disk id
  unsigned int disk() const { return int((id_ >> diskStartBit_) & diskMask_); }

  /// blade id
  unsigned int blade() const { return ((id_ >> bladeStartBit_) & bladeMask_); }

  /// panel id
  unsigned int panel() const { return ((id_ >> panelStartBit_) & panelMask_); }

  /// det id
  unsigned int module() const { return ((id_ >> moduleStartBit_) & moduleMask_); }

private:
  /// two bits would be enough, but  we could use the number "0" as a wildcard
  static const unsigned int sideStartBit_ = 23;
  static const unsigned int diskStartBit_ = 16;
  static const unsigned int bladeStartBit_ = 10;
  static const unsigned int panelStartBit_ = 8;
  static const unsigned int moduleStartBit_ = 2;
  /// two bits would be enough, but  we could use the number "0" as a wildcard

  static const unsigned int sideMask_ = 0x3;
  static const unsigned int diskMask_ = 0xF;
  static const unsigned int bladeMask_ = 0x3F;
  static const unsigned int panelMask_ = 0x3;
  static const unsigned int moduleMask_ = 0x3F;
};

#endif

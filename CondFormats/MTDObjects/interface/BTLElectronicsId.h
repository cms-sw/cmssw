#ifndef CondFormats_MTDObjects_BTLElectronicsId_h
#define CondFormats_MTDObjects_BTLElectronicsId_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <cstdint>
#include <iosfwd>

class BTLElectronicsId {
public:
  // Bit layout:
  //
  //  0 -  4 : channelId 0..31  (5 bits)
  //  5 -  9 : eLinkId   0..23  (5 bits)
  // 10 - 16 : hsLinkId  0..127 (7 bits)
  // 17 - 29 : fedId     ?      (13 bits)
  // 30 - 31 : free for now

  static constexpr uint32_t kChannelMask = 0x1F;  // 5 bits
  static constexpr uint32_t kELinkMask = 0x1F;    // 5 bits
  static constexpr uint32_t kHSLinkMask = 0x7F;   // 7 bits
  static constexpr uint32_t kFEDMask = 0x1FFF;    // 13 bits

  static constexpr unsigned kChannelShift = 0;
  static constexpr unsigned kELinkShift = 5;
  static constexpr unsigned kHSLinkShift = 10;
  static constexpr unsigned kFEDShift = 17;

  /** Default constructor **/
  BTLElectronicsId();

  /** Constructor from rawId **/
  explicit BTLElectronicsId(uint32_t rawid);

  /** Constructor from (FED id, HS-link id, e-link id, tofhir channel id) **/
  BTLElectronicsId(uint16_t fedId,  // sLinkId
                   uint8_t hsLinkId,
                   uint8_t eLinkId,
                   uint8_t channelId);

  /// Accessors
  int fedId() const { return (rawid_ >> kFEDShift) & kFEDMask; };
  int hsLinkId() const { return (rawid_ >> kHSLinkShift) & kHSLinkMask; };
  int eLinkId() const { return (rawid_ >> kELinkShift) & kELinkMask; };
  int channelId() const { return (rawid_ >> kChannelShift) & kChannelMask; }

  uint32_t rawId() const { return rawid_; };

  bool operator==(const BTLElectronicsId&) const;
  bool operator!=(const BTLElectronicsId&) const;

private:
  uint32_t rawid_ = 0;

  COND_SERIALIZABLE;
};

std::ostream& operator<<(std::ostream&, const BTLElectronicsId&);

#endif

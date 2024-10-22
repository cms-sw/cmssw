/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *
 ****************************************************************************/

#ifndef DataFormats_CTPPSDetId_TotemT2DetId
#define DataFormats_CTPPSDetId_TotemT2DetId

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

#include <iosfwd>
#include <string>

/**
 *\brief Detector ID class for Totem T2 detectors.
 * Bits [19:31] : Base CTPPSDetId class attributes
 * Bits [16:18] : 3 bits for T2 plane [0-7]
 * Bits [14:15] : 2 bits for T2 tile numbers [0-3]
 * Bits [0:13]  : unspecified yet
 **/

class TotemT2DetId : public CTPPSDetId {
public:
  /// Construct from a raw id
  explicit TotemT2DetId(uint32_t id);
  TotemT2DetId(const CTPPSDetId& id) : CTPPSDetId(id) {}

  /// Construct from hierarchy indices.
  TotemT2DetId(uint32_t arm, uint32_t plane, uint32_t channel = 0);

  static constexpr uint32_t startPlaneBit = 16, maskPlane = 0x7, maxPlane = 7, lowMaskPlane = 0xffff;
  static constexpr uint32_t startChannelBit = 14, maskChannel = 0x3, maxChannel = 3, lowMaskChannel = 0x1fff;

  /// returns true if the raw ID is a PPS-timing one
  static bool check(unsigned int raw) {
    return (((raw >> DetId::kDetOffset) & 0xF) == DetId::VeryForward &&
            ((raw >> DetId::kSubdetOffset) & 0x7) == sdTotemT2);
  }
  //-------------------- getting and setting methods --------------------

  uint32_t plane() const { return ((id_ >> startPlaneBit) & maskPlane); }

  void setPlane(uint32_t channel) {
    id_ &= ~(maskPlane << startPlaneBit);
    id_ |= ((channel & maskPlane) << startPlaneBit);
  }

  uint32_t channel() const { return ((id_ >> startChannelBit) & maskChannel); }

  void setChannel(uint32_t channel) {
    id_ &= ~(maskChannel << startChannelBit);
    id_ |= ((channel & maskChannel) << startChannelBit);
  }

  //-------------------- id getters for higher-level objects --------------------

  TotemT2DetId planeId() const { return TotemT2DetId(rawId() & (~lowMaskPlane)); }

  //-------------------- name methods --------------------

  void planeName(std::string& name, NameFlag flag = nFull) const;
  void channelName(std::string& name, NameFlag flag = nFull) const;
};

std::ostream& operator<<(std::ostream& os, const TotemT2DetId& id);

namespace std {
  template <>
  struct hash<TotemT2DetId> {
    typedef TotemT2DetId argument_type;
    typedef std::size_t result_type;
    result_type operator()(const argument_type& id) const noexcept { return std::hash<uint64_t>()(id.rawId()); }
  };
}  // namespace std

#endif

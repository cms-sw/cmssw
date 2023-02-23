#ifndef DataFormats_SiStripCluster_interface_SiStripTypes_h
#define DataFormats_SiStripCluster_interface_SiStripTypes_h

#include <cstdint>
#include <limits>

namespace stripgpu {
  using detId_t = std::uint32_t;
  using fedId_t = std::uint16_t;
  using fedCh_t = std::uint8_t;
  using apvPair_t = std::uint16_t;
  using stripId_t = std::uint16_t;

  static constexpr detId_t invalidDet = std::numeric_limits<detId_t>::max();
  static constexpr fedId_t invalidFed = std::numeric_limits<fedId_t>::max();
  static constexpr stripId_t invalidStrip = std::numeric_limits<stripId_t>::max();
}  // namespace stripgpu

#endif

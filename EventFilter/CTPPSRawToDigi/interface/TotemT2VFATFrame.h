/****************************************************************************
*
* This is a part of the TOTEM offline software.
* Authors:
*   Edoardo Bossini
*   Laurent Forthomme
*
****************************************************************************/

#ifndef EventFilter_CTPPSRawToDigi_TotemT2VFATFrame
#define EventFilter_CTPPSRawToDigi_TotemT2VFATFrame

#include "EventFilter/CTPPSRawToDigi/interface/VFATFrame.h"

#include <cstdint>

/**
 * Utilitary namespace to retrieve timing/status information from nT2 VFAT frame
**/
namespace totem::nt2::vfat {
  /// multiplicity of 32-bit words combined into a single channel payload
  static constexpr size_t num_words_per_channel = 4;
  /// multiplicity of channels combined into a single payload
  static constexpr size_t num_channels_per_payload = 2;
  static constexpr size_t header_offset = 3;
  /// get timing information for single leading edge
  inline uint16_t leadingEdgeTime(const VFATFrame& frame, size_t ch_id) {
    return frame.getData()[header_offset + 2 + num_words_per_channel * ch_id] & 0xffff;
  }
  /// get timing information for single trailing edge
  inline uint16_t trailingEdgeTime(const VFATFrame& frame, size_t ch_id) {
    return frame.getData()[header_offset + 3 + num_words_per_channel * ch_id] & 0xffff;
  }
  /// retrieve this channel marker
  inline uint8_t channelMarker(const VFATFrame& frame, size_t ch_id) {
    return frame.getData()[header_offset + 1 + num_words_per_channel * ch_id] & 0x1f;
  }

  /// retrieve the header status flags
  inline uint8_t statusMarker(const VFATFrame& frame) { return (frame.getData()[header_offset - 1] >> 8) & 0xf; }

  /// retrieve the HW identifier for this channel, in firmware >2.1
  inline uint16_t newChannelId(const VFATFrame& frame, size_t ch_id) {
    return frame.getData()[header_offset + 0 + num_words_per_channel * ch_id] & 0xffff;
  }
}  // namespace totem::nt2::vfat

#endif

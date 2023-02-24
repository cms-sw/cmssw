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
  /// get timing information for single leading edge
  inline uint16_t leadingEdgeTime(const VFATFrame& frame, size_t ch_id) {
    return frame.getData()[2 + num_words_per_channel * ch_id] & 0xffff;
  }
  /// get timing information for single trailing edge
  inline uint16_t trailingEdgeTime(const VFATFrame& frame, size_t ch_id) {
    return frame.getData()[3 + num_words_per_channel * ch_id] & 0xffff;
  }
  /// retrieve this channel marker
  inline uint8_t channelMarker(const VFATFrame& frame, size_t ch_id) {
    return frame.getData()[1 + num_words_per_channel * ch_id] & 0x1f;
  }
  /// retrieve the GEO information for this channel
  inline uint8_t geoId(const VFATFrame& frame, size_t ch_id) {
    return frame.getData()[0 + num_words_per_channel * ch_id] & 0xff;
  }
  /// retrieve this channel identifier
  inline uint8_t channelId(const VFATFrame& frame, size_t ch_id) {
    return (frame.getData()[0 + num_words_per_channel * ch_id] >> 8) & 0xff;
  }
}  // namespace totem::nt2::vfat

#endif

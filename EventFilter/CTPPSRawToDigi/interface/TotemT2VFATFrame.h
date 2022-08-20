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
  /// get timing information for single leading edge
  inline uint16_t leadingEdgeTime(const VFATFrame& frame) { return frame.getData()[2] & 0xffff; }
  /// get timing information for single trailing edge
  inline uint16_t trailingEdgeTime(const VFATFrame& frame) { return frame.getData()[3] & 0xffff; }
  /// retrieve this channel marker
  inline uint8_t channelMarker(const VFATFrame& frame) { return frame.getData()[1] & 0x1f; }
  /// retrieve the GEO information for this channel
  inline uint8_t geoId(const VFATFrame& frame) { return frame.getData()[0] & 0x3f; }
  /// retrieve this channel identifier
  inline uint8_t channelId(const VFATFrame& frame) { return (frame.getData()[0] >> 8) & 0x3f; }
}  // namespace totem::nt2::vfat

#endif

/****************************************************************************
*
* This is a part of the PPS offline software.
* Authors:
*   Seyed Mohsen Etesami (setesami@cern.ch)
*   Laurent Forthomme
*   Nicola Minafra
*
****************************************************************************/

#ifndef EventFilter_CTPPSRawToDigi_DiamondVFATFrame
#define EventFilter_CTPPSRawToDigi_DiamondVFATFrame

#include "EventFilter/CTPPSRawToDigi/interface/VFATFrame.h"

#include <cstdint>

/**
 * Utilitary namespace to retrieve timing/status information from diamond VFAT frame
**/
namespace pps::diamond::vfat {
  /// Account for MSB/LSB "HW feature" reversal in HPTDC interpolation bits
  inline uint32_t correctTime(const uint32_t& time) { return (time & 0xFFE7FFFF) << 2 | (time & 0x00180000) >> 19; }

  /// get timing information for single leading edge
  inline uint32_t getLeadingEdgeTime(const VFATFrame& frame) {
    uint32_t time = ((frame.getData()[7] & 0x1f) << 16) + frame.getData()[8];
    return correctTime(time);
  }
  /// get timing information for single trailing edge
  inline uint32_t getTrailingEdgeTime(const VFATFrame& frame) {
    uint32_t time = ((frame.getData()[5] & 0x1f) << 16) + frame.getData()[6];
    return correctTime(time);
  }
  /// retrieve the threshold voltage for this channel
  inline uint32_t getThresholdVoltage(const VFATFrame& frame) {
    return ((frame.getData()[3] & 0x7ff) << 16) + frame.getData()[4];
  }
  /// flag stating whether the HPTDC channel encountered multiple hits
  inline VFATFrame::word getMultihit(const VFATFrame& frame) { return frame.getData()[2] & 0x01; }
  /// retrieve the list of error/status flags for the HPTDC when the frame was recorded
  inline VFATFrame::word getHptdcErrorFlag(const VFATFrame& frame) { return frame.getData()[1] & 0xFFFF; }
}  // namespace pps::diamond::vfat

#endif

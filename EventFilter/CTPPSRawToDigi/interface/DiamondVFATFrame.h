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

#include <array>
#include <cstdint>

/**
 * This class intended to handle the timing infromation of diamond VFAT frame
**/
class DiamondVFATFrame {
public:
  DiamondVFATFrame(const VFATFrame::word* inputData = nullptr) {
    std::copy_n(inputData, 9, data_.begin());
  }
  ~DiamondVFATFrame() = default;

  /// get timing information for leading single edge
  uint32_t getLeadingEdgeTime() const {
    uint32_t time = ((data_[7] & 0x1f) << 16) + data_[8];
    return correctTime(time);
  }
  /// get timing information for trailing single edge
  uint32_t getTrailingEdgeTime() const {
    uint32_t time = ((data_[5] & 0x1f) << 16) + data_[6];
    return correctTime(time);
  }
  /// retrieve the threshold voltage for this channel
  uint32_t getThresholdVoltage() const { return ((data_[3] & 0x7ff) << 16) + data_[4]; }
  /// flag stating whether the HPTDC channel encountered multiple hits
  VFATFrame::word getMultihit() const { return data_[2] & 0x01; }
  /// retrieve the list of error/status flags for the HPTDC when the frame was recorded
  VFATFrame::word getHptdcErrorFlag() const { return data_[1] & 0xFFFF; }

private:
  /// Account for MSB/LSB "HW feature" reversal in HPTDC interpolation bits
  uint32_t correctTime(uint32_t& time) const {
    return (time & 0xFFE7FFFF) << 2 | (time & 0x00180000) >> 19;
  }
  std::array<VFATFrame::word,9> data_;
};

#endif

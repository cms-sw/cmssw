#ifndef CTPPSDigi_TotemTimingDigi_h
#define CTPPSDigi_TotemTimingDigi_h

/** \class TotemTimingDigi
 *
 * Digi Class for CTPPS Timing Detector
 *
 * \author Mirko Berretti
 * \author Nicola Minafra
 * \author Laurent Forthomme
 * March 2018
 */

#include <cstdint>
#include <vector>

#include <DataFormats/CTPPSDigi/interface/TotemTimingEventInfo.h>

class TotemTimingDigi {
public:
  TotemTimingDigi(const uint8_t hwId,
                  const uint64_t fpgaTimestamp,
                  const uint16_t timestampA,
                  const uint16_t timestampB,
                  const uint16_t cellInfo,
                  const std::vector<uint8_t>& samples,
                  const TotemTimingEventInfo& totemTimingEventInfo);
  TotemTimingDigi(const TotemTimingDigi& digi);
  TotemTimingDigi();
  ~TotemTimingDigi(){};

  /// Digis are equal if they have all the same values, NOT checking the samples!
  bool operator==(const TotemTimingDigi& digi) const;

  /// Return digi values number

  /// Hardware Id formatted as: bits 0-3 Channel Id, bit 4 Sampic Id, bits 5-7 Digitizer Board Id
  inline unsigned int hardwareId() const { return hwId_; }

  inline unsigned int hardwareBoardId() const { return (hwId_ & 0xE0) >> 5; }

  inline unsigned int hardwareSampicId() const { return (hwId_ & 0x10) >> 4; }

  inline unsigned int hardwareChannelId() const { return (hwId_ & 0x0F); }

  inline unsigned int fpgaTimestamp() const { return fpgaTimestamp_; }

  inline unsigned int timestampA() const { return timestampA_; }

  inline unsigned int timestampB() const { return timestampB_; }

  inline unsigned int cellInfo() const { return cellInfo_; }

  inline std::vector<uint8_t> samples() const { return samples_; }

  inline std::vector<uint8_t>::const_iterator samplesBegin() const { return samples_.cbegin(); }

  inline std::vector<uint8_t>::const_iterator samplesEnd() const { return samples_.cend(); }

  inline unsigned int numberOfSamples() const { return samples_.size(); }

  inline int sampleAt(const unsigned int i) const {
    int sampleValue = -1;
    if (i < samples_.size())
      sampleValue = (int)samples_.at(i);
    return sampleValue;
  }

  inline TotemTimingEventInfo eventInfo() const { return totemTimingEventInfo_; }

  /// Set digi values
  /// Hardware Id formatted as: bits 0-3 Channel Id, bit 4 Sampic Id, bits 5-7 Digitizer Board Id
  inline void setHardwareId(const uint8_t hwId) { hwId_ = hwId; }

  inline void setHardwareBoardId(const unsigned int boardId) {
    hwId_ &= 0x1F;  // set board bits to 0
    hwId_ |= ((boardId & 0x07) << 5) & 0xE0;
  }

  inline void setHardwareSampicId(const unsigned int sampicId) {
    hwId_ &= 0xEF;  // set Sampic bit to 0
    hwId_ |= ((sampicId & 0x01) << 4) & 0x10;
  }

  inline void setHardwareChannelId(const unsigned int channelId) {
    hwId_ &= 0xF0;  // Set Sampic bit to 0
    hwId_ |= (channelId & 0x0F) & 0x0F;
  }

  inline void setFPGATimestamp(const uint64_t fpgaTimestamp) { fpgaTimestamp_ = fpgaTimestamp; }

  inline void setTimestampA(const uint16_t timestampA) { timestampA_ = timestampA; }

  inline void setTimestampB(const uint16_t timestampB) { timestampB_ = timestampB; }

  inline void setCellInfo(const uint16_t cellInfo) { cellInfo_ = cellInfo & 0x3F; }

  inline void setSamples(const std::vector<uint8_t>& samples) { samples_ = samples; }

  inline void addSample(const uint8_t sampleValue) { samples_.emplace_back(sampleValue); }

  inline void setSampleAt(const unsigned int i, const uint8_t sampleValue) {
    if (i < samples_.size())
      samples_.at(i) = sampleValue;
  }

  inline void setEventInfo(const TotemTimingEventInfo& totemTimingEventInfo) {
    totemTimingEventInfo_ = totemTimingEventInfo;
  }

private:
  uint8_t hwId_;
  uint64_t fpgaTimestamp_;
  uint16_t timestampA_;
  uint16_t timestampB_;
  uint16_t cellInfo_;

  std::vector<uint8_t> samples_;

  TotemTimingEventInfo totemTimingEventInfo_;
};

#include <iostream>

inline bool operator<(const TotemTimingDigi& one, const TotemTimingDigi& other) {
  if (one.eventInfo() < other.eventInfo())
    return true;
  if (one.hardwareId() < other.hardwareId())
    return true;
  return false;
}

inline std::ostream& operator<<(std::ostream& os, const TotemTimingDigi& digi) {
  return os << "TotemTimingDigi:"
            << "\nHardwareId:\t" << std::hex << digi.hardwareId() << "\nDB: " << std::dec << digi.hardwareBoardId()
            << "\tSampic: " << digi.hardwareSampicId() << "\tChannel: " << digi.hardwareChannelId()
            << "\nFPGATimestamp:\t" << std::dec << digi.fpgaTimestamp() << "\nTimestampA:\t" << std::dec
            << digi.timestampA() << "\nTimestampB:\t" << std::dec << digi.timestampB() << "\nCellInfo:\t" << std::hex
            << digi.cellInfo() << "\nNumberOfSamples:\t" << std::dec << digi.numberOfSamples() << std::endl
            << digi.eventInfo() << std::endl;
}

#endif

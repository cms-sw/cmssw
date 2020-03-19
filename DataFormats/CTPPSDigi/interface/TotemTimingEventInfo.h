#ifndef CTPPSDigi_TotemTimingEventInfo_h
#define CTPPSDigi_TotemTimingEventInfo_h

/** \class TotemTimingEventInfo
 *
 * Event Info Class for CTPPS Timing Detector
 *
 * \author Mirko Berretti
 * \author Nicola Minafra
 * \author Laurent Forthomme
 * \date March 2018
 */

#include <cstdint>
#include <bitset>

class TotemTimingEventInfo {
public:
  TotemTimingEventInfo(const uint8_t hwId,
                       const uint64_t l1ATimestamp,
                       const uint16_t bunchNumber,
                       const uint32_t orbitNumber,
                       const uint32_t eventNumber,
                       const uint16_t channelMap,
                       const uint16_t l1ALatency,
                       const uint8_t numberOfSamples,
                       const uint8_t offsetOfSamples,
                       const uint8_t pllInfo);
  TotemTimingEventInfo(const TotemTimingEventInfo& eventInfo);
  TotemTimingEventInfo();
  ~TotemTimingEventInfo(){};

  /// Digis are equal if they have all the same values, NOT checking the samples!
  bool operator==(const TotemTimingEventInfo& eventInfo) const;

  /// Return digi values number

  /// Hardware Id formatted as: bits 0-3 Channel Id, bit 4 Sampic Id, bits 5-7 Digitizer Board Id
  inline unsigned int hardwareId() const { return hwId_; }

  inline unsigned int hardwareBoardId() const { return (hwId_ & 0xE0) >> 5; }

  inline unsigned int hardwareSampicId() const { return (hwId_ & 0x10) >> 4; }

  inline unsigned int hardwareChannelId() const { return (hwId_ & 0x0F); }

  inline unsigned int l1ATimestamp() const { return l1ATimestamp_; }

  inline unsigned int bunchNumber() const { return bunchNumber_; }

  inline unsigned int orbitNumber() const { return orbitNumber_; }

  inline unsigned int eventNumber() const { return eventNumber_; }

  inline uint16_t channelMap() const { return channelMap_; }

  inline unsigned int l1ALatency() const { return l1ALatency_; }

  inline unsigned int numberOfSamples() const { return numberOfSamples_; }

  inline unsigned int offsetOfSamples() const { return offsetOfSamples_; }

  inline uint8_t pllInfo() const { return pllInfo_; }

  /// Set digi values
  /// Hardware Id formatted as: bits 0-3 Channel Id, bit 4 Sampic Id, bits 5-7 Digitizer Board Id
  inline void setHardwareId(const uint8_t hwId) { hwId_ = hwId; }

  inline void setHardwareBoardId(const unsigned int boardId) {
    hwId_ &= 0x1F;  // Set board bits to 0
    hwId_ |= ((boardId & 0x07) << 5) & 0xE0;
  }

  inline void setHardwareSampicId(const unsigned int sampicId) {
    hwId_ &= 0xEF;  // set Sampic bit to 0
    hwId_ |= ((sampicId & 0x01) << 4) & 0x10;
  }

  inline void setHardwareChannelId(const unsigned int channelId) {
    hwId_ &= 0xF0;  // set Sampic bit to 0
    hwId_ |= (channelId & 0x0F) & 0x0F;
  }

  inline void setL1ATimestamp(const uint64_t l1ATimestamp) { l1ATimestamp_ = l1ATimestamp; }

  inline void setBunchNumber(const uint16_t bunchNumber) { bunchNumber_ = bunchNumber; }

  inline void setOrbitNumber(const uint32_t orbitNumber) { orbitNumber_ = orbitNumber; }

  inline void setEventNumber(const uint32_t eventNumber) { eventNumber_ = eventNumber; }

  inline void setChannelMap(const uint16_t channelMap) { channelMap_ = channelMap; }

  inline void setL1ALatency(const uint16_t l1ALatency) { l1ALatency_ = l1ALatency; }

  inline void setNumberOfSamples(const uint8_t numberOfSamples) { numberOfSamples_ = numberOfSamples; }

  inline void setOffsetOfSamples(const uint8_t offsetOfSamples) { offsetOfSamples_ = offsetOfSamples; }

  inline void setPLLInfo(const uint8_t pllInfo) { pllInfo_ = pllInfo; }

private:
  uint8_t hwId_;
  uint64_t l1ATimestamp_;
  uint16_t bunchNumber_;
  uint32_t orbitNumber_;
  uint32_t eventNumber_;
  uint16_t channelMap_;
  uint16_t l1ALatency_;
  uint8_t numberOfSamples_;
  uint8_t offsetOfSamples_;
  uint8_t pllInfo_;
};

#include <iostream>

inline bool operator<(const TotemTimingEventInfo& one, const TotemTimingEventInfo& other) {
  if (one.eventNumber() < other.eventNumber())
    return true;
  if (one.l1ATimestamp() < other.l1ATimestamp())
    return true;
  if (one.hardwareId() < other.hardwareId())
    return true;
  return false;
}

inline std::ostream& operator<<(std::ostream& o, const TotemTimingEventInfo& digi) {
  std::bitset<16> bitsPLLInfo(digi.pllInfo());
  return o << "TotemTimingEventInfo:"
           << "\nHardwareId:\t" << std::hex << digi.hardwareId() << "\nDB: " << std::dec << digi.hardwareBoardId()
           << "\tSampic: " << digi.hardwareSampicId() << "\tChannel: " << digi.hardwareChannelId()
           << "\nL1A Timestamp:\t" << std::dec << digi.l1ATimestamp() << "\nL1A Latency:\t" << std::dec
           << digi.l1ALatency() << "\nBunch Number:\t" << std::dec << digi.bunchNumber() << "\nOrbit Number:\t"
           << std::dec << digi.orbitNumber() << "\nEvent Number:\t" << std::dec << digi.eventNumber()
           << "\nChannels fired:\t" << std::hex << digi.channelMap() << "\nNumber of Samples:\t" << std::dec
           << digi.numberOfSamples() << "\nOffset of Samples:\t" << std::dec << digi.offsetOfSamples()
           << "\nPLL Info:\t" << bitsPLLInfo.to_string() << std::endl;
}

#endif

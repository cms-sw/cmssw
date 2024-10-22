#ifndef EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDZSChannelUnpacker_H  // {
#define EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDZSChannelUnpacker_H

#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQHeader.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQTrailer.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDChannel.h"
#include <cstdint>

namespace Phase2Tracker {

  class Phase2TrackerFEDZSChannelUnpacker {
  public:
    Phase2TrackerFEDZSChannelUnpacker(const Phase2TrackerFEDChannel& channel);
    uint8_t clusterIndex() const { return data_[currentOffset_ ^ 7]; }
    uint8_t clusterLength() const { return data_[(currentOffset_ + 1) ^ 7]; }
    bool hasData() const { return valuesLeft_; }
    Phase2TrackerFEDZSChannelUnpacker& operator++();
    Phase2TrackerFEDZSChannelUnpacker& operator++(int);

  private:
    const uint8_t* data_;
    uint8_t currentOffset_;
    uint16_t valuesLeft_;
  };

  // unpacker for ZS CBC data
  inline Phase2TrackerFEDZSChannelUnpacker::Phase2TrackerFEDZSChannelUnpacker(const Phase2TrackerFEDChannel& channel)
      : data_(channel.data()), currentOffset_(channel.offset()), valuesLeft_(channel.length() / 2) {}

  inline Phase2TrackerFEDZSChannelUnpacker& Phase2TrackerFEDZSChannelUnpacker::operator++() {
    currentOffset_ = currentOffset_ + 2;
    valuesLeft_--;
    return (*this);
  }

  inline Phase2TrackerFEDZSChannelUnpacker& Phase2TrackerFEDZSChannelUnpacker::operator++(int) {
    ++(*this);
    return *this;
  }

}  // namespace Phase2Tracker

#endif  // } end def EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDZSChannelUnpacker_H

#ifndef EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDRawChannelUnpacker_H  // {
#define EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDRawChannelUnpacker_H

#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQHeader.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQTrailer.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDChannel.h"
#include <cstdint>

namespace Phase2Tracker {

  // unpacker for RAW CBC data
  // each bit of the channel is related to one strip
  class Phase2TrackerFEDRawChannelUnpacker {
  public:
    Phase2TrackerFEDRawChannelUnpacker(const Phase2TrackerFEDChannel& channel);
    uint8_t stripIndex() const { return currentStrip_; }
    bool stripOn() const { return bool((currentWord_ >> bitInWord_) & 0x1); }
    bool hasData() const { return valuesLeft_; }
    Phase2TrackerFEDRawChannelUnpacker& operator++();
    Phase2TrackerFEDRawChannelUnpacker& operator++(int);

  private:
    const uint8_t* data_;
    uint8_t currentOffset_;
    uint8_t currentStrip_;
    uint16_t valuesLeft_;
    uint8_t currentWord_;
    uint8_t bitInWord_;
  };  // end of Phase2TrackerFEDRawChannelUnpacker

  inline Phase2TrackerFEDRawChannelUnpacker::Phase2TrackerFEDRawChannelUnpacker(const Phase2TrackerFEDChannel& channel)
      : data_(channel.data()),
        currentOffset_(channel.offset()),
        currentStrip_(0),
        valuesLeft_((channel.length()) * 8 - STRIPS_PADDING),
        currentWord_(channel.data()[currentOffset_ ^ 7]),
        bitInWord_(0) {}

  inline Phase2TrackerFEDRawChannelUnpacker& Phase2TrackerFEDRawChannelUnpacker::operator++() {
    bitInWord_++;
    currentStrip_++;
    if (bitInWord_ > 7) {
      bitInWord_ = 0;
      currentOffset_++;
      currentWord_ = data_[currentOffset_ ^ 7];
    }
    valuesLeft_--;
    return (*this);
  }

  inline Phase2TrackerFEDRawChannelUnpacker& Phase2TrackerFEDRawChannelUnpacker::operator++(int) {
    ++(*this);
    return *this;
  }

}  // namespace Phase2Tracker

#endif  // } end def EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDRawChannelUnpacker_H

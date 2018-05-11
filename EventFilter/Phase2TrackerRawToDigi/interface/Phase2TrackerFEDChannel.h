#ifndef EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDChannel_H // {
#define EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDChannel_H

#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQHeader.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQTrailer.h"
#include <cstdint>

namespace Phase2Tracker {

  // holds information about position of a channel in the buffer
  // for use by unpacker
  class Phase2TrackerFEDChannel
  {
    //forward declaration to avoid circular includes
    class Phase2TrackerFEDBuffer;
    public:
      Phase2TrackerFEDChannel(const uint8_t*const data, const size_t offset,
                 const uint16_t length): data_(data), offset_(offset), length_(length) {}

      //gets length from first 2 bytes (assuming normal FED channel)
      Phase2TrackerFEDChannel(const uint8_t*const data, const size_t offset);
      uint16_t length() const { return length_; }
      const uint8_t* data() const { return data_; }
      size_t offset() const { return offset_; }
      uint16_t cmMedian(const uint8_t apvIndex) const;
    private:
      friend class Phase2TrackerFEDBuffer;
      //third byte of channel data for normal FED channels
      uint8_t packetCode() const;
      const uint8_t* data_;
      size_t offset_;
      uint16_t length_;
  }; // end Phase2TrackerFEDChannel class

} // end of Phase2Tracker namespace

#endif // } end def EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDChannel_H


#ifndef EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDChannel_H // {
#define EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDChannel_H

#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQHeader.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQTrailer.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDBuffer.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/utils.h"
#include <stdint.h>

namespace Phase2Tracker {

  // holds information about position of a channel in the buffer
  // for use by unpacker
  class Phase2TrackerFEDChannel
  {
    public:
      Phase2TrackerFEDChannel(const uint8_t*const data, const size_t offset,
                 const uint16_t length, const uint8_t bitoffset = 0, const DET_TYPE dettype = UNUSED): data_(data), offset_(offset), length_(length), bitoffset_(bitoffset), dettype_(dettype) {}

      //gets length from first 2 bytes (assuming normal FED channel)
      Phase2TrackerFEDChannel(const uint8_t*const data, const size_t offset);
      uint16_t length() const { return length_; }
      const uint8_t* data() const { return data_; }
      size_t offset() const { return offset_; }
      uint16_t bitoffset() const { return bitoffset_; }
      DET_TYPE dettype() const { return dettype_; }
    private:
      friend class Phase2TrackerFEDBuffer;
      //third byte of channel data for normal FED channels
      uint8_t packetCode() const;
      const uint8_t* data_;
      size_t offset_;
      uint16_t length_;
      uint16_t bitoffset_;
      DET_TYPE dettype_;
  }; // end Phase2TrackerFEDChannel class

} // end of Phase2Tracker namespace

#endif // } end def EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDChannel_H


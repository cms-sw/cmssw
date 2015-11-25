#ifndef EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDBuffer_H // {
#define EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDBuffer_H

#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQHeader.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQTrailer.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDHeader.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDChannel.h"
#include <stdint.h>
#include <vector>
#include <map>

namespace Phase2Tracker {

  class Phase2TrackerFEDBuffer
  { 
    public:
      // gets data of one tracker FED to check, analyze and sort it
      Phase2TrackerFEDBuffer(const uint8_t* fedBuffer, const size_t fedBufferSize);
      ~Phase2TrackerFEDBuffer();

      //dump buffer to stream
      void dump(std::ostream& os) const { printHex(buffer_,bufferSize_,os); }

      //methods to get parts of the buffer
      FEDDAQHeader daqHeader() const { return daqHeader_; }
      FEDDAQTrailer daqTrailer() const { return daqTrailer_; }
      size_t bufferSize() const { return bufferSize_; }
      Phase2TrackerFEDHeader trackerHeader() const { return trackerHeader_; }
      const Phase2TrackerFEDChannel& channel(const uint32_t internalPhase2TrackerFEDChannelNum) const { return channels_[internalPhase2TrackerFEDChannelNum]; }
      std::map<uint32_t,uint32_t> conditionData();
      int isValid() { return valid_; }

      //methods to get info from DAQ header from FEDDAQHeader class
      FEDDAQEventType daqEventType() const { return daqHeader_.eventType(); }
      uint32_t daqLvl1ID() const { return daqHeader_.l1ID(); }
      uint16_t daqBXID() const { return daqHeader_.bxID(); }
      uint16_t daqSourceID() const { return daqHeader_.sourceID(); }

      //methods to get info from DAQ trailer from FEDDAQTrailer class
      uint32_t daqEventLengthIn64bitWords() const { return daqTrailer_.eventLengthIn64BitWords(); }
      uint32_t daqEventLengthInBytes() const { return daqTrailer_.eventLengthInBytes(); }
      uint16_t daqCRC() const { return daqTrailer_.crc(); }
      FEDTTSBits daqTTSState() const { return daqTrailer_.ttsBits(); }

      //methods to get info from the tracker header using Phase2TrackerFEDHeader class
      FEDReadoutMode readoutMode() const;
      inline const uint8_t* getPointerToPayload()  const { return trackerHeader_.getPointerToData(); }
      inline const uint8_t* getPointerToCondData() const { return condDataPointer_; }
      inline const uint16_t* getPointerToTriggerData() const { return triggerPointer_; }

    private:
      const uint8_t* buffer_;
      const size_t bufferSize_;
      std::vector<Phase2TrackerFEDChannel> channels_;
      FEDDAQHeader daqHeader_;
      FEDDAQTrailer daqTrailer_;
      Phase2TrackerFEDHeader trackerHeader_;
      const uint8_t* payloadPointer_;
      const uint8_t* condDataPointer_;
      const uint16_t* triggerPointer_;
      void findChannels();
      int valid_;

    //////////////// Deprecated or dummy implemented methods ///////////////////
    public:
      // check methods
      inline bool doChecks() const { return true; }  // FEDBuffer
      inline bool checkNoFEOverflows() const { return true; } // FEDBufferBase
      inline bool doCorruptBufferChecks() const { return true; } // FEDBuffer

  }; // end of FEDBuffer class

} // end of Phase2Tracker namespace

#endif // } end def EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDBuffer_H


#ifndef EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDHeader_H // {
#define EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDHeader_H

#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQHeader.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQTrailer.h"
#include <stdint.h>
#include <vector>

namespace Phase2Tracker {

  // tracker headers for new CBC system
  class Phase2TrackerFEDHeader
  {
    public:
      Phase2TrackerFEDHeader() { memset(headercopy_,0x00,16); valid_ = 1; }
      explicit Phase2TrackerFEDHeader(const uint8_t* headerPointer);

      // getters:
      inline uint8_t getDataFormatVersion() const { return dataFormatVersion_; }
      void setDataFormatVersion(uint8_t);

      inline READ_MODE getDebugMode() const { return debugMode_; }
      void setDebugMode(READ_MODE);

      inline uint8_t getEventType() const { return eventType_; }
      void setEventType(uint8_t);

      inline FEDReadoutMode getReadoutMode() const { return readoutMode_; }
      inline uint8_t getConditionData() const { return conditionData_; }
      inline uint8_t getDataType() const { return dataType_; }

      inline uint64_t getGlibStatusCode() const { return glibStatusCode_; }
      void setGlibStatusCode(uint64_t);
 
      inline uint16_t getNumberOfCBC() const { return numberOfCBC_; }
      void setNumberOfCBC(uint16_t);

      // get pointer to Payload data after tracker head
      const uint8_t* getPointerToData() const { return pointerToData_;}

      // get Front-End Status (up to 72 bits)
      std::vector<bool> frontendStatus() const;
      void setFrontendStatus(std::vector<bool>);

      inline uint8_t getTrackerHeaderSize() const { return trackerHeaderSize_; }
      // CBC status bits, according to debug mode 
      // (empty, 1bit per CBC, 8bits per CBC)
      std::vector<uint16_t> CBCStatus() const;
      void setCBCStatus();

      // get header raw data
      inline uint8_t* data() { return headercopy_; }

      // header validity
      int isValid() { return valid_; }

    private:
      // readers: read info from Tracker Header and store in local variables

      // version number (4 bits)
      uint8_t dataFormatVersion();
      // debug level (2 bits) :
      // 01 = full debug, 10 = CBC error mode, 00 = summary mode
      READ_MODE debugMode();
      // event type (4 bits):
      // RAW/ZS, condition data, data type (real or simulated)
      uint8_t eventType() const;
      // get readout mode (first bit of the above)
      FEDReadoutMode readoutMode();
      uint8_t conditionData() const;
      uint8_t dataType() const;
      // glib status registers code (38 bits)
      uint64_t glibStatusCode() const;
      // number of CBC chips (8 bits)
      uint16_t numberOfCBC() const;
      // get tracker size (see function) and pointer to end of header. Also sets the TrackerHeaderSize.
      const uint8_t* pointerToData();
      uint8_t headercopy_[16];

    private:
      void init();
      const uint8_t* trackerHeader_; // pointer to the begining of Tracker Header
      int valid_;                    // header validity
      const uint8_t* pointerToData_; // pointer next to end of Tracker Header
      uint8_t trackerHeaderSize_;    // Tracker Header in bytes
      uint8_t  dataFormatVersion_; // shoud be 1
      READ_MODE  debugMode_;       // debug, error, sumary ...
      uint8_t  eventType_;         // contains readoutMode_, conditionData_ and dataType_
      FEDReadoutMode readoutMode_; // proc raw or zero suppress
      uint8_t conditionData_;      // condition data present or not
      uint8_t dataType_;           // data fake or real
      uint64_t glibStatusCode_;    // glib status registers
      uint16_t numberOfCBC_;       // Total number of connected CBC
      uint64_t header_first_word_;

  }; // end of Phase2TrackerFEDHeader class
}
#endif // } end def EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDHeader_H


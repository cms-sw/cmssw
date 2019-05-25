#ifndef EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDHeader_H  // {
#define EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDHeader_H

#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQHeader.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQTrailer.h"
#include <cstdint>
#include <vector>

namespace Phase2Tracker {

  // tracker headers for new CBC system
  class Phase2TrackerFEDHeader {
  public:
    Phase2TrackerFEDHeader() {}

    explicit Phase2TrackerFEDHeader(const uint8_t* headerPointer);

    // getters:
    inline uint8_t getDataFormatVersion() const { return dataFormatVersion_; }
    inline READ_MODE getDebugMode() const { return debugMode_; }

    inline uint8_t getEventType() const { return eventType_; }
    inline FEDReadoutMode getReadoutMode() const { return readoutMode_; }
    inline uint8_t getConditionData() const { return conditionData_; }
    inline uint8_t getDataType() const { return dataType_; }

    inline uint64_t getGlibStatusCode() const { return glibStatusCode_; }
    inline uint16_t getNumberOfCBC() const { return numberOfCBC_; }

    // get pointer to Payload data after tracker head
    const uint8_t* getPointerToData() const { return pointerToData_; }
    // get Front-End Status (16 bits) ==> 16 bool
    std::vector<bool> frontendStatus() const;

    inline uint8_t getTrackerHeaderSize() const { return trackerHeaderSize_; }

    // CBC status bits, according to debug mode
    // (empty, 1bit per CBC, 8bits per CBC)
    std::vector<uint8_t> CBCStatus() const;

  private:
    // readers: read info from Tracker Header and store in local variables

    // version number (4 bits)
    uint8_t dataFormatVersion() const;
    // debug level (2 bits) :
    // 01 = full debug, 10 = CBC error mode, 00 = summary mode
    READ_MODE debugMode() const;
    // event type (4 bits):
    // RAW/ZS, condition data, data type (real or simulated)
    uint8_t eventType() const;
    // get readout mode (first bit of the above)
    FEDReadoutMode readoutMode() const;
    uint8_t conditionData() const;
    uint8_t dataType() const;
    // glib status registers code (38 bits)
    uint64_t glibStatusCode() const;
    // number of CBC chips (8 bits)
    uint16_t numberOfCBC() const;
    // get tracker size (see function) and pointer to end of header. Also sets the TrackerHeaderSize.
    const uint8_t* pointerToData();

  private:
    void init();
    const uint8_t* trackerHeader_;  // pointer to the begining of Tracker Header
    const uint8_t* pointerToData_;  // pointer next to end of Tracker Header
    uint8_t trackerHeaderSize_;     // Tracker Header in bytes
    uint64_t header_first_word_;
    uint64_t header_second_word_;
    uint8_t dataFormatVersion_;   // shoud be 1
    READ_MODE debugMode_;         // debug, error, sumary ...
    uint8_t eventType_;           // contains readoutMode_, conditionData_ and dataType_
    FEDReadoutMode readoutMode_;  // proc raw or zero suppress
    uint8_t conditionData_;       // condition data present or not
    uint8_t dataType_;            // data fake or real
    uint64_t glibStatusCode_;     // glib status registers
    uint16_t numberOfCBC_;        // Total number of connected CBC

  };  // end of Phase2TrackerFEDHeader class
}  // namespace Phase2Tracker
#endif  // } end def EventFilter_Phase2TrackerRawToDigi_Phase2TrackerPhase2TrackerFEDHeader_H

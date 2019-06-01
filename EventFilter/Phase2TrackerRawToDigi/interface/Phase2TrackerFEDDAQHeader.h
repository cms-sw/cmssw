#ifndef EventFilter_Phase2TrackerRawToDigi_Phase2TrackerFEDDAQHeader_H  // {
#define EventFilter_Phase2TrackerRawToDigi_Phase2TrackerFEDDAQHeader_H

#include <cstring>
#include "EventFilter/Phase2TrackerRawToDigi/interface/utils.h"

namespace Phase2Tracker {

  //
  // Constants
  //

  //enum values are values which appear in buffer. DO NOT CHANGE!
  //see http://cmsdoc.cern.ch/cms/TRIDAS/horizontal/RUWG/DAQ_IF_guide/DAQ_IF_guide.html
  enum FEDDAQEventType {
    DAQ_EVENT_TYPE_PHYSICS = 0x1,
    DAQ_EVENT_TYPE_CALIBRATION = 0x2,
    DAQ_EVENT_TYPE_TEST = 0x3,
    DAQ_EVENT_TYPE_TECHNICAL = 0x4,
    DAQ_EVENT_TYPE_SIMULATED = 0x5,
    DAQ_EVENT_TYPE_TRACED = 0x6,
    DAQ_EVENT_TYPE_ERROR = 0xF,
    DAQ_EVENT_TYPE_INVALID = INVALID
  };

  //to make enums printable
  inline std::ostream& operator<<(std::ostream& os, const FEDDAQEventType& value);

  //see http://cmsdoc.cern.ch/cms/TRIDAS/horizontal/RUWG/DAQ_IF_guide/DAQ_IF_guide.html
  class FEDDAQHeader {
  public:
    FEDDAQHeader() {}
    explicit FEDDAQHeader(const uint8_t* header);

    // getters
    //0x5 in first fragment
    uint8_t boeNibble() const;
    uint8_t eventTypeNibble() const;
    FEDDAQEventType eventType() const;
    uint32_t l1ID() const;
    uint16_t bxID() const;
    uint16_t sourceID() const;
    uint8_t version() const;
    //0 if current header word is last, 1 otherwise
    bool hBit() const;
    bool lastHeader() const;
    void print(std::ostream& os) const;

    //used by digi2Raw
    const uint8_t* data() const;

    // setters
    void setEventType(const FEDDAQEventType evtType);
    void setL1ID(const uint32_t l1ID);
    void setBXID(const uint16_t bxID);
    void setSourceID(const uint16_t sourceID);
    FEDDAQHeader(const uint32_t l1ID,
                 const uint16_t bxID,
                 const uint16_t sourceID,
                 const FEDDAQEventType evtType = DAQ_EVENT_TYPE_PHYSICS);

  private:
    uint8_t header_[8];
  };  // end of FEDDAQHeader class

  //FEDDAQHeader

  inline FEDDAQHeader::FEDDAQHeader(const uint8_t* header) { memcpy(header_, header, 8); }

  inline uint8_t FEDDAQHeader::boeNibble() const { return ((header_[7] & 0xF0) >> 4); }

  inline uint8_t FEDDAQHeader::eventTypeNibble() const { return (header_[7] & 0x0F); }

  inline uint32_t FEDDAQHeader::l1ID() const { return (header_[4] | (header_[5] << 8) | (header_[6] << 16)); }

  inline uint16_t FEDDAQHeader::bxID() const { return ((header_[3] << 4) | ((header_[2] & 0xF0) >> 4)); }

  inline uint16_t FEDDAQHeader::sourceID() const { return (((header_[2] & 0x0F) << 8) | header_[1]); }

  inline uint8_t FEDDAQHeader::version() const { return ((header_[0] & 0xF0) >> 4); }

  inline bool FEDDAQHeader::hBit() const { return (header_[0] & 0x8); }

  inline bool FEDDAQHeader::lastHeader() const { return !hBit(); }

  inline const uint8_t* FEDDAQHeader::data() const { return header_; }

  inline void FEDDAQHeader::print(std::ostream& os) const { printHex(header_, 8, os); }

}  // namespace Phase2Tracker

#endif  // } end def EventFilter_Phase2TrackerRawToDigi_Phase2TrackerFEDDAQHeader_H

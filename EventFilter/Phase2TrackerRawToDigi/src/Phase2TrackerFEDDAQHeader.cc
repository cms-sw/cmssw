#include <iomanip>
#include <ostream>
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQHeader.h"

namespace Phase2Tracker {

  std::ostream& operator<<(std::ostream& os, const FEDDAQEventType& value) {
    switch (value) {
      case DAQ_EVENT_TYPE_PHYSICS:
        os << "Physics trigger";
        break;
      case DAQ_EVENT_TYPE_CALIBRATION:
        os << "Calibration trigger";
        break;
      case DAQ_EVENT_TYPE_TEST:
        os << "Test trigger";
        break;
      case DAQ_EVENT_TYPE_TECHNICAL:
        os << "Technical trigger";
        break;
      case DAQ_EVENT_TYPE_SIMULATED:
        os << "Simulated event";
        break;
      case DAQ_EVENT_TYPE_TRACED:
        os << "Traced event";
        break;
      case DAQ_EVENT_TYPE_ERROR:
        os << "Error";
        break;
      case DAQ_EVENT_TYPE_INVALID:
        os << "Unknown";
        break;
      default:
        os << "Unrecognized";
        os << " (";
        printHexValue(value, os);
        os << ")";
        break;
    }
    return os;
  }

  FEDDAQEventType FEDDAQHeader::eventType() const {
    switch (eventTypeNibble()) {
      case DAQ_EVENT_TYPE_PHYSICS:
      case DAQ_EVENT_TYPE_CALIBRATION:
      case DAQ_EVENT_TYPE_TEST:
      case DAQ_EVENT_TYPE_TECHNICAL:
      case DAQ_EVENT_TYPE_SIMULATED:
      case DAQ_EVENT_TYPE_TRACED:
      case DAQ_EVENT_TYPE_ERROR:
        return FEDDAQEventType(eventTypeNibble());
      default:
        return DAQ_EVENT_TYPE_INVALID;
    }
  }

  void FEDDAQHeader::setEventType(const FEDDAQEventType evtType) { header_[7] = ((header_[7] & 0xF0) | evtType); }

  void FEDDAQHeader::setL1ID(const uint32_t l1ID) {
    header_[4] = (l1ID & 0x000000FF);
    header_[5] = ((l1ID & 0x0000FF00) >> 8);
    header_[6] = ((l1ID & 0x00FF0000) >> 16);
  }

  void FEDDAQHeader::setBXID(const uint16_t bxID) {
    header_[3] = ((bxID & 0x0FF0) >> 4);
    header_[2] = ((header_[2] & 0x0F) | ((bxID & 0x000F) << 4));
  }

  void FEDDAQHeader::setSourceID(const uint16_t sourceID) {
    header_[2] = ((header_[2] & 0xF0) | ((sourceID & 0x0F00) >> 8));
    header_[1] = (sourceID & 0x00FF);
  }

  FEDDAQHeader::FEDDAQHeader(const uint32_t l1ID,
                             const uint16_t bxID,
                             const uint16_t sourceID,
                             const FEDDAQEventType evtType) {
    //clear everything (FOV,H,x,$ all set to 0)
    memset(header_, 0x0, 8);
    //set the BoE nibble to indicate this is the last fragment
    header_[7] = 0x50;
    //set variable fields vith values supplied
    setEventType(evtType);
    setL1ID(l1ID);
    setBXID(bxID);
    setSourceID(sourceID);
  }

}  // namespace Phase2Tracker

#include <iomanip>
#include <ostream>
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDDAQTrailer.h"

namespace Phase2Tracker {

  std::ostream& operator<<(std::ostream& os, const FEDTTSBits& value) {
    switch (value) {
      case TTS_DISCONNECTED0:
        os << "Disconected 0";
        break;
      case TTS_WARN_OVERFLOW:
        os << "Warning overflow";
        break;
      case TTS_OUT_OF_SYNC:
        os << "Out of sync";
        break;
      case TTS_BUSY:
        os << "Busy";
        break;
      case TTS_READY:
        os << "Ready";
        break;
      case TTS_ERROR:
        os << "Error";
        break;
      case TTS_INVALID:
        os << "Invalid";
        break;
      case TTS_DISCONNECTED1:
        os << "Disconected 1";
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

  FEDTTSBits FEDDAQTrailer::ttsBits() const {
    switch (ttsNibble()) {
      case TTS_DISCONNECTED0:
      case TTS_WARN_OVERFLOW:
      case TTS_OUT_OF_SYNC:
      case TTS_BUSY:
      case TTS_READY:
      case TTS_ERROR:
      case TTS_DISCONNECTED1:
        return FEDTTSBits(ttsNibble());
      default:
        return TTS_INVALID;
    }
  }

  FEDDAQTrailer::FEDDAQTrailer(const uint32_t eventLengthIn64BitWords,
                               const uint16_t crc,
                               const FEDTTSBits ttsBits,
                               const bool slinkTransmissionError,
                               const bool badFEDID,
                               const bool slinkCRCError,
                               const uint8_t eventStatusNibble) {
    //clear everything (T,x,$ all set to 0)
    memset(trailer_, 0x0, 8);
    //set the EoE nibble to indicate this is the last fragment
    trailer_[7] = 0xA0;
    //set variable fields vith values supplied
    setEventLengthIn64BitWords(eventLengthIn64BitWords);
    setEventStatusNibble(eventStatusNibble);
    setTTSBits(ttsBits);
    setCRC(crc);
    setSLinkTransmissionErrorBit(slinkTransmissionError);
    setBadSourceIDBit(badFEDID);
    setSLinkCRCErrorBit(slinkCRCError);
  }

  void FEDDAQTrailer::setEventLengthIn64BitWords(const uint32_t eventLengthIn64BitWords) {
    trailer_[4] = (eventLengthIn64BitWords & 0x000000FF);
    trailer_[5] = ((eventLengthIn64BitWords & 0x0000FF00) >> 8);
    trailer_[6] = ((eventLengthIn64BitWords & 0x00FF0000) >> 16);
  }

  void FEDDAQTrailer::setCRC(const uint16_t crc) {
    trailer_[2] = (crc & 0x00FF);
    trailer_[3] = ((crc >> 8) & 0x00FF);
  }

  void FEDDAQTrailer::setSLinkTransmissionErrorBit(const bool bitSet) {
    if (bitSet)
      trailer_[1] |= 0x80;
    else
      trailer_[1] &= (~0x80);
  }

  void FEDDAQTrailer::setBadSourceIDBit(const bool bitSet) {
    if (bitSet)
      trailer_[1] |= 0x40;
    else
      trailer_[1] &= (~0x40);
  }

  void FEDDAQTrailer::setSLinkCRCErrorBit(const bool bitSet) {
    if (bitSet)
      trailer_[0] |= 0x04;
    else
      trailer_[0] &= (~0x40);
  }

  void FEDDAQTrailer::setEventStatusNibble(const uint8_t eventStatusNibble) {
    trailer_[1] = ((trailer_[1] & 0xF0) | (eventStatusNibble & 0x0F));
  }

  void FEDDAQTrailer::setTTSBits(const FEDTTSBits ttsBits) { trailer_[0] = ((trailer_[0] & 0x0F) | (ttsBits & 0xF0)); }

}  // namespace Phase2Tracker

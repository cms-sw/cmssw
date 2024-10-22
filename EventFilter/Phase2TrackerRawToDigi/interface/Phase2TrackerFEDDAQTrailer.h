#ifndef EventFilter_Phase2TrackerRawToDigi_Phase2TrackerFEDDAQTrailer_H  // {
#define EventFilter_Phase2TrackerRawToDigi_Phase2TrackerFEDDAQTrailer_H

#include <cstring>
#include "EventFilter/Phase2TrackerRawToDigi/interface/utils.h"

namespace Phase2Tracker {

  //
  // Constants
  //

  //enum values are values which appear in buffer. DO NOT CHANGE!
  //see http://cmsdoc.cern.ch/cms/TRIDAS/horizontal/RUWG/DAQ_IF_guide/DAQ_IF_guide.html
  enum FEDTTSBits {
    TTS_DISCONNECTED0 = 0x0,
    TTS_WARN_OVERFLOW = 0x1,
    TTS_OUT_OF_SYNC = 0x2,
    TTS_BUSY = 0x4,
    TTS_READY = 0x8,
    TTS_ERROR = 0x12,
    TTS_DISCONNECTED1 = 0xF,
    TTS_INVALID = INVALID
  };

  //to make enums printable
  inline std::ostream& operator<<(std::ostream& os, const FEDTTSBits& value);

  //see http://cmsdoc.cern.ch/cms/TRIDAS/horizontal/RUWG/DAQ_IF_guide/DAQ_IF_guide.html
  class FEDDAQTrailer {
  public:
    FEDDAQTrailer() {}
    explicit FEDDAQTrailer(const uint8_t* trailer);

    // getters
    //0xA in first fragment
    uint8_t eoeNibble() const;
    uint32_t eventLengthIn64BitWords() const;
    uint32_t eventLengthInBytes() const;
    uint16_t crc() const;
    //set to 1 if FRL detects a transmission error over S-link
    bool cBit() const;
    bool slinkTransmissionError() const { return cBit(); }
    //set to 1 if the FED ID is not the one expected by the FRL
    bool fBit() const;
    bool badSourceID() const { return fBit(); }
    uint8_t eventStatusNibble() const;
    uint8_t ttsNibble() const;
    FEDTTSBits ttsBits() const;
    //0 if the current trailer is the last, 1 otherwise
    bool tBit() const;
    bool lastTrailer() const { return !tBit(); }
    //set to 1 if the S-link sender card detects a CRC error
    // (the CRC it computes is put in the CRC field)
    bool rBit() const;
    bool slinkCRCError() const { return rBit(); }
    void print(std::ostream& os) const;
    //used by digi2Raw
    const uint8_t* data() const;

    // setters
    void setEventLengthIn64BitWords(const uint32_t eventLengthIn64BitWords);
    void setCRC(const uint16_t crc);
    void setSLinkTransmissionErrorBit(const bool bitSet);
    void setBadSourceIDBit(const bool bitSet);
    void setSLinkCRCErrorBit(const bool bitSet);
    void setEventStatusNibble(const uint8_t eventStatusNibble);
    void setTTSBits(const FEDTTSBits ttsBits);
    FEDDAQTrailer(const uint32_t eventLengthIn64BitWords,
                  const uint16_t crc = 0,
                  const FEDTTSBits ttsBits = TTS_READY,
                  const bool slinkTransmissionError = false,
                  const bool badFEDID = false,
                  const bool slinkCRCError = false,
                  const uint8_t eventStatusNibble = 0);

  private:
    uint8_t trailer_[8];

  };  // end of FEDDAQTrailer class

  //FEDDAQTrailer methods definintions {

  //FEDDAQTrailer

  inline FEDDAQTrailer::FEDDAQTrailer(const uint8_t* trailer) { memcpy(trailer_, trailer, 8); }

  inline uint8_t FEDDAQTrailer::eoeNibble() const { return ((trailer_[7] & 0xF0) >> 4); }

  inline uint32_t FEDDAQTrailer::eventLengthIn64BitWords() const {
    return (trailer_[4] | (trailer_[5] << 8) | (trailer_[6] << 16));
  }

  inline uint32_t FEDDAQTrailer::eventLengthInBytes() const { return eventLengthIn64BitWords() * 8; }

  inline uint16_t FEDDAQTrailer::crc() const { return (trailer_[2] | (trailer_[3] << 8)); }

  inline bool FEDDAQTrailer::cBit() const { return (trailer_[1] & 0x80); }

  inline bool FEDDAQTrailer::fBit() const { return (trailer_[1] & 0x40); }

  inline uint8_t FEDDAQTrailer::eventStatusNibble() const { return (trailer_[1] & 0x0F); }

  inline uint8_t FEDDAQTrailer::ttsNibble() const { return ((trailer_[0] & 0xF0) >> 4); }

  inline bool FEDDAQTrailer::tBit() const { return (trailer_[0] & 0x08); }

  inline bool FEDDAQTrailer::rBit() const { return (trailer_[0] & 0x04); }

  inline void FEDDAQTrailer::print(std::ostream& os) const { printHex(trailer_, 8, os); }

  inline const uint8_t* FEDDAQTrailer::data() const { return trailer_; }

  // End of method definitions }

}  // namespace Phase2Tracker

#endif  // } end def EventFilter_Phase2TrackerRawToDigi_Phase2TrackerFEDDAQHeader_H

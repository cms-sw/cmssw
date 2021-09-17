#ifndef IOPool_Streamer_FRDEventMessage_h
#define IOPool_Streamer_FRDEventMessage_h

/**
 * This class is used to view the contents of "FED Raw Data" (FRD) events.
 * Initially, the only time that such events are created is
 * when an error occurs in the HLT.
 * So, in the beginning, "error event" == "FRD event".
 * However, if new ways of creating events of this type are added
 * to the HLT, hopefully this class can be used or upgraded to handle those
 * events as well.
 *
 * 08-Aug-2008 - KAB  - Initial Implementation
 * 06-Oct-2008 - KAB  - Added version number and lumi block number (version #2)
 * 14-Nov-2013 - RKM  - Added event size, adler32 and padding size (version #3)
 * 15-Oct-2014 - WDD  - Event number from 32 bits to 64 bits (version #4)
 * 01-Apr-2015 - SM   - replaced adler32 with crc32c which is accelerated in SSE 4.2 (version #5)
 * 22-Sep-2020 - SM   - reused high version 16-bits for event info via flags (version #6)
 *
 *
 * Version 6 Format:
 *   uint16 - format version number
 *   uint16 - event flags
 *   uint32 - run number
 *   uint32 - lumi number
 *   uint32 - event number
 *   uint32 - event size
 *   uint32 - crc32c checksum of FED data (excluding event header)
 *   variable size - FED data
 *
 * Version 5 Format:
 *   uint32 - format version number
 *   uint32 - run number
 *   uint32 - lumi number
 *   uint32 - event number
 *   uint32 - event size
 *   uint32 - crc32c checksum of FED data (excluding event header)
 *   variable size - FED data
 *
 * Version 4 Format:
 *   uint32 - format version number
 *   uint32 - run number
 *   uint32 - lumi number
 *   uint32 - event number low 32 bits
 *   uint32 - event number high 32 bits
 *   uint32 - event size
 *   uint32 - padding size needed to fill memory page size (_SC_PAGE_SIZE)
 *   uint32 - adler32 checksum of FED data (excluding event header)
 *   variable size - FED data
 *
 * Version 3 Format:
 *   uint32 - format version number
 *   uint32 - run number
 *   uint32 - lumi number
 *   uint32 - event number
 *   uint32 - event size
 *   uint32 - padding size needed to fill memory page size (_SC_PAGE_SIZE)
 *   uint32 - adler32 checksum of FED data (excluding event header)
 *   variable size - FED data
 *
 * Version 2 Format:
 *   uint32 - format version number
 *   uint32 - run number
 *   uint32 - lumi number
 *   uint32 - event number
 *   1024 * uint32 - size values for all 1024 FED buffers
 *   variable size - FED data
 */

#include "IOPool/Streamer/interface/MsgTools.h"

#include <array>

struct FRDEventHeader_V6 {
  uint16 version_;
  uint16 flags_;
  uint32 run_;
  uint32 lumi_;
  uint32 event_;
  uint32 eventSize_;
  uint32 crc32c_;
};

struct FRDEventHeader_V5 {
  uint32 version_;
  uint32 run_;
  uint32 lumi_;
  uint32 event_;
  uint32 eventSize_;
  uint32 crc32c_;
};

struct FRDEventHeader_V4 {
  uint32 version_;
  uint32 run_;
  uint32 lumi_;
  uint32 eventLow_;
  uint32 eventHigh_;
  uint32 eventSize_;
  uint32 paddingSize_;
  uint32 adler32_;
};

struct FRDEventHeader_V3 {
  uint32 version_;
  uint32 run_;
  uint32 lumi_;
  uint32 event_;
  uint32 eventSize_;
  uint32 paddingSize_;
  uint32 adler32_;
};

struct FRDEventHeader_V2 {
  uint32 version_;
  uint32 run_;
  uint32 lumi_;
  uint32 event_;
};

struct FRDEventHeader_V1 {
  uint32 run_;
  uint32 event_;
};

const uint16 FRDEVENT_MASK_ISGENDATA = 1;

constexpr size_t FRDHeaderMaxVersion = 6;
constexpr std::array<uint32, FRDHeaderMaxVersion + 1> FRDHeaderVersionSize{{0,
                                                                            2 * sizeof(uint32),
                                                                            (4 + 1024) * sizeof(uint32),
                                                                            7 * sizeof(uint32),
                                                                            8 * sizeof(uint32),
                                                                            6 * sizeof(uint32),
                                                                            6 * sizeof(uint32)}};

class FRDEventMsgView {
public:
  FRDEventMsgView(void* buf);

  uint8* startAddress() const { return buf_; }
  void* payload() const { return payload_; }
  uint32 size() const { return size_; }

  uint16 version() const { return version_; }
  uint16 flags() const { return flags_; }
  uint32 run() const { return run_; }
  uint32 lumi() const { return lumi_; }
  uint64 event() const { return event_; }
  uint32 eventSize() const { return eventSize_; }
  uint32 paddingSize() const { return paddingSize_; }
  uint32 adler32() const { return adler32_; }
  uint32 crc32c() const { return crc32c_; }

  bool isRealData() const { return !(flags_ & FRDEVENT_MASK_ISGENDATA); }

private:
  uint8* buf_;
  void* payload_;
  uint32 size_;
  uint16 version_;
  uint16 flags_;
  uint32 run_;
  uint32 lumi_;
  uint64 event_;
  uint32 eventSize_;
  uint32 paddingSize_;
  uint32 adler32_;
  uint32 crc32c_;
};

#endif

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

struct FRDEventHeader_V3
{
  uint32 version_;
  uint32 run_;
  uint32 lumi_;
  uint32 event_;
  uint32 eventSize_;
  uint32 paddingSize_;
  uint32 adler32_;
};

struct FRDEventHeader_V2
{
  uint32 version_;
  uint32 run_;
  uint32 lumi_;
  uint32 event_;
};

struct FRDEventHeader_V1
{
  uint32 run_;
  uint32 event_;
};

class FRDEventMsgView
{
 public:

  FRDEventMsgView(void* buf);

  uint8* startAddress() const { return buf_; }
  void* payload() const { return payload_; }
  uint32 size() const { return size_; }

  uint32 version() const { return version_; }
  uint32 run() const { return run_; }
  uint32 lumi() const { return lumi_; }
  uint32 event() const { return event_; }
  uint32 eventSize() const { return eventSize_; }
  uint32 paddingSize() const { return paddingSize_; }
  uint32 adler32() const { return adler32_; }

 private:

  uint8* buf_;
  void* payload_;
  uint32 size_;
  uint32 version_;
  uint32 run_;
  uint32 lumi_;
  uint32 event_;
  uint32 eventSize_;
  uint32 paddingSize_;
  uint32 adler32_;
};

#endif

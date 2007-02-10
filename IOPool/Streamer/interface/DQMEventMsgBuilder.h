#ifndef _DQMEventMsgBuilder_h
#define _DQMEventMsgBuilder_h

/**
 * This class is used to build and view the DQM Event data
 * that is exchanged between the filter units and the storage manager.
 *
 * 09-Feb-2007 - Initial Implementation
 */

/**
 * DQM Event Message Format:
 * - Code (1 byte)
 * - Size (4 bytes)
 * - Run Number (4 bytes)
 * - Reserved Word (4 bytes)
 * - Folder Name Length (4 bytes)
 * - Folder Name (varies)
 * - DQM Event Data Length (4 bytes)
 * - DQM Event Data (varies)
 */

#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/MsgHeader.h"
#include "IOPool/Streamer/interface/DQMEventMessage.h"

// ------------------ dqm event message builder ----------------

class DQMEventMsgBuilder
{
 public:
  DQMEventMsgBuilder(void* buf, uint32 bufSize, uint32 run,
                     std::string const& folderName);

  uint32 bufferSize() const { return bufSize_; }
  uint8* startAddress() const { return buf_; }
  uint32 headerSize() const { return eventAddr_-buf_; }
  void setReserved(uint32);
  uint8* eventAddress() const  { return eventAddr_; }
  void setEventLength(uint32 len);
  uint32 size() const;

 private:
  uint8* buf_;
  uint32 bufSize_;
  uint8* eventAddr_;
};

#endif

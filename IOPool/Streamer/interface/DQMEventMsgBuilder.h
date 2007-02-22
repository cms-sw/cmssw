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
 * - Protocol Version (4 bytes)
 * - Header Size (4 bytes)
 * - Run Number (4 bytes)
 * - Event Number at Update (4 bytes)
 * - Compression Reserved Word (4 bytes)
 * - Reserved Word (4 bytes)
 * - Release Tag Length (4 bytes)
 * - Release Tag (varies)
 * - Top-level Folder Name Length (4 bytes)
 * - Top-level Folder Name (varies)
 * - DQM Event Data Length (4 bytes)
 * - DQM Event Data (varies)
 *
 * The DQM Event Data has sub-structure:
 * - Number of Subfolders (4 bytes)
 * - Subfolder Name Length (4 bytes)                     | Repeated
 * - Subfolder Name (varies)                             | structure
 * - Number of Monitor Elements in Subfolder (4 bytes)   | for each
 * - ME Data Length (4 bytes)                            | subfolder
 * - ME Data (varies)                                    |
 */

#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/MsgHeader.h"
#include "IOPool/Streamer/interface/DQMEventMessage.h"
#include <TBuffer.h>

// ------------------ dqm event message builder ----------------

class DQMEventMsgBuilder
{
 public:
  DQMEventMsgBuilder(void* buf, uint32 bufSize, uint32 run, uint32 event,
                     std::string const& releaseTag,
                     std::string const& topFolderName);

  uint32 bufferSize() const { return bufSize_; }
  uint8* startAddress() const { return buf_; }
  uint32 headerSize() const { return eventAddr_-buf_; }
  void setCompressionFlag(uint32);
  void setReserved(uint32);
  uint8* eventAddress() const  { return eventAddr_; }
  void setEventLength(uint32 len);
  uint32 size() const;
  uint32 eventLength() const;

  void addMEData(std::string const& subFolderName,
                 uint32 const monitorElementCount,
                 TBuffer const& serializedMEData);

 private:
  uint8* buf_;
  uint32 bufSize_;
  uint8* eventAddr_;
};

#endif

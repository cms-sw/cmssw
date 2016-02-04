#ifndef IOPool_Streamer_DQMEventMsgBuilder_h
#define IOPool_Streamer_DQMEventMsgBuilder_h

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
 * - Luminosity Section (4 bytes)
 * - Update Number (4 bytes)
 * - Compression Flag (4 bytes)   | size of data before compression
 * - Filter Unit Process ID (4 bytes)
 * - Filter Unit Unique ID [GUID] (4 bytes)
 * - Reserved Word (4 bytes)
 * - Release Tag Length (4 bytes)
 * - Release Tag (varies)
 * - Top-level Folder Name Length (4 bytes)
 * - Top-level Folder Name (varies)
 * - Number of Subfolders (4 bytes)
 * - Number of Monitor Elements in Subfolder I (4 bytes)   | Repeated
 * - Subfolder I Name Length (4 bytes)                     | for each
 * - Subfolder I Name (varies)                             | subfolder
 * - DQM Event Data checksum (4 bytes)
 * - Host name length (1 byte)
 * - Host name (variable)
 * - DQM Event Data Length (4 bytes)
 * - DQM Event Data (varies)
 */

#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/DQMEventMessage.h"

#include "DataFormats/Provenance/interface/Timestamp.h"


// ------------------ dqm event message builder ----------------

class DQMEventMsgBuilder
{
 public:
  DQMEventMsgBuilder(void* buf, uint32 bufSize, uint32 run, uint32 event,
		     edm::Timestamp timeStamp,
                     uint32 lumiSection, uint32 updateNumber,
                     uint32 adler32_chksum,
                     const char* host_name,
                     std::string const& releaseTag,
                     std::string const& topFolderName,
                     DQMEvent::TObjectTable monitorElementsBySubFolder);

  uint32 bufferSize() const { return bufSize_; }
  uint8* startAddress() const { return buf_; }
  uint32 headerSize() const { return eventAddr_-buf_; }
  void setCompressionFlag(uint32);
  void setFUProcessId(uint32);
  void setFUGuid(uint32);
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

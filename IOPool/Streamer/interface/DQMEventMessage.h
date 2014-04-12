#ifndef IOPool_Streamer_DQMEventMessage_h
#define IOPool_Streamer_DQMEventMessage_h

/**
 * The DQMEventMsgView class is used to view the DQM data messages that
 * are exchanged between the filter units and the storage manager.
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
 * - Time stamp (8 bytes)
 * - Luminosity Section (4 bytes)
 * - Update Number (4 bytes)
 * - Compression Flag (4 bytes)   | size of data before compression
 * - Filter Unit Process ID (4 bytes)
 * - Filter Unit Unique ID [GUID] (4 bytes)
 * - Merge Count (4 bytes)
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
#include "IOPool/Streamer/interface/MsgHeader.h"
#include "boost/shared_ptr.hpp"
#include "TObject.h"
#include <map>

#include "DataFormats/Provenance/interface/Timestamp.h"


// ------------------ dqm event message ----------------

namespace DQMEvent
{
  typedef std::map< std::string, std::vector<TObject *> >
      TObjectTable;
}

struct DQMEventHeader
{
  Header header_;
  char_uint32 protocolVersion_;
  char_uint32 headerSize_;
  char_uint32 runNumber_;
  char_uint32 eventNumber_;
  char_uint64 timeStamp_;
  char_uint32 lumiSection_;
  char_uint32 updateNumber_;
  char_uint32 compressionFlag_;
  char_uint32 fuProcessId_;
  char_uint32 fuGuid_;
  char_uint32 mergeCount_;
};

class DQMEventMsgView
{
 public:
  DQMEventMsgView(void* buf);

  uint32 code() const { return head_.code(); }
  uint32 size() const { return head_.size(); }

  uint8* startAddress() const { return buf_; }
  uint8* eventAddress() const { return eventAddr_; }
  uint32 eventLength() const { return eventLen_; }
  uint32 adler32_chksum() const {return adler32_chksum_;}
  std::string hostName() const;
  uint32 hostName_len() const {return host_name_len_;}

  uint32 protocolVersion() const;
  uint32 headerSize() const;
  uint32 runNumber() const;
  uint32 eventNumberAtUpdate() const;
  uint32 lumiSection() const;
  uint32 updateNumber() const;
  uint32 compressionFlag() const;
  uint32 fuProcessId() const;
  uint32 fuGuid() const;
  uint32 mergeCount() const;

  edm::Timestamp timeStamp() const;

  std::string releaseTag() const { return releaseTag_; }
  std::string topFolderName() const { return folderName_; }

  uint32 subFolderCount() const { return subFolderCount_; }
  boost::shared_ptr< std::vector<std::string> > subFolderNames() const;

  std::string subFolderName(uint32 const subFolderIndex) const;
  uint32 meCount(std::string const& subFolderName) const;
  uint32 meCount(uint32 const subFolderIndex) const;

 private:
  uint8* buf_;
  HeaderView head_;
  std::string releaseTag_;
  std::string folderName_;
  uint8* eventAddr_;
  uint32 eventLen_;
  uint32 adler32_chksum_;
  uint8* host_name_start_;
  uint32 host_name_len_;
  uint32 subFolderCount_;
  std::map<std::string, uint32> subFolderIndexTable_;
  boost::shared_ptr< std::vector<std::string> > nameListPtr_;
  std::vector<uint32> meCountList_;
};

#endif

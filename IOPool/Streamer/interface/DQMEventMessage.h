#ifndef _DQMEventMessage_h
#define _DQMEventMessage_h

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
#include "boost/shared_ptr.hpp"
#include <TObject.h>
#include <map>

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
  char_uint32 compressionFlag_;
  char_uint32 reserved_;
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

  uint32 protocolVersion() const;
  uint32 headerSize() const;
  uint32 runNumber() const;
  uint32 eventNumberAtUpdate() const;
  uint32 compressionFlag() const;
  uint32 reserved() const;

  std::string releaseTag() const { return releaseTag_; }
  std::string topFolderName() const { return folderName_; }

  uint32 subFolderCount() const { return subFolderCount_; }
  boost::shared_ptr< std::vector<std::string> > subFolderNames() const;

  uint32 meCount(std::string const& subFolderName) const;
  uint32 meCount(uint32 const subFolderIndex) const;
  uint32 meDataSize(std::string const& subFolderName) const;
  uint32 meDataSize(uint32 const subFolderIndex) const;
  uint8* meDataAddress(std::string const& subFolderName) const;
  uint8* meDataAddress(uint32 const subFolderIndex) const;

 private:
  uint8* buf_;
  HeaderView head_;
  std::string releaseTag_;
  std::string folderName_;
  uint8* eventAddr_;
  uint32 eventLen_;
  uint32 subFolderCount_;
  boost::shared_ptr< std::vector<std::string> > nameListPtr_;
  std::map<std::string, uint32> subFolderIndexTable_;
  std::vector<uint32> meCountList_;
  std::vector<uint32> meSizeList_;
  std::vector<uint8*> meAddressList_;
};

#endif

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
 * - Run Number (4 bytes)
 * - Reserved Word (4 bytes)
 * - Folder Name Length (4 bytes)
 * - Folder Name (varies)
 * - DQM Event Data Length (4 bytes)
 * - DQM Event Data (varies)
 */

#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/MsgHeader.h"

// ------------------ dqm event message ----------------

struct DQMEventHeader
{
  Header header_;
  char_uint32 runNumber_;
  char_uint32 reserved_;
};

class DQMEventMsgView
{
 public:
  DQMEventMsgView(void* buf);

  uint32 code() const { return head_.code(); }
  uint32 size() const { return head_.size(); }

  uint8* startAddress() const { return buf_; }
  uint32 headerSize() const { return eventAddr_-buf_; }
  uint8* eventAddress() const { return eventAddr_; }
  uint32 eventLength() const { return eventLen_; }

  uint32 run() const;
  uint32 reserved() const;

  std::string getFolderName() const { return folderName_; }

 private:
  uint8* buf_;
  HeaderView head_;
  std::string folderName_;
  uint8* eventAddr_;
  uint32 eventLen_;
};

#endif

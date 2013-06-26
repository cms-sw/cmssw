#ifndef IOPool_Streamer_MsgHeader_h
#define IOPool_Streamer_MsgHeader_h

#include "IOPool/Streamer/interface/MsgTools.h"
// as it is in memory of file
struct Header
{
  Header(uint32 code,uint32 size):code_(code)
  { convert(size,size_); }

  uint8 code_; // type of the message
  char_uint32 size_; // of entire message including all headers

  // 20-Jul-2006, KAB: added enumeration for message types
  enum Codes { INVALID = 0, INIT = 1, EVENT = 2, DONE = 3, EOFRECORD = 4,
               HEADER_REQUEST = 5, EVENT_REQUEST = 6,
               CONS_REG_REQUEST = 7, CONS_REG_RESPONSE = 8,
               DQM_INIT = 9, DQM_EVENT = 10, DQMEVENT_REQUEST = 11,
               INIT_SET = 12, NEW_INIT_AVAILABLE = 13, ERROR_EVENT = 14,
               FILE_CLOSE_REQUEST = 15, SPARE1 = 16, SPARE2 = 17 };
};

// as we need to see it
class HeaderView
{
public:
  HeaderView(void* buf)
  {
    Header* h = (Header*)buf;
    code_ = h->code_;
    size_ = convert32(h->size_);
  }

  uint32 code() const { return code_; }
  uint32 size() const { return size_; }
private:
  uint32 code_;
  uint32 size_;
};

#endif

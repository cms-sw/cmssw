#ifndef _MsgHeader_h
#define _MsgHeader_h

#include "IOPool/Streamer/interface/MsgTools.h"
// as it is in memory of file
struct Header
{
  Header(uint32 code,uint32 size):code_(code)
  { convert(size,size_); }

  uint8 code_; // type of the message
  char_uint32 size_; // of entire message including all headers
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

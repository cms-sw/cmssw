#ifndef _EventMsgBuilder_h
#define _EventMsgBuilder_h

#include "MsgTools.h"
#include "MsgHeader.h"
#include "EventMessage.h"

// ------------------ event message builder ----------------

class EventMsgBuilder
{
public:
  EventMsgBuilder(void* buf, uint32 size,
                  uint32 run, uint32 event, uint32 lumi,
                  std::vector<bool>& l1_bits,
                  uint8* hlt_bits, uint32 hlt_bit_count);

  void setReserved(uint32);
  void setEventLength(uint32 len);
  uint8* eventAddr() { return event_addr_; }

  uint32 size() const;

private:
  uint8* buf_;
  uint32 size_;
  uint8* event_addr_;
};

#endif


#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/MsgHeader.h"
#include <cassert>
#include <cstring>

#define MAX_HOSTNAME_LEN 25

EventMsgBuilder::EventMsgBuilder(void* buf, uint32 size,
                                 uint32 run, uint32 event, uint32 lumi,
                                 uint32 outModId, uint32 droppedEventsCount,
                                 std::vector<bool>& l1_bits,
                                 uint8* hlt_bits, uint32 hlt_bit_count, 
                                 uint32 adler_chksum, const char* host_name):
  buf_((uint8*)buf),size_(size)
{
  EventHeader* h = (EventHeader*)buf_;
  h->protocolVersion_ = 9;
  convert(run,h->run_);
  convert(event,h->event_);
  convert(lumi,h->lumi_);
  convert(outModId,h->outModId_);
  convert(droppedEventsCount,h->droppedEventsCount_);
  uint8* pos = buf_ + sizeof(EventHeader);

  // l1 count
  uint32 l1_count = l1_bits.size();
  convert(l1_count, pos);
  pos = pos + sizeof(uint32); 

  // set the l1 
  uint32 l1_sz = l1_bits.size();
  if (l1_sz != 0) l1_sz = 1 + ((l1_sz-1)/8);
 
  uint8* pos_end = pos + l1_sz;
  memset(pos,0x00, pos_end-pos); // clear the bits
  for(std::vector<bool>::size_type i = 0; i < l1_bits.size(); ++i) {
      uint8 v = l1_bits[i] ? 1 : 0;
      pos[i/8] |= (v << (i&0x07));
  }
  pos = pos_end;

  // hlt count
  convert(hlt_bit_count, pos); 
  pos = pos + sizeof(uint32);

  uint32 hlt_sz = hlt_bit_count;
  if (hlt_sz != 0) hlt_sz = 1+ ((hlt_sz-1)/4);

  // copy the hlt bits over
  pos = std::copy(hlt_bits, hlt_bits+hlt_sz, pos);

  // adler32 check sum of data blob
  convert(adler_chksum, pos);
  pos = pos + sizeof(uint32);

  // put host name (Length and then Name) right after check sum
  //uint32 host_name_len = strlen(host_name);
  // actually make the host_name a fixed length as the event header size appears in the
  // Init message and only one goes to a file whereas events can come from any node
  // We want the max length to be determined inside this Event Message Builder
  uint32 host_name_len = MAX_HOSTNAME_LEN;
  assert(host_name_len < 0x00ff);
  //Put host_name_len
  *pos++ = host_name_len;

  //Put host_name
  uint32 real_len = strlen(host_name);
  if(real_len < host_name_len) {
    char hostname_2use[MAX_HOSTNAME_LEN];
    memset(hostname_2use,'\0',host_name_len);
    memcpy(hostname_2use,host_name,real_len);
    memcpy(pos,hostname_2use,host_name_len);
  } else {
    memcpy(pos,host_name,host_name_len);
  }
  pos += host_name_len;

  event_addr_ = pos + sizeof(char_uint32);
  setEventLength(0);
}

void EventMsgBuilder::setOrigDataSize(uint32 value)
{
  EventHeader* h = (EventHeader*)buf_;
  convert(value,h->origDataSize_);
}

void EventMsgBuilder::setEventLength(uint32 len)
{
  convert(len,event_addr_-sizeof(char_uint32));
  EventHeader* h = (EventHeader*)buf_;
  new (&h->header_) Header(Header::EVENT,event_addr_-buf_+len);
}

uint32 EventMsgBuilder::size() const
{
  HeaderView v(buf_);
  return v.size();
}

/** Event Message Represented here

Protocol Versions 1-4:
code 1 | size 4 | run 4 | event 4 | lumi 4 | reserved 4 |
l1_count 4| l1bits l1_count/8 | 
hlt_count 4| hltbits hlt_count/4 |
eventdatalength 4 | eventdata blob {variable} 

Protocol Version 5:
code 1 | size 4 | protocol version 1 |
run 4 | event 4 | lumi 4 | origDataSize 4 | outModId 4 |
l1_count 4| l1bits l1_count/8 | 
hlt_count 4| hltbits hlt_count/4 |
eventdatalength 4 | eventdata blob {variable} 

Protocol Version 6:  // no change here, only INIT msg change
code 1 | size 4 | protocol version 1 |
run 4 | event 4 | lumi 4 | origDataSize 4 | outModId 4 |
l1_count 4| l1bits l1_count/8 | 
hlt_count 4| hltbits hlt_count/4 |
eventdatalength 4 | eventdata blob {variable} 

Protocol Version 7:  // no change here, only data blob changes
code 1 | size 4 | protocol version 1 |
run 4 | event 4 | lumi 4 | origDataSize 4 | outModId 4 |
l1_count 4| l1bits l1_count/8 | 
hlt_count 4| hltbits hlt_count/4 |
eventdatalength 4 | eventdata blob {variable} 

Protocol Version 8:  // add in checksum of data blob changes
code 1 | size 4 | protocol version 1 |
run 4 | event 4 | lumi 4 | origDataSize 4 | outModId 4 |
l1_count 4| l1bits l1_count/8 | 
hlt_count 4| hltbits hlt_count/4 |
adler32_chksum 4 | host name length 1 | host name {Fixed size}
eventdatalength 4 | eventdata blob {variable} 

Protocol Version 9:  // add dropped events counter
code 1 | size 4 | protocol version 1 |
run 4 | event 4 | lumi 4 | origDataSize 4 | outModId 4 |
droppedEventsCount 4 |
l1_count 4 | l1bits l1_count/8 | 
hlt_count 4 | hltbits hlt_count/4 |
adler32_chksum 4 | host name length 1 | host name {Fixed size}
eventdatalength 4 | eventdata blob {variable} 

Protocol Version 10: identical to version 9, but incremented to keep in sync with init msg version 

Protocol Version 11: identical to version 10, except event changed from 4 bytes to 8 bytes

*/

#ifndef IOPool_Streamer_EventMessage_h
#define IOPool_Streamer_EventMessage_h

#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/MsgHeader.h"

// ----------------------- event message ------------------------

struct EventHeader
{
  Header header_;
  uint8 protocolVersion_;
  char_uint32 run_;
  char_uint64 event_;
  char_uint32 lumi_;
  char_uint32 origDataSize_;
  char_uint32 outModId_;
  char_uint32 droppedEventsCount_;
};

class EventMsgView
{
public:

  EventMsgView(void* buf);

  uint32 code() const { return head_.code(); }
  uint32 size() const { return head_.size(); }

  const uint8* eventData() const { return event_start_; }
  uint8* startAddress() const { return buf_; }
  uint32 eventLength() const { return event_len_; }
  uint32 headerSize() const {return event_start_-buf_;}
  uint32 protocolVersion() const;
  uint32 run() const;
  uint64 event() const;
  uint32 lumi() const;
  uint32 origDataSize() const;
  uint32 outModId() const;
  uint32 droppedEventsCount() const;

  void l1TriggerBits(std::vector<bool>& put_here) const;
  void hltTriggerBits(uint8* put_here) const;

  uint32 hltCount() const {return hlt_bits_count_;}
  uint32 l1Count() const {return l1_bits_count_;}
  uint32 adler32_chksum() const {return adler32_chksum_;}
  std::string hostName() const;
  uint32 hostName_len() const {return host_name_len_;}

private:
  uint8* buf_;
  HeaderView head_;

  uint8* hlt_bits_start_;
  uint32 hlt_bits_count_;
  uint8* l1_bits_start_;
  uint32 l1_bits_count_;
  uint8* event_start_;
  uint32 event_len_;
  uint32 adler32_chksum_;
  uint8* host_name_start_;
  uint32 host_name_len_;
  bool v2Detected_;
};

#endif

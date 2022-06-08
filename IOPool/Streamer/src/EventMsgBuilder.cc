#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/MsgHeader.h"
#include <cassert>
#include <cstring>

#define MAX_HOSTNAME_LEN 25

EventMsgBuilder::EventMsgBuilder(void* buf,
                                 uint32 size,
                                 uint32 run,
                                 uint64 event,
                                 uint32 lumi,
                                 uint32 outModId,
                                 uint32 droppedEventsCount,
                                 std::vector<bool>& l1_bits,
                                 uint8* hlt_bits,
                                 uint32 hlt_bit_count,
                                 uint32 adler_chksum,
                                 const char* host_name)
    : buf_((uint8*)buf), size_(size) {
  uint32 expectedHeaderSize = computeHeaderSize(l1_bits.size(), hlt_bit_count);
  if (expectedHeaderSize > size_) {
    throw cms::Exception("EventMsgBuilder")
        << "The buffer used to build the event message (" << size_
        << " bytes) is not large enough to holde the event header (" << expectedHeaderSize << " bytes)";
  }

  // Note: any change to the pos increment logic should be reflected in computeHeaderSize()
  uint8* pos = buf_;

  // Set the event header
  EventHeader* h = (EventHeader*)pos;
  h->protocolVersion_ = 11;
  convert(run, h->run_);
  convert(event, h->event_);
  convert(lumi, h->lumi_);
  convert(outModId, h->outModId_);
  convert(droppedEventsCount, h->droppedEventsCount_);
  pos += sizeof(EventHeader);

  // Set the L1T triggers count
  uint32 l1_count = l1_bits.size();
  convert(l1_count, pos);
  pos += sizeof(uint32);

  // Set the L1T bits
  uint32 l1_sz = (l1_bits.size() + 8 - 1) / 8;  // L1T results (1 bit per trigger)
  memset(pos, 0x00, l1_sz);                     // clear the bits
  for (std::vector<bool>::size_type i = 0; i < l1_bits.size(); ++i) {
    uint8 v = l1_bits[i] ? 1 : 0;
    pos[i / 8] |= (v << (i & 0x07));
  }
  pos += l1_sz;

  // Set HLT triggers count
  convert(hlt_bit_count, pos);
  pos += sizeof(uint32);

  // Copy the HLT bits and increment pos
  uint32 hlt_sz = (hlt_bit_count + 4 - 1) / 4;  // HLT results (2 bits per trigger)
  pos = std::copy(hlt_bits, hlt_bits + hlt_sz, pos);

  // Set the Adler32 check sum of data blob
  convert(adler_chksum, pos);
  pos += sizeof(uint32);

  // Put the host name (length and then name) right after the check sum.
  // Use a fixed length for the host_name because the event header size appears in the
  // init message and only one goes to a file whereas events can come from any node,
  // while we want the max length to be determined inside this Event Message Builder.

  // Set the host_name_len and increment pos
  assert(MAX_HOSTNAME_LEN < 0x00ff);
  *pos++ = MAX_HOSTNAME_LEN;

  // Copy up to MAX_HOSTNAME_LEN characters of the host_name and pad any extra space
  // with null characters
  memset(pos, '\0', MAX_HOSTNAME_LEN);
  strncpy((char*)pos, host_name, MAX_HOSTNAME_LEN - 1);
  pos += MAX_HOSTNAME_LEN;

  event_addr_ = pos + sizeof(char_uint32);
  setEventLength(0);

  // Check that the size computed by computeHeaderSize() matches what is actually used.
  if (headerSize() != expectedHeaderSize) {
    throw cms::Exception("EventMsgBuilder")
        << "The event message header size (" << headerSize() << " bytes) does not match the computed value ("
        << expectedHeaderSize << " bytes)";
  }
}

void EventMsgBuilder::setOrigDataSize(uint32 value) {
  EventHeader* h = (EventHeader*)buf_;
  convert(value, h->origDataSize_);
}

void EventMsgBuilder::setEventLength(uint32 len) {
  convert(len, event_addr_ - sizeof(char_uint32));
  EventHeader* h = (EventHeader*)buf_;
  new (&h->header_) Header(Header::EVENT, event_addr_ - buf_ + len);
}

uint32 EventMsgBuilder::size() const {
  HeaderView v(buf_);
  return v.size();
}

uint32 EventMsgBuilder::computeHeaderSize(uint32 l1t_bit_count, uint32 hlt_bit_count) {
  uint32 size = sizeof(EventHeader);    // event header
  size += sizeof(uint32);               // L1T triggers count
  size += (l1t_bit_count + 8 - 1) / 8;  // L1T results (1 bit per trigger)
  size += sizeof(uint32);               // HLT triggers count
  size += (hlt_bit_count + 4 - 1) / 4;  // HLT results (2 bits per trigger)
  size += sizeof(uint32);               // adler32 check sum
  size += 1;                            // host name length
  size += MAX_HOSTNAME_LEN;             // host name
  size += sizeof(char_uint32);          // event address
  return size;
}

#include "IOPool/Streamer/interface/EventMessage.h"
#include "FWCore/Utilities/interface/Exception.h"


EventMsgView::EventMsgView(void* buf):
  buf_((uint8*)buf),head_(buf),
  v2Detected_(false)
{ 
  // 29-Jan-2008, KAB - adding an explicit version number.
  // We'll start with 5 to match the new version of the INIT message.
  // We support earlier versions of the full protocol, of course, but since
  // we didn't have an explicit version number in the Event Message before
  // now, we have to limit what we can handle to versions that have the
  // version number included (>= 5).
  if (protocolVersion() < 5 || protocolVersion() > 5) {
    throw cms::Exception("EventMsgView", "Invalid Message Version:")
      << "Only message version 5 is currently supported "
      << "(invalid value = " << protocolVersion() << ").\n";
  }

  uint8* l1_bit_size_ptr = buf_ + sizeof(EventHeader); //Just after Header 
  l1_bits_count_ = convert32(l1_bit_size_ptr); 
  uint32 l1_sz = l1_bits_count_;
  //Lets detect if thats V2 message
  if (l1_bits_count_ == 11) {
          l1_sz = 1; 
          v2Detected_=true;
  }

  l1_bits_start_ = buf_ + sizeof(EventHeader) + sizeof(uint32); 

  if (v2Detected_ == false) { 
     if (l1_sz != 0) l1_sz = 1 + ((l1_sz-1)/8);
  }
  uint8* hlt_bit_size_ptr = l1_bits_start_ + l1_sz; 
  hlt_bits_count_ = convert32(hlt_bit_size_ptr); 
  hlt_bits_start_ = hlt_bit_size_ptr + sizeof(uint32); 
  uint32 hlt_sz = hlt_bits_count_;
  if (hlt_sz != 0) hlt_sz = 1+ ((hlt_sz-1)/4);

  if(v2Detected_) hlt_sz=2;
  event_start_ = hlt_bits_start_ + hlt_sz; 
  event_len_ = convert32(event_start_); 
  event_start_ += sizeof(char_uint32); 
}

uint32 EventMsgView::protocolVersion() const
{
  EventHeader* h = (EventHeader*)buf_;
  return h->protocolVersion_;
}

uint32 EventMsgView::run() const
{
  EventHeader* h = (EventHeader*)buf_;
  return convert32(h->run_);
}

uint32 EventMsgView::event() const
{
  EventHeader* h = (EventHeader*)buf_;
  return convert32(h->event_);
}

uint32 EventMsgView::lumi() const
{
  EventHeader* h = (EventHeader*)buf_;
  return convert32(h->lumi_);
}

uint32 EventMsgView::origDataSize() const
{
  EventHeader* h = (EventHeader*)buf_;
  return convert32(h->origDataSize_);
}

uint32 EventMsgView::outModId() const
{
  EventHeader* h = (EventHeader*)buf_;
  return convert32(h->outModId_);
}

void EventMsgView::l1TriggerBits(std::vector<bool>& put_here) const
{
  put_here.clear();
  put_here.resize(l1_bits_count_);

  for(std::vector<bool>::size_type i = 0; i < l1_bits_count_; ++i)
    put_here[i] = (bool)(l1_bits_start_[i/8] & (1<<((i&0x07))));
}

void EventMsgView::hltTriggerBits(uint8* put_here) const
{
  uint32 hlt_sz = hlt_bits_count_;
  if (hlt_sz != 0) hlt_sz = 1 + ((hlt_sz-1)/4);

  if(v2Detected_) hlt_sz=2;

  std::copy(hlt_bits_start_,hlt_bits_start_ + hlt_sz,
            put_here);
}



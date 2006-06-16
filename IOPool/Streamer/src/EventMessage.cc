#include "IOPool/Streamer/interface/EventMessage.h"


EventMsgView::EventMsgView(void* buf, uint32 size,
                           uint32 hlt_bit_cnt, uint32 l1_bit_cnt):
  buf_((uint8*)buf),size_(size),head_(buf,size),
  hlt_bits_count_(hlt_bit_cnt),l1_bits_count_(l1_bit_cnt)
{
  l1_bits_start_ = buf_ + sizeof(EventHeader);
  hlt_bits_start_ = l1_bits_start_ + (l1_bits_count_ / 8);
  event_start_ = hlt_bits_start_ + (hlt_bits_count_ / 4);
  event_len_ = convert32(event_start_);
  event_start_ += sizeof(char_uint32);

  //cout << "l1 at offset=" << (l1_bits_start_-buf_) << "\n";
  //cout << "hlt at offset=" << (hlt_bits_start_-buf_) << "\n";
  //cout << "event at offset=" << (event_start_-buf_) << "\n";
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

uint32 EventMsgView::reserved() const
{
  EventHeader* h = (EventHeader*)buf_;
  return convert32(h->reserved_);
}

void EventMsgView::l1TriggerBits(std::vector<bool>& put_here) const
{
  put_here.clear();
  put_here.resize(l1_bits_count_);


  for(unsigned int i=0;i<l1_bits_count_;++i)
    put_here[i] = (bool)(l1_bits_start_[i/8] & (1<<((i&0x07))));
}

void EventMsgView::hltTriggerBits(uint8* put_here) const
{
  std::copy(hlt_bits_start_,hlt_bits_start_ + (hlt_bits_count_/4),
            put_here);
}



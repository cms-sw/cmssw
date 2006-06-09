#include "IOPool/Streamer/interface/InitMessage.h"


InitMsgView::InitMsgView(void* buf, uint32 size):
  buf_((uint8*)buf),size_(size),head_(buf,size)
{

  release_start_ = buf_ + sizeof(InitHeader);
  release_len_ = *release_start_;
  release_start_ += sizeof(uint8);

  hlt_trig_start_ = release_start_ + release_len_;
  hlt_trig_count_ = convert32(hlt_trig_start_);
  hlt_trig_start_ += sizeof(char_uint32);
  hlt_trig_len_ = convert32(hlt_trig_start_);
  hlt_trig_start_ += sizeof(char_uint32);
  l1_trig_start_ = hlt_trig_start_ + hlt_trig_len_;
  l1_trig_count_ = convert32(l1_trig_start_);
  l1_trig_start_ += sizeof(char_uint32);
  l1_trig_len_ = convert32(l1_trig_start_);
  l1_trig_start_ += sizeof(char_uint32);

  desc_start_ = l1_trig_start_ + l1_trig_len_;
  desc_len_ = convert32(desc_start_);
  desc_start_ += sizeof(char_uint32);
}

uint32 InitMsgView::run() const
{
  InitHeader* h = (InitHeader*)buf_;
  return convert32(h->run_);
}

uint32 InitMsgView::protocolVersion() const
{
  InitHeader* h = (InitHeader*)buf_;
  return h->version_.protocol_;
}

void InitMsgView::pset(uint8* put_here) const
{
  InitHeader* h = (InitHeader*)buf_;
  memcpy(put_here,h->version_.pset_id_,sizeof(h->version_.pset_id_));
}

std::string InitMsgView::releaseTag() const
{
  return std::string((char*)release_start_,release_len_);
                                                                      
}

void InitMsgView::hltTriggerNames(Strings& save_here) const
{
  getNames(hlt_trig_start_,hlt_trig_len_,save_here);
}

void InitMsgView::l1TriggerNames(Strings& save_here) const
{
  getNames(l1_trig_start_,l1_trig_len_,save_here);
}

void InitMsgView::getNames(uint8* from, uint32 from_len, Strings& to) const
{
  // not the most efficient way to do this
  std::istringstream ist(std::string((char*)from,from_len));
  typedef std::istream_iterator<std::string> Iter;
  std::copy(Iter(ist),Iter(),std::back_inserter(to));
}



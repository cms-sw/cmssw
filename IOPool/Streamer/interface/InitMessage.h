#ifndef _InitMessage_h
#define _InitMessage_h

#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/MsgHeader.h"
#include "IOPool/Streamer/interface/InitMessage.h"

struct Version
{
  Version(uint32 proto, const uint8* pset):protocol_(proto)
  { std::copy(pset,pset+sizeof(pset_id_),&pset_id_[0]); }

  uint8 protocol_; // version of the protocol
  unsigned char pset_id_[16]; // parameter set ID
};

struct InitHeader
{
  InitHeader(const Header& h, uint32 run, const Version& v):
    header_(h),version_(v)
  { convert(run,run_); }

  Header header_;
  Version version_;
  char_uint32 run_;
};

class InitMsgView
{
public:

  InitMsgView(void* buf, uint32 size);

  uint32 code() const { return head_.code(); }
  uint32 size() const { return head_.size(); }

  uint32 run() const;
  uint32 protocolVersion() const;
  void pset(uint8* put_here) const;
  std::string releaseTag() const;

  void hltTriggerNames(Strings& save_here) const;
  void l1TriggerNames(Strings& save_here) const;

  // needed for streamer file
  uint32 descLength() const { return desc_len_; }
  const uint8* descData() const { return desc_start_; }

private:
  void getNames(uint8* from, uint32 from_len, Strings& to) const;

  uint8* buf_;
  uint32 size_;

  HeaderView head_;

  uint8* release_start_; // points to the string
  uint32 release_len_;
  uint8* hlt_trig_start_; // points to the string
  uint32 hlt_trig_count_; // number of strings
  uint32 hlt_trig_len_; // length of strings character array only
  uint8* l1_trig_start_; // points to the string
  uint32 l1_trig_count_; // number of strings
  uint32 l1_trig_len_; // length of strings character array only

  // does not need to be present in the message sent over the network,
  // but is needed for the index file
  uint8* desc_start_; // point to the bytes
  uint32 desc_len_;
};

#endif


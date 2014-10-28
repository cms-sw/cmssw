/** Init Message

Protocol Versoion 2:
code 1 | size 4 | protocol version 1 | pset 16 | run 4 | Init Header Size 4| Event Header Seize 4| releaseTagLength 1 | ReleaseTag var| HLT count 4| HLT Trig Legth 4 | HLT Trig names var | L1 Trig Count 4| L1 TrigName len 4| L1 Trig Names var |desc legth 4 | description blob var

Protocol Version 3:
code 1 | size 4 | protocol version 1 | pset 16 | run 4 | Init Header Size 4| Event Header Size 4| releaseTagLength 1 | ReleaseTag var| HLT count 4| HLT Trig Legth 4 | HLT Trig names var | L1 Trig Count 4| L1 TrigName len 4| L1 Trig Names var |desc legth 4 | description blob var

Protocol Version 4:
code 1 | size 4 | protocol version 1 | pset 16 | run 4 | Init Header Size 4| Event Header Size 4| releaseTagLength 1 | ReleaseTag var| processNameLength 1 | processName var | HLT count 4| HLT Trig Legth 4 | HLT Trig names var | L1 Trig Count 4| L1 TrigName len 4| L1 Trig Names var |desc legth 4 | description blob var

Protocol Version 5:
code 1 | size 4 | protocol version 1 | pset 16 | run 4 | Init Header Size 4| Event Header Size 4| releaseTagLength 1 | ReleaseTag var| processNameLength 1 | processName var| outputModuleLabelLength 1 | outputModuleLabel var | HLT Trig count 4| HLT Trig Length 4 | HLT Trig names var | HLT Selection count 4| HLT Selection Length 4 | HLT Selection names var | L1 Trig Count 4| L1 TrigName len 4| L1 Trig Names var |desc legth 4 | description blob var

Protocol Version 6:
code 1 | size 4 | protocol version 1 | pset 16 | run 4 | Init Header Size 4| Event Header Size 4| releaseTagLength 1 | ReleaseTag var| processNameLength 1 | processName var| outputModuleLabelLength 1 | outputModuleLabel var | outputModuleId 4 | HLT Trig count 4| HLT Trig Length 4 | HLT Trig names var | HLT Selection count 4| HLT Selection Length 4 | HLT Selection names var | L1 Trig Count 4| L1 TrigName len 4| L1 Trig Names var |desc legth 4 | description blob var
  
Protocol Version 7: No change to protocol, only description blob (and event data blob) changed
code 1 | size 4 | protocol version 1 | pset 16 | run 4 | Init Header Size 4| Event Header Size 4| releaseTagLength 1 | ReleaseTag var| processNameLength 1 | processName var| outputModuleLabelLength 1 | outputModuleLabel var | outputModuleId 4 | HLT Trig count 4| HLT Trig Length 4 | HLT Trig names var | HLT Selection count 4| HLT Selection Length 4 | HLT Selection names var | L1 Trig Count 4| L1 TrigName len 4| L1 Trig Names var |desc legth 4 | description blob var

Protocol Version 8: added data blob checksum and hostname
code 1 | size 4 | protocol version 1 | pset 16 | run 4 | Init Header Size 4| Event Header Size 4| releaseTagLength 1 | ReleaseTag var| processNameLength 1 | processName var| outputModuleLabelLength 1 | outputModuleLabel var | outputModuleId 4 | HLT Trig count 4| HLT Trig Length 4 | HLT Trig names var | HLT Selection count 4| HLT Selection Length 4 | HLT Selection names var | L1 Trig Count 4| L1 TrigName len 4| L1 Trig Names var | adler32 chksum 4| host name length 4| host name var|desc legth 4 | description blob var

Protocol Version 9: identical to version 8, but incremented to keep in sync with event msg protocol version 

Protocol Version 10: removed hostname
code 1 | size 4 | protocol version 1 | pset 16 | run 4 | Init Header Size 4| Event Header Size 4| releaseTagLength 1 | ReleaseTag var| processNameLength 1 | processName var| outputModuleLabelLength 1 | outputModuleLabel var | outputModuleId 4 | HLT Trig count 4| HLT Trig Length 4 | HLT Trig names var | HLT Selection count 4| HLT Selection Length 4 | HLT Selection names var | L1 Trig Count 4| L1 TrigName len 4| L1 Trig Names var | adler32 chksum 4| desc legth 4 | description blob var

Protocol Version 11: identical to version 10, but incremented to keep in sync with event msg protocol version

*/

#ifndef IOPool_Streamer_InitMessage_h
#define IOPool_Streamer_InitMessage_h

#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/MsgHeader.h"

struct Version
{
  Version(const uint8* pset):protocol_(11)
  { std::copy(pset,pset+sizeof(pset_id_),&pset_id_[0]); }

  uint8 protocol_; // version of the protocol
  unsigned char pset_id_[16]; // parameter set ID
};

struct InitHeader
{
  InitHeader(const Header& h, uint32 run, const Version& v,
           uint32 init_header_size=0, uint32 event_header_size=0):
    header_(h),version_(v)
  {
   convert(run,run_); 
   convert(init_header_size, init_header_size_);
   convert(event_header_size, event_header_size_);
  }

  Header header_;
  Version version_;
  char_uint32 run_;
  char_uint32 init_header_size_;
  char_uint32 event_header_size_;
};

class InitMsgView
{
public:

  InitMsgView(void* buf);

  uint32 code() const { return head_.code(); }
  uint32 size() const { return head_.size(); }
  uint8* startAddress() const { return buf_; }

  uint32 run() const;
  uint32 protocolVersion() const;
  void pset(uint8* put_here) const;
  std::string releaseTag() const;
  std::string processName() const;
  std::string outputModuleLabel() const;
  uint32 outputModuleId() const { return outputModuleId_; }

  void hltTriggerNames(Strings& save_here) const;
  void hltTriggerSelections(Strings& save_here) const;
  void l1TriggerNames(Strings& save_here) const;

  uint32 get_hlt_bit_cnt() const { return hlt_trig_count_; }
  uint32 get_l1_bit_cnt() const { return l1_trig_count_; }

  // needed for streamer file
  uint32 descLength() const { return desc_len_; }
  const uint8* descData() const { return desc_start_; }
  uint32 headerSize() const {return desc_start_-buf_;}
  uint32 eventHeaderSize() const;
  uint32 adler32_chksum() const {return adler32_chksum_;}
  std::string hostName() const;
  uint32 hostName_len() const {return host_name_len_;}

private:
  uint8* buf_;
  HeaderView head_;

  uint8* release_start_; // points to the string
  uint32 release_len_;

  uint8* processName_start_; // points to the string
  uint32 processName_len_;

  uint8* outputModuleLabel_start_; // points to the string
  uint32 outputModuleLabel_len_;
  uint32 outputModuleId_;

  uint8* hlt_trig_start_; // points to the string
  uint32 hlt_trig_count_; // number of strings
  uint32 hlt_trig_len_; // length of strings character array only
  uint8* hlt_select_start_; // points to the string
  uint32 hlt_select_count_; // number of strings
  uint32 hlt_select_len_; // length of strings character array only
  uint8* l1_trig_start_; // points to the string
  uint32 l1_trig_count_; // number of strings
  uint32 l1_trig_len_; // length of strings character array only
  uint32 adler32_chksum_;
  uint8* host_name_start_;
  uint32 host_name_len_;

  // does not need to be present in the message sent over the network,
  // but is needed for the index file
  uint8* desc_start_; // point to the bytes
  uint32 desc_len_;
};

#endif

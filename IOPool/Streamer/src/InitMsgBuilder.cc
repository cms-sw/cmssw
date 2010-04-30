#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "IOPool/Streamer/interface/MsgHeader.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include <cassert>
#include <cstring>
#include <unistd.h>

#define MAX_INITHOSTNAME_LEN 25

InitMsgBuilder::InitMsgBuilder(void* buf, uint32 size,
                               uint32 run, const Version& v,
                               const char* release_tag,
                               const char* process_name,		       
                               const char* output_module_label,
                               uint32 output_module_id,
                               const Strings& hlt_names,
                               const Strings& hlt_selections,
                               const Strings& l1_names,
                               uint32 adler_chksum, const char* host_name):
  buf_((uint8*)buf),size_(size)
{
  InitHeader* h = (InitHeader*)buf_;
  // fixed length parts
  new (&h->version_) Version(v);
  convert(run,h->run_);
  // variable length parts
  uint32 tag_len = strlen(release_tag);
  assert(tag_len < 0x00ff);
  uint8* pos = buf_+sizeof(InitHeader);

  *pos++ = tag_len; // length of release tag
  memcpy(pos,release_tag,tag_len); // copy release tag in
  pos += tag_len;

  //Lets put Process Name (Length and then Name) right after release_tag
  uint32 process_name_len = strlen(process_name);
  assert(process_name_len < 0x00ff);
  //Put process_name_len
  *pos++ = process_name_len;
  //Put process_name
  memcpy(pos,process_name,process_name_len);
  pos += process_name_len;

  // output module label next
  uint32 outmod_label_len = strlen(output_module_label);
  assert(outmod_label_len < 0x00ff);
  *pos++ = outmod_label_len;
  memcpy(pos,output_module_label,outmod_label_len);
  pos += outmod_label_len;

  // output module ID next
  convert(output_module_id, pos);
  pos += sizeof(char_uint32);

  pos = MsgTools::fillNames(hlt_names,pos);
  pos = MsgTools::fillNames(hlt_selections,pos);
  pos = MsgTools::fillNames(l1_names,pos);

  // adler32 check sum of data blob
  convert(adler_chksum, pos);
  pos = pos + sizeof(uint32);

  // put host name (Length and then Name) right after check sum
  //uint32 host_name_len = strlen(host_name);
  // actually make the host_name a fixed length as the init message header size appears in the
  // Init message and only one goes to a file whereas events can come from any node
  // We want the max length to be determined inside this Init Message Builder
  uint32 host_name_len = MAX_INITHOSTNAME_LEN;
  assert(host_name_len < 0x00ff);
  //Put host_name_len
  *pos++ = host_name_len;

  //Put host_name
  uint32 real_len = strlen(host_name);
  if(real_len < host_name_len) {
    char hostname_2use[MAX_INITHOSTNAME_LEN];
    memset(hostname_2use,'\0',host_name_len);
    memcpy(hostname_2use,host_name,real_len);
    memcpy(pos,hostname_2use,host_name_len);
  } else {
    memcpy(pos,host_name,host_name_len);
  }
  pos += host_name_len;

  data_addr_ = pos + sizeof(char_uint32);
  setDataLength(0);

  // Two news fileds added to InitMsg in Proto V3 init_header_size, and event_header_size.
  //Set the size of Init Header Start of buf to Start of desc.
  convert((uint32)(data_addr_ - buf_), h->init_header_size_);

  // 18-Apr-2008, KAB:  create a dummy event message so that we can
  // determine the expected event header size.  (Previously, the event
  // header size was hard-coded.)
  std::vector<bool> dummyL1Bits(l1_names.size());
  std::vector<char> dummyHLTBits(hlt_names.size());
  const uint32 TEMP_BUFFER_SIZE = 256;
  char msgBuff[TEMP_BUFFER_SIZE];  // not large enough for a real event!
  uint32_t adler32 = 0;
  //char host_name[255];
  //int got_host = gethostname(host_name, 255);
  //if(got_host != 0) strcpy(host_name, "noHostNameFoundOrTooLong");
  EventMsgBuilder dummyMsg(&msgBuff[0], TEMP_BUFFER_SIZE, 0, 0, 0, 0,
                           dummyL1Bits, (uint8*) &dummyHLTBits[0],
                           hlt_names.size(), adler32, host_name);

  //Size of Event Header
  uint32 eventHeaderSize = dummyMsg.headerSize();
  convert(eventHeaderSize, h->event_header_size_);
}

void InitMsgBuilder::setDataLength(uint32 len)
{
  convert(len,data_addr_-sizeof(char_uint32));
  InitHeader* h = (InitHeader*)buf_;
  new (&h->header_) Header(Header::INIT, data_addr_ - buf_ + len);
}


uint32 InitMsgBuilder::size() const
{

  HeaderView v(buf_);
  return v.size();
}


uint32 InitMsgBuilder::run() const
{
  InitHeader* h = (InitHeader*)buf_;
  return convert32(h->run_);
}


#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/MsgHeader.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include <cassert>

InitMsgBuilder::InitMsgBuilder(void* buf, uint32 size,
                               uint32 run, const Version& v,
                               const char* release_tag,
                               const char* process_name,		       
                               const char* output_module_label,
                               const Strings& hlt_names,
                               const Strings& hlt_selections,
                               const Strings& l1_names):
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

  pos = MsgTools::fillNames(hlt_names,pos);
  pos = MsgTools::fillNames(hlt_selections,pos);
  pos = MsgTools::fillNames(l1_names,pos);

  desc_addr_ = pos + sizeof(char_uint32);
  setDescLength(0);

  // Two news fileds added to InitMsg in Proto V3 init_header_size, and event_header_size.
  //Set the size of Init Header Start of buf to Start of desc.
  convert((uint32)(desc_addr_-buf_), h->init_header_size_);

  uint32 hlt_sz = hlt_names.size();
  if (hlt_sz != 0) hlt_sz = 1+ ((hlt_sz-1)/4);

  uint32 l1_sz = l1_names.size();
  if (l1_sz != 0) l1_sz = 1 + ((l1_sz-1)/8);

  //Size of Event Header
  uint32 eventHeaderSize = 2 + (9*4) + hlt_sz + l1_sz;
  convert(eventHeaderSize, h->event_header_size_);
}

void InitMsgBuilder::setDescLength(uint32 len)
{
  convert(len,desc_addr_-sizeof(char_uint32));
  InitHeader* h = (InitHeader*)buf_;
  new (&h->header_) Header(Header::INIT,desc_addr_-buf_+len);
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


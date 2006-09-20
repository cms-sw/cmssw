#ifndef _InitMsgBuilder_h
#define _InitMsgBuilder_h

#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/InitMessage.h"

// ----------------- init -------------------

class InitMsgBuilder
{
public:
  InitMsgBuilder(void* msg_mem, uint32 size,
                 uint32 run, const Version& v,
                 const char* release_tag,
                 const Strings& hlt_names,
                 const Strings& l1_names);

  uint8* startAddress() const { return buf_; }
  void setDescLength(uint32 registry_length);
  uint8* dataAddress() const  { return desc_addr_; }
  uint32 headerSize() const {return desc_addr_-buf_;}
  uint32 size() const ;
  uint32 run() const;  /** Required by EOF Record Builder */  
  uint32 bufferSize() const {return size_;}

private:
  uint8* fillNames(const Strings& names, uint8* pos);
  
  uint8* buf_;
  uint32 size_;
  uint8* desc_addr_;
};

#endif


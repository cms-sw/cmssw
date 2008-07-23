#ifndef IOPool_Streamer_InitMsgBuilder_h
#define IOPool_Streamer_InitMsgBuilder_h

#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/InitMessage.h"

// ----------------- init -------------------

class InitMsgBuilder
{
public:
  InitMsgBuilder(void* msg_mem, uint32 size,
                 uint32 run, const Version& v,
                 const char* release_tag,
                 const char* process_name,
                 const char* output_module_label,
                 uint32 output_module_id,
                 const Strings& hlt_names,
                 const Strings& hlt_selections,
                 const Strings& l1_names);

  uint8* startAddress() const { return buf_; }
  void setDataLength(uint32 registry_length);
  uint8* dataAddress() const  { return data_addr_; }
  uint32 headerSize() const {return data_addr_-buf_;}
  uint32 size() const ;
  uint32 run() const;  /** Required by EOF Record Builder */  
  uint32 bufferSize() const {return size_;}

private:
  uint8* buf_;
  uint32 size_;
  uint8* data_addr_;
};

#endif


/** Other type of Messages Represented here

code 1 | size 4 | msg_body 4 

For now use can provide any code, in future we may restrict 0, 1 and 2.

*/

#ifndef IOPool_Streamer_OtherMessage_h
#define IOPool_Streamer_OtherMessage_h

#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/MsgHeader.h"

//------------------------------Builds the Message -------------------------

class OtherMessageBuilder
{
public:

  //Constructor to Create OtherMessage
  OtherMessageBuilder(void* buf, uint32 code, uint32 bodySize=0):
  buf_((uint8*)buf),
  h_((Header*)buf) 
   {
   new (h_) Header (code, (unsigned int)sizeof(Header)+bodySize);
   }

  uint32 code() const { return h_->code_; }
  uint32 size() const { return convert32(h_->size_); }
  uint8* msgBody()    { return buf_+sizeof(Header); }
  uint8* startAddress() { return buf_; }

private:
  uint8* buf_;
  Header* h_;
};

// ----------------------- Looks at the Message  ------------------------
 
class OtherMessageView 
{ 
public: 
 
 
  //Constructor to View OtherMessage 
  OtherMessageView(void* buf): 
  buf_((uint8*)buf), 
  head_((Header*)buf) 
  { 
   msg_body_start_ = buf_ + sizeof(Header); 
  } 
 
  uint32 code() const { return head_->code_; } 
  uint32 size() const { return convert32(head_->size_); } 
  uint8* msgBody() const {return msg_body_start_; } 
  uint8* startAddress() { return buf_; }
  uint32 bodySize() const {
    return convert32(head_->size_) - (msg_body_start_ - buf_);
  }
 
private: 
  uint8* buf_; 
  uint8* msg_body_start_; 
  Header* head_; 
}; 

#endif


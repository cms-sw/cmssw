/** Other type of Messages Represented here

code 1 | size 4 | msg_body 4 

For now use can provide any code, in future we may restrict 0, 1 and 2.

*/

#ifndef _OtherMessage_h
#define _OtherMessage_h

#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/MsgHeader.h"

//------------------------------Builds the Message -------------------------

class OtherMessageBuilder
{
public:

  //Constructor to Create OtherMessage
  OtherMessageBuilder(void* buf, uint32 code, uint32 msgBody=0):
  buf_((uint8*)buf)
   {
   Header* h = (Header*)buf_;
   h->code_ = code;
   convert((unsigned int)4, h->size_);  //Size if fixed for now to 4 (uint32 msgBody)
   //convert(code,h->code_);
   msgBody_ = msgBody; 
   }

  uint32 code() const { HeaderView* h = (HeaderView*)buf_; return h->code(); }
  uint32 size() const { HeaderView* h = (HeaderView*)buf_; return h->size(); }
  uint32 msgBody() const {return msgBody_; } 

private:
  uint8* buf_;
  uint32 msgBody_;
};

// ----------------------- Looks at the Message  ------------------------
 
class OtherMessageView 
{ 
public: 
 
 
  //Constructor to View OtherMessage 
  OtherMessageView(void* buf): 
  buf_((uint8*)buf), 
  head_(buf) 
  { 
   msg_body_start_ = buf_ + sizeof(HeaderView); 
   msgBody_ = convert32(msg_body_start_); 
  } 
 
  uint32 code() const { return head_.code(); } 
  uint32 size() const { return head_.size(); } 
  uint32 msgBody() const {return msgBody_; } 
 
private: 
  uint8* buf_; 
  uint8* msg_body_start_; 
 
  HeaderView head_; 
  uint32 msgBody_; 
}; 

#endif


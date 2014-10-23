#ifndef IOPool_Streamer_Messages_h
#define IOPool_Streamer_Messages_h

/*
  The header for each of the data buffer that will be transferred.
  The packing and unpacking of the data in the header is directly
  handled here.

  Every incoming message has a length and message code. Beyond that,
  each type has a specific header, followed by a specific 
  payload - the packed ROOT data buffer.
*/

#include <vector>
#include <cassert>
#include <iostream>

#include "DataFormats/Provenance/interface/EventID.h"

/*
  There is still a problem here - the message lengths are not
  well specified.  There is dataSize, msgSize, getDataSize, 
  and payloadSize calls.  This sounds like too many.

 There is still an inconsistency between the way that dataSize is treated
 in the EventMsg and in the InitMsg.  The EventMsg method is better.  The
 problem is that the base class (MsgCode) may have a zero size if it is
 referring to an incoming buffer.  This makes a call to payloadSize useless.
 */

namespace edm {

  inline unsigned int decodeInt(unsigned char* v)
  {
    // first four bytes are code,  LSB first
    unsigned int a=v[0], b=v[1], c=v[2], d=v[3];
    a|=(b<<8)|(c<<16)|(d<<24);
    return a;
  }

  inline void encodeInt(unsigned int i, unsigned char* v)
  {
    v[0]=i&0xff;
    v[1]=(i>>8)&0xff;
    v[2]=(i>>16)&0xff;
    v[3]=(i>>24)&0xff;
  }

  // -----------------------------------------------

  // we currently only support two message types:
  //  1 = SendJobHead (INIT)
  //  2 = SendEvent (EVENT)

  class MsgCode
  {
  public:
    enum Codes { INVALID = 0, INIT = 1, EVENT = 2, DONE = 3 };
    
    // header is the message code
    // the size kept in this object is the available space (i.e. -header)
    
    // the entire size of the buffer
    MsgCode(void* buffer, int size):
      buffer_((unsigned char*)buffer),size_(size<4?0:size-4)
    { assert(size_>=0); } 
    
    // unknown size
    explicit MsgCode(void* buffer):
      buffer_((unsigned char*)buffer),size_(0)
    { } 

    // with header message code set
    MsgCode(void* buffer, Codes c):
      buffer_((unsigned char*)buffer),size_(0)
    { setCode(c); }

    // with header message code set and a total buffer length
    MsgCode(void* buffer, int size, Codes c):
      buffer_((unsigned char*)buffer),size_(size-4)
    { setCode(c); }

    void setCode(Codes c)
    {
      encodeInt(c,buffer_);
    }

    Codes getCode() const
    {
      return (Codes)decodeInt(buffer_);
    }

    // adjust for header (header not included in payload address
    void* payload() const { return &buffer_[4]; }
    int payloadSize() const { return size_; }
    int codeSize() const { return 4; }
    int totalSize() const { return size_+4; }

  private:
    unsigned char* buffer_;
    int size_;

  };

  // -----------------------------------------------

  /*
    added two more fields - "the m out of n data".
    Each message will contain these fragment counts and the header
    data.

    Event Message Format:
    |MSG_ID|WHICH_SEG|TOTAL_SEGS|EVENT_ID|RUN_ID|DATA_SIZE| ...DATA... |
    |-code-|-------------------header --------------------|

    In order to know the original data, you must concatenate all the
    segments in the proper order.  The field WHICH_SEG which one of the
    TOTAL_SEGS this part of the data is.
   */

  class EventMsg : public MsgCode
  {
  public:
    struct EventMsgHeader
    {
      unsigned char which_seg_[4];
      unsigned char total_segs_[4];
      unsigned char event_num_[4];
      unsigned char run_num_[4];
      unsigned char data_size_[4];
    };

    // incoming data buffer to be looked at
    EventMsg(MsgCode& mc):
      MsgCode(mc),
      head_((EventMsgHeader*)MsgCode::payload()),
      which_seg_(getWhichSeg()),
      total_segs_(getTotalSegs()),
      event_num_(getEventNumber()),
      run_num_(getRunNumber()),
      data_size_(getDataSize())
    { }
    
    // incoming data buffer to be looked at
    explicit EventMsg(void* buffer, int size=0):
      MsgCode(buffer,size),
      head_((EventMsgHeader*)MsgCode::payload()),
      which_seg_(getWhichSeg()),
      total_segs_(getTotalSegs()),
      event_num_(getEventNumber()),
      run_num_(getRunNumber()),
      data_size_(getDataSize())
    { }

    // outgoing data buffer to be filled
    EventMsg(void* buffer, int size,
	     edm::EventNumber_t e,
	     edm::RunNumber_t r,
	     int which_seg,
	     int total_segs):
      MsgCode(buffer,size),
      head_((EventMsgHeader*)MsgCode::payload()),
      which_seg_(which_seg),
      total_segs_(total_segs),
      event_num_(e),
      run_num_(r),
      data_size_(payloadSize() - sizeof(EventMsgHeader))
    {
      setCode(MsgCode::EVENT);
      setWhichSeg(which_seg);
      setTotalSegs(total_segs);
      setEventNumber(e);
      setRunNumber(r);
      setDataSize(data_size_);
    }

    // the data is really a SendEvent
    void* data()   const { return (char*)payload() + sizeof(EventMsgHeader); } 
    // the size is max size here
    int dataSize() const { return data_size_ + sizeof(EventMsgHeader); }

    int getDataSize() const 
    {
      return decodeInt(head_->data_size_);
    }

    void setDataSize(int s)
    {
      encodeInt(s,head_->data_size_);
    }

    int getWhichSeg() const 
    {
      return decodeInt(head_->which_seg_);
    }

    void setWhichSeg(int s)
    {
      encodeInt(s,head_->which_seg_);
    }

    int getTotalSegs() const 
    {
      return decodeInt(head_->total_segs_);
    }

    void setTotalSegs(int s)
    {
      encodeInt(s,head_->total_segs_);
    }

    edm::EventNumber_t getEventNumber() const 
    {
      // assert(sizeof(edm::EventNumber_t) == sizeof(int) && "event ID streaming only knows how to work with 4 byte event ID numbers right now");
      return decodeInt(head_->event_num_);
    }

    void setEventNumber(edm::EventNumber_t e)
    {
      assert(sizeof(edm::RunNumber_t) == sizeof(int) && "run number streaming only knows how to work with 4 byte event ID numbers right now");
      encodeInt(e,head_->event_num_);
    }

    edm::RunNumber_t getRunNumber() const
    {
      // assert(sizeof(edm::EventNumber_t) == sizeof(int) && "event ID streaming only knows how to work with 4 byte event ID numbers right now");
      return decodeInt(head_->run_num_);
    }

    void setRunNumber(edm::RunNumber_t r)
    {
      assert(sizeof(edm::RunNumber_t) == sizeof(int) && "run number streaming only knows how to work with 4 byte event ID numbers right now");
      return encodeInt(r,head_->run_num_);
    }

    // the number of bytes used, including the headers
    int msgSize() const
    {
      return codeSize()+sizeof(EventMsgHeader)+getDataSize();
    }

  private:
    EventMsgHeader* head_;
    int which_seg_;
    int total_segs_;
    edm::EventNumber_t event_num_;
    edm::RunNumber_t run_num_;
    int data_size_;
  };
  
  // -------------------------------------------------
  /*
    Format:
    | MSG_ID | DATA_SIZE | ... DATA ... |
   */

  class InitMsg : public MsgCode
  {
  public:
    struct InitMsgHeader
    {
      unsigned char data_size_[4];
    };

    InitMsg(MsgCode& m):
      MsgCode(m),
      head_((InitMsgHeader*)MsgCode::payload()),
      data_size_(payloadSize() - sizeof(InitMsgHeader))
    { setDataSize(data_size_); }

    InitMsg(void* buffer, int size, bool setcode = false):
      MsgCode(buffer,size),
      head_((InitMsgHeader*)MsgCode::payload()),
      data_size_(payloadSize() - sizeof(InitMsgHeader)) // default to full length
    {
      if(setcode)
	{
	  // for new message
	  setCode(MsgCode::INIT);
	  setDataSize(data_size_);
	}
      else
	// for existing message
	data_size_ = getDataSize();
    }

    // the data is really a SendJobHeader
    // for this message, there is nothing manually encoded/decoded in the
    // header, it is all contained in the ROOT buffer.
    // currently we supply no extra data header
    void* data() const { return (char*)payload()+sizeof(InitMsgHeader); } 
    int dataSize() const { return payloadSize() + sizeof(InitMsgHeader); }

    int getDataSize() const 
    {
      return decodeInt(head_->data_size_);
    }

    void setDataSize(int s)
    {
      encodeInt(s,head_->data_size_);
    }

    // the number of bytes used, including the headers
    int msgSize() const
    {
      return codeSize()+sizeof(InitMsgHeader)+getDataSize();
    }

  private:
    InitMsgHeader* head_;
    int data_size_;
  };

}
#endif


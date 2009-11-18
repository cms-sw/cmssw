#ifndef EVENTFILTER_PROCESSOR_MSG_BUF_H
#define EVENTFILTER_PROCESSOR_MSG_BUF_H

#include "queue_defs.h"

namespace evf{
  class MsgBuf{
  public:
    MsgBuf() : msize_(MAX_MSG_SIZE+sizeof(struct msgbuf)+1)
      {
	buf = new unsigned char[MAX_MSG_SIZE+sizeof(struct msgbuf)+1];
	ptr_ = (msgbuf*)buf;
	ptr_->mtype = MSQM_MESSAGE_TYPE_NOP;
      }

    MsgBuf(unsigned int size, unsigned int type) : msize_(size)
      {
	buf = new unsigned char[msize_+sizeof(struct msgbuf)+1];
	ptr_ = (msgbuf*)buf;
	ptr_->mtype = type;
      }
    size_t msize(){return msize_;}
    virtual ~MsgBuf(){delete[] buf;}
    msgbuf* operator->(){return ptr_;}
  private:
    struct msgbuf *ptr_;
    unsigned char *buf;
    size_t msize_;
    friend class MasterQueue;
    friend class SlaveQueue;
  };
}
#endif

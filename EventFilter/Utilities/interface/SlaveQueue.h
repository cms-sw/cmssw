#ifndef EVENTFILTER_UTILITIES_SLAVEQUEUE_H
#define EVENTFILTER_UTILITIES_SLAVEQUEUE_H

#include <stdio.h>       /* standard I/O functions.              */
#include <stdlib.h>      /* malloc(), free() etc.                */
#include <sys/types.h>   /* various type definitions.            */
#include <sys/ipc.h>     /* general SysV IPC structures          */
#include <sys/msg.h>     /* message queue functions and structs. */
#include <errno.h>
#include <string.h>

#include "EventFilter/Utilities/interface/MsgBuf.h"

//@EM ToDo move implementation to .cc file

namespace evf{

  class SlaveQueue{

  public:

    SlaveQueue(unsigned int ind) : queue_id_(0)
      {
	
	/* get an (existing) public message queue */
	queue_id_ = msgget(QUEUE_ID+ind, 0);
	if (queue_id_ == -1) {
	  XCEPT_RAISE(evf::Exception, "failed to get message queue");
	}
      }
    ~SlaveQueue()
      {
      }

    int post(MsgBuf &ptr)
      {
	int rc;                  /* error code retuend by system calls. */
	rc = msgsnd(queue_id_,ptr.ptr_, ptr.msize(),0);
	//	delete ptr;
	if(rc==-1)
	  std::cout << "snd::Slave failed to post message - error:"
		    << strerror(errno) << std::endl;
	return rc;
      }
    unsigned long rcv(MsgBuf &ptr)
      {
	unsigned long msg_type = MSQS_MESSAGE_TYPE_SLA;
	int rc = msgrcv(queue_id_, ptr.ptr_, ptr.msize(), - msg_type, 0);
	if (rc == -1 && errno != ENOMSG) 
	  {
	    std::string serr = "rcv::Slave failed to get message from queue - error:";
	    serr += strerror(errno);
	    XCEPT_RAISE(evf::Exception, serr);
	  }
	else if(rc == -1 && errno == ENOMSG) return MSGQ_MESSAGE_TYPE_RANGE;
	return msg_type;
      }
    unsigned long rcvNonBlocking(MsgBuf &ptr)
      {
	unsigned long msg_type = MSQS_MESSAGE_TYPE_SLA;
	int rc = msgrcv(queue_id_, ptr.ptr_, ptr.msize(), - msg_type, IPC_NOWAIT);
	if (rc == -1 && errno != ENOMSG) 
	  {
	    std::string serr = "rcvnb::Slave failed to get message from queue - error:";
	    serr += strerror(errno);
	    XCEPT_RAISE(evf::Exception, serr);
	  }
	else if(rc == -1 && errno == ENOMSG) return MSGQ_MESSAGE_TYPE_RANGE;
	return msg_type;
      }
    unsigned long rcvNonBlockingAny(MsgBuf &ptr)
      {
	unsigned long msg_type = 0;
	int rc = msgrcv(queue_id_, ptr.ptr_, ptr.msize(), msg_type, IPC_NOWAIT);
	if (rc == -1 && errno != ENOMSG) 
	  {
	    std::string serr = "rcvnb::Slave failed to get message from queue - error:";
	    serr += strerror(errno);
	    XCEPT_RAISE(evf::Exception, serr);
	  }
	else if(rc == -1 && errno == ENOMSG) return MSGQ_MESSAGE_TYPE_RANGE;
	return msg_type;
      }
    int id() const {return queue_id_;}
  private:

    int queue_id_;             /* ID of the created queue.            */

  };
}
#endif

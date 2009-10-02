#ifndef EVENTFILTER_PROCESSOR_SLAVEQUEUE_H
#define EVENTFILTER_PROCESSOR_SLAVEQUEUE_H

#include <stdio.h>       /* standard I/O functions.              */
#include <stdlib.h>      /* malloc(), free() etc.                */
#include <sys/types.h>   /* various type definitions.            */
#include <sys/ipc.h>     /* general SysV IPC structures          */
#include <sys/msg.h>     /* message queue functions and structs. */

#include "MsgBuf.h"

namespace evf{

  class SlaveQueue{

  public:

    SlaveQueue(unsigned int ind) : queue_id_(0)
      {
	
	/* create a public message queue, with access only to the owning user. */
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
	rc = msgsnd(queue_id_,ptr.ptr_, ptr.msize()+1,0);
	//	delete ptr;
	return rc;
      }
    unsigned long rcv(MsgBuf &ptr)
      {
	unsigned long msg_type = MSQS_MESSAGE_TYPE_SLA;
	int rc = msgrcv(queue_id_, ptr.ptr_, ptr.msize()+1, - msg_type, 0);
	if (rc == -1) 
	  {
	    XCEPT_RAISE(evf::Exception, "Slave failed to get message from queue");
	  }
	return msg_type;
      }
    unsigned long rcvNonBlocking(MsgBuf &ptr)
      {
	unsigned long msg_type = MSQS_MESSAGE_TYPE_SLA;
	int rc = msgrcv(queue_id_, ptr.ptr_, ptr.msize()+1, - msg_type, IPC_NOWAIT);
	if (rc == -1) 
	  {
	    XCEPT_RAISE(evf::Exception, "Slave failed to get message from queue");
	  }
	return msg_type;
      }
  private:

    int queue_id_;             /* ID of the created queue.            */

  };
}
#endif

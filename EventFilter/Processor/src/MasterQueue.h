#ifndef EVENTFILTER_PROCESSOR_MASTERQUEUE_H
#define EVENTFILTER_PROCESSOR_MASTERQUEUE_H

#include <stdio.h>       /* standard I/O functions.              */
#include <stdlib.h>      /* malloc(), free() etc.                */
#include <sys/types.h>   /* various type definitions.            */
#include <sys/ipc.h>     /* general SysV IPC structures          */
#include <sys/msg.h>     /* message queue functions and structs. */

#include "MsgBuf.h"

namespace evf{

  class MasterQueue{

  public:

    MasterQueue(unsigned int ind)
      {
	
	/* create a public message queue, with access only to the owning user. */
	queue_id_ = msgget(QUEUE_ID+ind, IPC_CREAT | IPC_EXCL | 0600);
	if (queue_id_ == -1) {
	  XCEPT_RAISE(evf::Exception, "failed to get message queue");
	}
	status_ = 1;
	occup_ = 0;
      }
    ~MasterQueue()
      {
	if(status_>0) msgctl(queue_id_,IPC_RMID,0);
      }

    int post(MsgBuf &ptr)
      {
	int rc;                  /* error code returned by system calls. */
	rc = msgsnd(queue_id_, ptr.ptr_, ptr.msize()+1,0);
	//	delete ptr;
	return rc;
      }
    unsigned long rcv(MsgBuf &ptr)
      {
	unsigned long msg_type = ptr->mtype;
	int rc = msgrcv(queue_id_, ptr.ptr_, ptr.msize()+1, ptr->mtype, 0);
	if (rc == -1) 
	  {
	    XCEPT_RAISE(evf::Exception, "Master failed to get message from queue");
	  }
	return msg_type;
      }
    unsigned long rcvNonBlocking(MsgBuf &ptr)
      {
	unsigned long msg_type = ptr->mtype;
	int rc = msgrcv(queue_id_, ptr.ptr_, ptr.msize()+1, msg_type, IPC_NOWAIT);
	if (rc == -1) 
	  {
	    XCEPT_RAISE(evf::Exception, "Master failed to get message from queue");
	  }
	return msg_type;
      }
    int disconnect()
      {
	int ret = msgctl(queue_id_,IPC_RMID,0);
	status_ = -1000;
	return ret;
      }
    int id(){return queue_id_;}
    int status()
      {
	char cbuf[sizeof(struct msqid_ds)];
	struct msqid_ds *buf= (struct msqid_ds*)cbuf;
	int ret = msgctl(queue_id_,IPC_STAT,buf);	
	if(ret!=0) status_ = -1;
	else
	  {
	    occup_ = buf->msg_qnum;
	    pidOfLastSend_ = buf->msg_lspid;
	    pidOfLastReceive_ = buf->msg_lrpid;
	    std::cout << "queue " << buf->msg_qnum << " " 
		      << buf->msg_lspid << " " 
		      << buf->msg_lrpid << std::endl;
	  }
	return status_;
      }
    int occupancy()const{return occup_;}
    pid_t pidOfLastSend()const{return pidOfLastSend_;}
    pid_t pidOfLastReceive()const{return pidOfLastReceive_;}
  private:

    int queue_id_;             /* ID of the created queue.            */
    int status_;
    int occup_;
    int pidOfLastSend_;
    int pidOfLastReceive_;
  };
}
#endif

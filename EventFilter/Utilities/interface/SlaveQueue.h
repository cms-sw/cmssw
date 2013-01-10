#ifndef EVENTFILTER_UTILITIES_SLAVEQUEUE_H
#define EVENTFILTER_UTILITIES_SLAVEQUEUE_H

#include <stdio.h>       /* standard I/O functions.              */
#include <stdlib.h>      /* malloc(), free() etc.                */
#include <sys/types.h>   /* various type definitions.            */
#include <sys/ipc.h>     /* general SysV IPC structures          */
#include <sys/msg.h>     /* message queue functions and structs. */
#include <errno.h>
#include <string.h>

#include "EventFilter/Utilities/interface/Exception.h"
#include "EventFilter/Utilities/interface/MsgBuf.h"

namespace evf {

class SlaveQueue {

public:

	SlaveQueue(unsigned int ind);
	~SlaveQueue();

	int post(MsgBuf &ptr);
	unsigned long rcv(MsgBuf &ptr);
	bool rcvQuiet(MsgBuf &ptr);
	unsigned long rcvNonBlocking(MsgBuf &ptr);
	unsigned long rcvNonBlockingAny(MsgBuf &ptr);
	int id() const;

private:

	int queue_id_; /* ID of the created queue.            */

};
}
#endif

#ifndef EVENTFILTER_UTILITIES_MASTERQUEUE_H
#define EVENTFILTER_UTILITIES_MASTERQUEUE_H

#include <stdio.h>       /* standard I/O functions.              */
#include <stdlib.h>      /* malloc(), free() etc.                */
#include <sys/types.h>   /* various type definitions.            */
#include <sys/ipc.h>     /* general SysV IPC structures          */
#include <sys/msg.h>     /* message queue functions and structs. */
#include <errno.h>
#include <string.h>

#include <iostream>
#include <sstream>
#include <vector>

#include "EventFilter/Utilities/interface/Exception.h"
#include "EventFilter/Utilities/interface/MsgBuf.h"

namespace evf {

class MasterQueue {

public:

	MasterQueue(unsigned int ind);
	~MasterQueue();

	int post(MsgBuf &ptr);
	int postLength(MsgBuf &ptr, unsigned int length);
	unsigned long rcv(MsgBuf &ptr);
	bool rcvQuiet(MsgBuf &ptr);
	unsigned long rcvNonBlocking(MsgBuf &ptr);
	int disconnect();
	int id();
	int status();
	int occupancy() const;
	void drain();
	pid_t pidOfLastSend() const;
	pid_t pidOfLastReceive() const;

	std::vector<int> getReceivers() const { return receivers_; }

private:

	void updateReceivers();

private:

	int queue_id_; /* ID of the created queue.            */
	int status_;
	int occup_;
	int pidOfLastSend_;
	int pidOfLastReceive_;
	std::vector<int> receivers_;
};
}
#endif

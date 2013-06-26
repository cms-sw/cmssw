#include "EventFilter/Utilities/interface/SlaveQueue.h"

using namespace evf;

SlaveQueue::SlaveQueue(unsigned int ind) :
	queue_id_(0) {

	/* get an (existing) public message queue */
	queue_id_ = msgget(QUEUE_ID + ind, 0);
	if (queue_id_ == -1) {
		XCEPT_RAISE(evf::Exception, "failed to get message queue");
	}
}

SlaveQueue::~SlaveQueue() {
}

int SlaveQueue::post(MsgBuf &ptr) {
	int rc; /* error code retuend by system calls. */
	rc = msgsnd(queue_id_, ptr.ptr_, ptr.msize() + 1, 0);
	//	delete ptr;
	if (rc == -1)
		std::cout << "snd::Slave failed to post message - error:" << strerror(
				errno) << std::endl;
	return rc;
}

unsigned long SlaveQueue::rcv(MsgBuf &ptr) {
	unsigned long msg_type = MSQS_MESSAGE_TYPE_SLA;
	int rc = msgrcv(queue_id_, ptr.ptr_, ptr.msize() + 1, -msg_type, 0);
	if (rc == -1 && errno != ENOMSG) {
		std::string serr =
				"rcv::Slave failed to get message from queue - error:";
		serr += strerror(errno);
		XCEPT_RAISE(evf::Exception, serr);
	} else if (rc == -1 && errno == ENOMSG)
		return MSGQ_MESSAGE_TYPE_RANGE;
	return msg_type;
}

bool SlaveQueue::rcvQuiet(MsgBuf &ptr) {
	int rc = msgrcv(queue_id_, ptr.ptr_, ptr.msize() + 1, ptr->mtype, 0);
	if (rc == -1 && errno != ENOMSG) {
		return false;
	}
	return true;
}

unsigned long SlaveQueue::rcvNonBlocking(MsgBuf &ptr) {
	unsigned long msg_type = MSQS_MESSAGE_TYPE_SLA;
	int rc =
			msgrcv(queue_id_, ptr.ptr_, ptr.msize() + 1, -msg_type, IPC_NOWAIT);
	if (rc == -1 && errno != ENOMSG) {
		std::string serr =
				"rcvnb::Slave failed to get message from queue - error:";
		serr += strerror(errno);
		XCEPT_RAISE(evf::Exception, serr);
	} else if (rc == -1 && errno == ENOMSG)
		return MSGQ_MESSAGE_TYPE_RANGE;
	return msg_type;
}

unsigned long SlaveQueue::rcvNonBlockingAny(MsgBuf &ptr) {
	unsigned long msg_type = 0;
	int rc = msgrcv(queue_id_, ptr.ptr_, ptr.msize() + 1, msg_type, IPC_NOWAIT);
	if (rc == -1 && errno != ENOMSG) {
		std::string serr =
				"rcvnb::Slave failed to get message from queue - error:";
		serr += strerror(errno);
		XCEPT_RAISE(evf::Exception, serr);
	} else if (rc == -1 && errno == ENOMSG)
		return MSGQ_MESSAGE_TYPE_RANGE;
	return msg_type;
}

int SlaveQueue::id() const {
	return queue_id_;
}


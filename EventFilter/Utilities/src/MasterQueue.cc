#include "EventFilter/Utilities/interface/MasterQueue.h"
//todo remove
#include <iostream>

using namespace evf;

MasterQueue::MasterQueue(unsigned int ind) :
	status_(0), occup_(0), pidOfLastSend_(0), pidOfLastReceive_(0) {

	/* create or attach a public message queue, with read/write access to every user. */
	queue_id_ = msgget(QUEUE_ID + ind, IPC_CREAT | 0666);
	if (queue_id_ == -1) {
		std::ostringstream ost;
		ost << "failed to get message queue:" << strerror(errno);
		XCEPT_RAISE(evf::Exception, ost.str());
	}
	// it may be necessary to drain the queue here if it already exists !!!
	drain();
}

MasterQueue::~MasterQueue() {
	if (status_ > 0)
		msgctl(queue_id_, IPC_RMID, 0);
}

int MasterQueue::post(MsgBuf &ptr) {
	int rc; /* error code returned by system calls. */
	rc = msgsnd(queue_id_, ptr.ptr_, ptr.msize() + 1, 0);
	if (rc == -1)
		std::cout << "snd::Master failed to post message - error:" << strerror(
				errno) << std::endl;
	//	delete ptr;
	return rc;
}

int MasterQueue::postLength(MsgBuf &ptr, unsigned int length) {
	int rc; /* error code returned by system calls. */
	rc = msgsnd(queue_id_, ptr.ptr_, length, 0);
	if (rc == -1)
		std::cout << "snd::Master failed to post message - error:" << strerror(
				errno) << std::endl;
	//	delete ptr;
	return rc;
}

/*
 int MasterQueue::postOnlyUsefulData(SimpleMsgBuf &ptr) {
 int rc;
 rc = msgsnd(queue_id_, ptr.ptr_, ptr.usedSize_ , 0);
 if (rc == -1)
 std::cout << "snd::Master failed to post message - error:" << strerror(
 errno) << std::endl;
 //	delete ptr;
 return rc;
 }
 */

unsigned long MasterQueue::rcv(MsgBuf &ptr) {
	unsigned long msg_type = ptr->mtype;
	int rc = msgrcv(queue_id_, ptr.ptr_, ptr.msize() + 1, ptr->mtype, 0);
	if (rc == -1 && errno != ENOMSG) {
		std::string serr =
				"rcv::Master failed to get message from queue - error:";
		serr += strerror(errno);
		XCEPT_RAISE(evf::Exception, serr);
	} else if (rc == -1 && errno == ENOMSG)
		return MSGQ_MESSAGE_TYPE_RANGE;

	//updateReceivers();

	return msg_type;
}

bool MasterQueue::rcvQuiet(MsgBuf &ptr) {
	int rc = msgrcv(queue_id_, ptr.ptr_, ptr.msize() + 1, ptr->mtype, 0);
	if (rc == -1 && errno != ENOMSG) {
		return false;
	}
	return true;
}

unsigned long MasterQueue::rcvNonBlocking(MsgBuf &ptr) {
	unsigned long msg_type = ptr->mtype;
	int rc = msgrcv(queue_id_, ptr.ptr_, ptr.msize() + 1, msg_type, IPC_NOWAIT);
	if (rc == -1 && errno != ENOMSG) {
		std::string serr =
				"rcvnb::Master failed to get message from queue - error:";
		serr += strerror(errno);
		XCEPT_RAISE(evf::Exception, serr);
	} else if (rc == -1 && errno == ENOMSG)
		return MSGQ_MESSAGE_TYPE_RANGE;
	return msg_type;
}

int MasterQueue::disconnect() {
	int ret = msgctl(queue_id_, IPC_RMID, 0);
	status_ = -1000;
	return ret;
}

int MasterQueue::id() {
	return queue_id_;
}

int MasterQueue::status() {
	char cbuf[sizeof(struct msqid_ds)];
	struct msqid_ds *buf = (struct msqid_ds*) cbuf;
	int ret = msgctl(queue_id_, IPC_STAT, buf);
	if (ret != 0)
		status_ = -1;
	else {
		status_ = 1;
		occup_ = buf->msg_qnum;
		pidOfLastSend_ = buf->msg_lspid;
		pidOfLastReceive_ = buf->msg_lrpid;
		//	    std::cout << "queue " << buf->msg_qnum << " "
		//	      << buf->msg_lspid << " "
		//	      << buf->msg_lrpid << std::endl;
	}
	return status_;
}

int MasterQueue::occupancy() const {
	return occup_;
}

void MasterQueue::drain() {
	status();
	if (occup_ > 0)
		std::cout << "message queue id " << queue_id_ << " contains " << occup_
				<< "leftover messages, going to drain " << std::endl;
	//drain the queue before using it
	MsgBuf msg;
	while (occup_ > 0) {
		msgrcv(queue_id_, msg.ptr_, msg.msize() + 1, 0, 0);
		status();
		std::cout << "drained one message, occupancy now " << occup_
				<< std::endl;
	}
}

pid_t MasterQueue::pidOfLastSend() const {
	return pidOfLastSend_;
}

pid_t MasterQueue::pidOfLastReceive() const {
	return pidOfLastReceive_;
}

void MasterQueue::updateReceivers() {
	//update status
	status();
	int lastReceiver = pidOfLastReceive_;
	if (lastReceiver == 0)
		return;
	for (unsigned int i = 0; i < receivers_.size(); ++i)
		if (receivers_[i] == lastReceiver)
			return;
	receivers_.push_back(lastReceiver);
}

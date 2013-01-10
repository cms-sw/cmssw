#ifndef EVENTFILTER_UTILITIES_MSG_BUF_H
#define EVENTFILTER_UTILITIES_MSG_BUF_H

#include "EventFilter/Utilities/interface/queue_defs.h"
#include <string.h>

namespace evf {

class MsgBuf {

public:
	MsgBuf();
	MsgBuf(unsigned int size, unsigned int type);
	MsgBuf(const MsgBuf &b);

	MsgBuf &operator=(const MsgBuf &);
	size_t msize();
	virtual ~MsgBuf();
	msgbuf* operator->();

protected:
	size_t msize_;
	unsigned char *buf_;
	struct msgbuf *ptr_;
	friend class MasterQueue;
	friend class SlaveQueue;
};

}

#endif

/*
 * Message Buffer for Message Queue
 *  - holds an entire FUShmRawCell to transport
 *
 *  Author: aspataru : aspataru@cern.ch
 */

#ifndef EVENTFILTER_RB_RAW_MSG_BUF_H
#define EVENTFILTER_RB_RAW_MSG_BUF_H

#include "EventFilter/Utilities/interface/MsgBuf.h"
#include "EventFilter/ShmBuffer/interface/FUShmRawCell.h"

namespace evf {

/**
 * Message Buffer for Message Queue, holds an entire FUShmRawCell to transport.
 *
 * $Author: aspataru $
 *
 */

class RawMsgBuf: public MsgBuf {

public:
	RawMsgBuf(unsigned int size, unsigned int type);
	virtual ~RawMsgBuf();
	/**
	 * Construct the Raw cell object in-place in the message buffer
	 */
	void initialise(unsigned int rawCellSize);
	/**
	 * Returns a pointer to the Raw cell contained in the message buffer
	 */
	FUShmRawCell* rawCell() {
		return (FUShmRawCell*) ptr_->mtext;
	}
	/**
	 * Returns the actually used size in bytes of the buffer.
	 */
	inline unsigned int usedSize() {
		return theRawCell_->eventSize() + sizeof(long int);
	}

private:
	FUShmRawCell* theRawCell_;

	friend class MasterQueue;
};

}

#endif

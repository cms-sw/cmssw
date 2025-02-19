/*
 * Message Buffer for Message Queue
 *  - holds an entire FUShmDqmCell to transport
 *
 *  Author: aspataru : aspataru@cern.ch
 */

#ifndef EVENTFILTER_RB_DQM_MSG_BUF_H
#define EVENTFILTER_RB_DQM_MSG_BUF_H

#include "EventFilter/Utilities/interface/MsgBuf.h"
#include "EventFilter/ShmBuffer/interface/FUShmDqmCell.h"

namespace evf {

/**
 * Message buffer containing a DQM Cell.
 *
 * $Author: aspataru $
 *
 */

class DQMMsgBuf: public MsgBuf {

public:
	DQMMsgBuf(unsigned int size, unsigned int type);
	virtual ~DQMMsgBuf();

	/**
	 * Construct the DQM cell object in-place in the message buffer
	 */
	void initialise(unsigned int dqmCellSize);

	/**
	 * Returns a pointer to the DQM cell contained in the message buffer
	 */
	FUShmDqmCell* dqmCell() {
		return (FUShmDqmCell*) ptr_->mtext;
	}

private:
	FUShmDqmCell* theDqmCell_;
	friend class MasterQueue;
};

}

#endif

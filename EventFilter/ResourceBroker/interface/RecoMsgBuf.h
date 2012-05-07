/*
 * Message Buffer for Message Queue
 *  - holds an entire FUShmRecoCell to transport
 *
 *  Author: aspataru : aspataru@cern.ch
 */

#ifndef EVENTFILTER_RB_RECO_MSG_BUF_H
#define EVENTFILTER_RB_RECO_MSG_BUF_H

#include "EventFilter/Utilities/interface/MsgBuf.h"
#include "EventFilter/ShmBuffer/interface/FUShmRecoCell.h"

namespace evf {

/**
 * Message Buffer for Message Queue, holds an entire FUShmRecoCell to transport.
 *
 * $Author: aspataru $
 *
 */

class RecoMsgBuf: public MsgBuf {

public:
	RecoMsgBuf(unsigned int size, unsigned int type);
	virtual ~RecoMsgBuf();
	/**
	 * Construct the Reco cell object in-place in the message buffer
	 */
	void initialise(unsigned int recoCellSize);
	/**
	 * Returns a pointer to the Raw cell contained in the message buffer
	 */
	FUShmRecoCell* recoCell() {
		return (FUShmRecoCell*) ptr_->mtext;
	}

private:
	FUShmRecoCell* theRecoCell_;

	friend class MasterQueue;
};

}

#endif

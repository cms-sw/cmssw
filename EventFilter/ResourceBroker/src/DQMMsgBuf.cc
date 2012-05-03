/*
 * Message Buffer for Message Queue
 *  - holds an entire FUShmDqmCell to transport
 *
 *  Author: aspataru : aspataru@cern.ch
 */

#include "EventFilter/ResourceBroker/interface/DQMMsgBuf.h"
#include <iostream>
#include <cstdlib>

using namespace evf;

DQMMsgBuf::DQMMsgBuf(unsigned int size, unsigned int type) :
	MsgBuf(size, type), theDqmCell_(0) {
}

DQMMsgBuf::~DQMMsgBuf() {
	theDqmCell_->~FUShmDqmCell();
	//delete[] buf_;
}

void DQMMsgBuf::initialise(unsigned int dqmCellSize) {
	// construct the FUShmDqmCell at the buffer address
	// the memory is allocated in the super constructor
	theDqmCell_ = new (ptr_->mtext) FUShmDqmCell(dqmCellSize);
}

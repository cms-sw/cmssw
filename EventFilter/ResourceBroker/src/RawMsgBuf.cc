#include "EventFilter/ResourceBroker/interface/RawMsgBuf.h"
#include <iostream>
#include <cstdlib>

using namespace evf;

// create a raw message with the maximum message size
RawMsgBuf::RawMsgBuf(unsigned int size, unsigned int type) :
	MsgBuf(size, type), theRawCell_(0) {

}

RawMsgBuf::~RawMsgBuf() {
	theRawCell_->~FUShmRawCell();
	//delete[] buf_;
}

void RawMsgBuf::initialise(unsigned int rawCellSize) {
	// construct the FUShmRawCell at the buffer address
	// the memory is allocated in the super constructor
	theRawCell_ = new (ptr_->mtext) FUShmRawCell(rawCellSize);

}

#include "EventFilter/ResourceBroker/interface/RecoMsgBuf.h"
#include <iostream>
#include <cstdlib>

using namespace evf;

RecoMsgBuf::RecoMsgBuf(unsigned int size, unsigned int type) :
	MsgBuf(size, type), theRecoCell_(0) {
}

RecoMsgBuf::~RecoMsgBuf() {
	theRecoCell_->~FUShmRecoCell();
	//delete[] buf_;
}

void RecoMsgBuf::initialise(unsigned int recoCellSize) {
	// construct the FUShmRawCell at the buffer address
	// the memory is allocated in the super constructor
	theRecoCell_ = new (ptr_->mtext) FUShmRecoCell(recoCellSize);
}

#include "EventFilter/Utilities/interface/SimpleMsgBuf.h"
#include <cstring>
#include <iostream>

using namespace evf;

SimpleMsgBuf::SimpleMsgBuf() :
	MsgBuf(), usedSize_(0) {
	setBufferPointers();
}

SimpleMsgBuf::SimpleMsgBuf(unsigned int size, unsigned int type) :
	MsgBuf(size, type), usedSize_(0) {
	setBufferPointers();
}

SimpleMsgBuf::~SimpleMsgBuf() {

}

//TODO ensure buffer is big enough for all these operations

void SimpleMsgBuf::setRLE(unsigned int run, unsigned int lumi, unsigned int evt) {
	// copy run number
	memcpy(pRun_, &run, sizeof(unsigned int));
	// copy lumi number
	memcpy(pLumi_, &lumi, sizeof(unsigned int));
	// copy event number
	memcpy(pEvt_, &evt, sizeof(unsigned int));
}

void SimpleMsgBuf::addFedSize(unsigned int fedSize) {
	memcpy(pVarSize_, &fedSize, sizeof(unsigned int));
	pVarSize_ += sizeof(unsigned int);
}

void SimpleMsgBuf::addFedData(const unsigned char* fedRawData,
		unsigned int segmentSize) {
	memcpy(pVarData_, fedRawData, segmentSize);
	pVarData_ += segmentSize;

	// increase used size by current fed data
	usedSize_ += segmentSize;

}

void SimpleMsgBuf::getRun(unsigned int& run) const {
	memcpy(&run, pRun_, sizeof(unsigned int));
}

void SimpleMsgBuf::getLumi(unsigned int& lumi) const {
	memcpy(&lumi, pLumi_, sizeof(unsigned int));
}

void SimpleMsgBuf::getEvt(unsigned int& evt) const {
	memcpy(&evt, pEvt_, sizeof(unsigned int));
}

char* SimpleMsgBuf::getFedSizeStart() {
	return pFedSizes_;
}

char* SimpleMsgBuf::getFedDataStart() {
	return pFedRawData_;
}

char* SimpleMsgBuf::getFedDataEnd() {
	return pVarData_;
}

void SimpleMsgBuf::setBufferPointers() {
	pRun_ = ptr_->mtext;
	pLumi_ = pRun_ + sizeof(unsigned int);
	pEvt_ = pLumi_ + sizeof(unsigned int);
	pFedSizes_ = pEvt_ + sizeof(unsigned int);
	pFedRawData_ = pFedSizes_ + N_FEDS * sizeof(unsigned int);

	// used size is size of header (run, lumi, evt, N_FEDS * sizeof(unsigned int))
	// increases as raw data is added
	usedSize_ = pFedRawData_ - pRun_;

	pVarSize_ = pFedSizes_;
	pVarData_ = pFedRawData_;
}

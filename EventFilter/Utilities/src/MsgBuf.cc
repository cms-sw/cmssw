#include "EventFilter/Utilities/interface/MsgBuf.h"
#include <iostream>

namespace evf {
MsgBuf::MsgBuf() :
	msize_(MAX_MSG_SIZE) {

	buf_ = new unsigned char[msize_ + sizeof(long int)];
	ptr_ = (msgbuf*) buf_;
	ptr_->mtype = MSQM_MESSAGE_TYPE_NOP;
}

MsgBuf::MsgBuf(unsigned int size, unsigned int type) :
	msize_(size) {
	buf_ = new unsigned char[msize_ + sizeof(long int)];
	ptr_ = (msgbuf*) buf_;
	ptr_->mtype = type;
}

MsgBuf::MsgBuf(const MsgBuf &b) :
	msize_(b.msize_) {
	buf_ = new unsigned char[msize_ + sizeof(long int)];
	ptr_ = (msgbuf*) buf_;
	memcpy(buf_, b.buf_, msize_);
}
MsgBuf & MsgBuf::operator=(const MsgBuf &b) {
	msize_ = b.msize_;
	buf_ = new unsigned char[msize_ + sizeof(long int)];
	ptr_ = (msgbuf*) buf_;
	memcpy(buf_, b.buf_, msize_ + sizeof(long int));
	return *this;
}
size_t MsgBuf::msize() {
	return msize_;
}
MsgBuf::~MsgBuf() {
	delete[] buf_;
}
msgbuf* MsgBuf::operator->() {
	return ptr_;
}
}

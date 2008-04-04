/*
  A ring buffer for the Storage Manager Event Server
  Serialized events are stored (EventMsgView). The size of the
  buffer can be given in the constructor but the max size
  of each event is hardwired to 7MB (actually 7000000).
    This is used by the Storage Manager FragmentCollector.
  When the ring buffer is full it the oldest event gets
  overwritten.

  31 Jul 2006 - KAB - First implementation.
*/
#include "EventFilter/StorageManager/interface/EvtMsgRingBuffer.h"

#include "FWCore/Utilities/interface/DebugMacros.h"

#include <iostream>

using namespace stor;
using namespace edm;
using namespace std;

EvtMsgRingBuffer::EvtMsgRingBuffer(unsigned int size):
  maxsize_(size), head_(0), tail_(0), nextfree_(0),
  ring_buffer_(maxsize_), ring_totmsgsize_(maxsize_)
{
  for (int i=0; i<(int)maxsize_; i++)
    ring_buffer_[i].reserve(MAX_EVTBUF_SIZE);
}

EventMsgView EvtMsgRingBuffer::pop_front()
{
  // check to see if empty - what to return?
  // calling code needs to check ifEmpty() and do appropriate thing
  // should really return a message an appropriate MsgCode
  if(isEmpty()) return EventMsgView(&ring_buffer_[head_][0]);

  EventMsgView popthis(&ring_buffer_[head_][0]);
  if(head_ == (int)maxsize_-1) head_ = 0;
  else head_++;
  // see if now empty and need to move the tail
  if(head_ == nextfree_) tail_ = head_;
  FDEBUG(10) << "after pop_front head = " << head_ << " tail = " << tail_
             << " next free = " << nextfree_ << endl;
  FDEBUG(10) << "popping event = " << popthis.event() << endl;
  FDEBUG(10) << "pop_front msg event size " << popthis.eventLength() << std::endl;
  return popthis;
}

void EvtMsgRingBuffer::push_back(EventMsgView inputMsgView)
{
  // see if full and need to move the head
  if(isFull()) {
    if(head_ == (int)maxsize_-1) head_ = 0;
    else head_++;
  }

  // 31-Jul-2006, KAB:  I expect that the correct way to copy an
  // event message is create an EventMsgBuilder using data fetched from
  // the input message.  However, there currently doesn't seem to be 
  // sufficient methods in EventMsgView to support this.  So, for now,
  // we just copy the data directly (ugly hack).
  unsigned char* pos = (unsigned char*) &ring_buffer_[nextfree_][0];
  unsigned char* from = (unsigned char*)
    ((unsigned int) inputMsgView.eventData() - inputMsgView.headerSize());
  int dsize = inputMsgView.eventLength() + inputMsgView.headerSize();
  copy(from,from+dsize,pos);
  ring_totmsgsize_[nextfree_]=dsize;

  tail_ = nextfree_;
  if(tail_ == (int)maxsize_-1) nextfree_ = 0;
  else nextfree_++;
  FDEBUG(10) << "after push_back head = " << head_ << " tail = " << tail_
             << " next free = " << nextfree_ << endl;
}

bool EvtMsgRingBuffer::isEmpty()
{
  return ((head_ == tail_) && (head_ == nextfree_));
}

bool EvtMsgRingBuffer::isFull()
{
  return ((head_ != tail_) && (head_ == nextfree_));
}

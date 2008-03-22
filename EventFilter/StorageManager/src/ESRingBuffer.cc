/*
  A ring buffer for the Storage Manager Event Server
  Serialized events are stored (EventMsg). The size of the
  buffer can be given in the constructor but the max size
  of each event is hardwired to 7MB (actually 7000000).
    This is used by the Storage Manager FragmentCollector.
  When the ring buffer is full it the oldest event gets
  overwritten.

  30 Mar 2006 - HWKC - First implementation.
*/
#include "EventFilter/StorageManager/interface/ESRingBuffer.h"

#include "FWCore/Utilities/interface/DebugMacros.h"

using namespace stor;
using namespace edm;
using namespace std;

ESRingBuffer::ESRingBuffer(unsigned int size):
  maxsize_(size), head_(0), tail_(0), nextfree_(0),
  ring_buffer_(maxsize_), ring_totmsgsize_(maxsize_)
{
  for (int i=0; i<(int)maxsize_; i++)
    ring_buffer_[i].reserve(MAX_EVTBUF_SIZE);
}

edm::EventMsg ESRingBuffer::pop_front()
{
  // check to see if empty - what to return?
  // calling code needs to check ifEmpty() and do appropriate thing
  // should really return a message an appropriate MsgCode
  if(isEmpty()) return edm::EventMsg(&ring_buffer_[head_][0]);

  edm::EventMsg popthis(&ring_buffer_[head_][0],ring_totmsgsize_[head_]);
  if(head_ == (int)maxsize_-1) head_ = 0;
  else head_++;
  // see if now empty and need to move the tail
  if(head_ == nextfree_) tail_ = head_;
  FDEBUG(10) << "after pop_front head = " << head_ << " tail = " << tail_
             << " next free = " << nextfree_ << endl;
  FDEBUG(10) << "popping event = " << popthis.getEventNumber() << endl;
  FDEBUG(10) << "pop_front msg event size " << popthis.totalSize() << std::endl;
  return popthis;
}

void ESRingBuffer::push_back(edm::EventMsg msg)
{
  // see if full and need to move the head
  if(isFull()) {
    if(head_ == (int)maxsize_-1) head_ = 0;
    else head_++;
  }
  // use the correct total message size msgSize() instead of msg.totalSize()
  // which is always set to 7000000 by FragmentCollector
  edm::EventMsg em(&ring_buffer_[nextfree_][0],msg.msgSize(),
              msg.getEventNumber(),msg.getRunNumber(),
              1,1);
  unsigned char* pos = (unsigned char*)em.data();
  int dsize = msg.getDataSize();
  unsigned char* from=(unsigned char*)msg.data();
  copy(from,from+dsize,pos);
  ring_totmsgsize_[nextfree_]=em.totalSize();

  tail_ = nextfree_;
  if(tail_ == (int)maxsize_-1) nextfree_ = 0;
  else nextfree_++;
  FDEBUG(10) << "after push_back head = " << head_ << " tail = " << tail_
             << " next free = " << nextfree_ << endl;
}

bool ESRingBuffer::isEmpty()
{
  return ((head_ == tail_) && (head_ == nextfree_));
}

bool ESRingBuffer::isFull()
{
  return ((head_ != tail_) && (head_ == nextfree_));
}

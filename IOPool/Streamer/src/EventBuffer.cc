
#include "IOPool/Streamer/interface/EventBuffer.h"

namespace edm
{

  EventBuffer::EventBuffer(int max_event_size, int max_queue_depth):
    max_event_size_(max_event_size),max_queue_depth_(max_queue_depth),
    pos_(max_queue_depth-1),mem_(max_event_size * max_queue_depth),
    queue_(max_queue_depth),
    fpos_(),bpos_()
  {
    // throw if event size 0 or queue depth 0

    for(char* i = &mem_[0]; i < &mem_[mem_.size()]; i += max_event_size)
      buffer_pool_.push_back(i);

  }

  EventBuffer::~EventBuffer() { }

  EventBuffer::Buffer EventBuffer::getProducerBuffer()
  {
    // get lock
    boost::mutex::scoped_lock sl(pool_lock_);
    // wait for buffer to appear
    while(pos_ < 0) {
	pool_cond_.wait(sl);
    }
    void* v = buffer_pool_[pos_];
    --pos_;
    return Buffer(v,max_event_size_);
  }

  void EventBuffer::releaseProducerBuffer(void* v)
  {
    // get lock
    boost::mutex::scoped_lock sl(pool_lock_);
    ++pos_;
    buffer_pool_[pos_] = v;
    pool_cond_.notify_one();
    // pool_cond_.notify_all();
  }

  void EventBuffer::commitProducerBuffer(void* v, int len)
  {
    // get lock
    boost::mutex::scoped_lock sl(queue_lock_);
    // if full, wait for item to be removed
    while((bpos_+max_queue_depth_)==fpos_) {
	push_cond_.wait(sl);
    }
    
    // put buffer into queue
    queue_[fpos_ % max_queue_depth_]=Buffer(v,len);
    ++fpos_;
    // signal consumer
    pop_cond_.notify_one();
    // pop_cond_.notify_all();
  }
  
  EventBuffer::Buffer EventBuffer::getConsumerBuffer()
  {
    // get lock
    boost::mutex::scoped_lock sl(queue_lock_);
    // if empty, wait for item to appear
    while(bpos_==fpos_) {
	pop_cond_.wait(sl);
    }
    // get a buffer from the queue and return it
    Buffer v = queue_[bpos_ % max_queue_depth_];
    ++bpos_;
    // note that these operations cannot throw
    // signal producer
    push_cond_.notify_one();
    // push_cond_.notify_all();
    return v;
  }

  void EventBuffer::releaseConsumerBuffer(void* v)
  {
    // should the buffer be placed back onto the queue and not released?
    // we got here because a commit did to occur in the consumer.
    // we will allow consumers to call or not call commit for now, meaning
    // that we cannot distinguish between exception conditions and normal
    // return.  The buffer will always be released
    releaseProducerBuffer(v);
  }

  void EventBuffer::commitConsumerBuffer(void* v, int)
  {
    releaseProducerBuffer(v);
  }
}

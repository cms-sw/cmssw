#ifndef IOPool_Streamer_EventBuffer_h
#define IOPool_Streamer_EventBuffer_h

// -*- C++ -*-

/*
 A bounded queue for use in a multi-threaded producer/consumer application.
 This is a simple design.  It is only meant to be used where there is
 one consumer and one or more producers using the a queue instance.

 The problem with multiple consumers is the separate front/pop
 member functions.  If they are pulled together into one function,
 multiple consumers may be possible, but exception safety would then
 be a problem - popping an item off the queue to be held as a local
 variable, followed by its removal from the queue.  Having fixed size
 buffers within a fixed size pool and using a circular buffer as the
 queue alleviates most of this problem because exceptions will not
 occur during manipulation.  The only problem left to be checked is
 how (or if) the boost mutex manipulation can throw and when.

 Note: the current implementation has no protection again unsigned int
 overflows

 Missing:
  - the ring buffer is really not used to its fullest extent
  - the buffer sizes are fixed and cannot grow
  - a simple Buffer object is returned that has the pointer and len
    separate.  The length should be stored as the first word of the
    buffer itself
  - timeouts for consumer
  - good way to signal to consumer to end
  - keeping the instance of this thing around until all using threads are 
    done with it

*/

#include <vector>
#include "boost/thread/mutex.hpp"
#include "boost/thread/condition.hpp"

namespace edm {

  class EventBuffer
  {
  public:
    struct Buffer
    {
      Buffer():ptr_(),len_() { }
      Buffer(void* p,int len):ptr_(p),len_(len) { }

      void* ptr_;
      int len_;
    };

    EventBuffer(int max_event_size, int max_queue_depth);
    ~EventBuffer();

    struct ConsumerType
    {
      static EventBuffer::Buffer get(EventBuffer& b)
      { return b.getConsumerBuffer(); }
      static void release(EventBuffer& b, void* v)
      { b.releaseConsumerBuffer(v); }
      static void commit(EventBuffer& b, void* v,int size)
      { b.commitConsumerBuffer(v,size); }
    };
    struct ProducerType
    {
      static EventBuffer::Buffer get(EventBuffer& b)
      { return b.getProducerBuffer(); }
      static void release(EventBuffer& b, void* v)
      { b.releaseProducerBuffer(v); }
      static void commit(EventBuffer& b, void* v,int size)
      { b.commitProducerBuffer(v,size); }
    };

    template <class T>
    class OperateBuffer
    {
    public:
      explicit OperateBuffer(EventBuffer& b):
	b_(b),v_(T::get(b)),committed_(false) { }
      ~OperateBuffer()
      { if(!committed_) T::release(b_,v_.ptr_); }

      void* buffer() const { return v_.ptr_; }
      int size() const { return v_.len_; }
      void commit(int size=0) { T::commit(b_,v_.ptr_,size); committed_=true; }

    private:
      EventBuffer& b_;
      EventBuffer::Buffer v_;
      bool committed_;
    };

    typedef OperateBuffer<ConsumerType> ConsumerBuffer;
    typedef OperateBuffer<ProducerType> ProducerBuffer;

    Buffer getProducerBuffer();
    void releaseProducerBuffer(void*);
    void commitProducerBuffer(void*,int);

    Buffer getConsumerBuffer();
    void releaseConsumerBuffer(void*);
    void commitConsumerBuffer(void*,int);

    int maxEventSize() const { return max_event_size_; }
    int maxQueueDepth() const { return max_queue_depth_; }

  private:
    // no copy
    EventBuffer(const EventBuffer&) { }

    // the memory for the buffers
    typedef std::vector<char> ByteArray;
    // the pool of buffers
    typedef std::vector<void*> Pool;
    // the queue
    typedef std::vector<Buffer> Queue;

    int max_event_size_;
    int max_queue_depth_;
    int pos_; // use pool as stack of avaiable buffers
    ByteArray mem_;
    Pool buffer_pool_;
    Queue queue_;
    unsigned int fpos_, bpos_; // positions for queue - front and back 

    boost::mutex pool_lock_;
    boost::mutex queue_lock_;
    boost::condition pool_cond_;
    boost::condition pop_cond_;
    boost::condition push_cond_;

 public:
    bool empty() { boost::mutex::scoped_lock sl(queue_lock_); return (bpos_==fpos_); }

  };


}
#endif




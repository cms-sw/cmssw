#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "EventFilter/StorageManager/interface/ConcurrentQueue.h"
#include "FWCore/Utilities/interface/CPUTimer.h"

#include "boost/date_time/posix_time/posix_time_types.hpp"
#include "boost/thread.hpp"
#include "boost/shared_ptr.hpp"
#include "boost/bind.hpp"

#include <math.h>
#include <vector>

struct QueueElement
{
  QueueElement() : value_(0), size_(0) {};
  QueueElement(const int& value) : value_(value), size_(sizeof(int)) {};
  QueueElement(const unsigned int& value) : value_(value), size_(sizeof(unsigned int)) {};
  QueueElement(const unsigned long& value) : value_(value), size_(sizeof(unsigned long)) {};
  QueueElement(const unsigned long long& value) : value_(value), size_(sizeof(unsigned long long)) {};

  bool operator==(const QueueElement& other) const
  { return (other.value_ == value_); }

  friend std::ostream& operator<<(std::ostream& os, const QueueElement& qe)
  { os << qe.value_; return os; }
  
  size_t memoryUsed() const { return size_; };

  unsigned long long value_;
  size_t size_;
};

typedef stor::ConcurrentQueue<QueueElement> queue_t;
typedef stor::FailIfFull<QueueElement>::QueueIsFull exception_t;
typedef stor::ConcurrentQueue<QueueElement, stor::KeepNewest<QueueElement> > keepnewest_t;
typedef stor::ConcurrentQueue<QueueElement, stor::RejectNewest<QueueElement> > rejectnewest_t;

class FillQueue
{
public:
  FillQueue(boost::shared_ptr<queue_t>& p, 
	    unsigned int delay,
	    unsigned int nEntries):
    sharedQueue_(p),
    delay_(delay),
    counter_(nEntries+1)
  { }  

  void operator()();

  void waiting_fill();
  
private:
  boost::shared_ptr<queue_t> sharedQueue_;
  unsigned int               delay_;
  unsigned int               counter_;
};


void FillQueue::operator()()
{
  while(--counter_)
    {
      sleep(delay_);
      sharedQueue_->enqNowait(counter_);
    }
}

void FillQueue::waiting_fill()
{
  while(--counter_) sharedQueue_->enqWait(counter_);
}

class DrainQueue
{
public:
  DrainQueue(boost::shared_ptr<queue_t>& p, unsigned int delay) :
    sharedQueue_(p),
    delay_(delay),
    counter_(0)
  { }  

  void operator()();
  unsigned int count() const;

private:
  boost::shared_ptr<queue_t> sharedQueue_;
  unsigned int               delay_;
  unsigned int               counter_;
};

void DrainQueue::operator()()
{
  queue_t::ValueType val;
  while(true)
    {
      sleep(delay_);
      if (sharedQueue_->deqNowait(val)) ++counter_;
      else return;
    }
}

unsigned int DrainQueue::count() const
{
  return counter_;
}


/*
  DrainTimedQueue is used for testing the timed-wait version of the
  dequeue functionality.
 */

class DrainTimedQueue
{
public:
  DrainTimedQueue(boost::shared_ptr<queue_t>& p, unsigned int delay) :
    sharedQueue_(p),
    delay_(delay),
    counter_(0)
  { }  

  void operator()();
  unsigned int count() const;

private:
  boost::shared_ptr<queue_t> sharedQueue_;
  unsigned int               delay_;
  unsigned int               counter_;
};

void DrainTimedQueue::operator()()
{
  queue_t::ValueType val;
  while(true)
    {
      sleep(delay_);
      if (sharedQueue_->deqNowait(val)) ++counter_;
      else return;
    }
}

unsigned int DrainTimedQueue::count() const
{
  return counter_;
}

class testConcurrentQueue : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testConcurrentQueue);
  CPPUNIT_TEST(default_q_is_empty);
  CPPUNIT_TEST(queue_is_fifo);
  CPPUNIT_TEST(enq_and_deq);
  CPPUNIT_TEST(many_fillers);
  CPPUNIT_TEST(enq_timing);
  CPPUNIT_TEST(change_capacity);
  CPPUNIT_TEST(failiffull);
  CPPUNIT_TEST(failiffull_memlimit);
  CPPUNIT_TEST(keepnewest);
  CPPUNIT_TEST(keepnewest_memlimit);
  CPPUNIT_TEST(keepnewest_memlimit2);
  CPPUNIT_TEST(rejectnewest);
  CPPUNIT_TEST(rejectnewest_memlimit);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp();
  void tearDown();

  void default_q_is_empty();
  void queue_is_fifo();
  void enq_and_deq();
  void many_fillers();
  void enq_timing();
  void change_capacity();
  void failiffull();
  void failiffull_memlimit();
  void keepnewest();
  void keepnewest_memlimit();
  void keepnewest_memlimit2();
  void rejectnewest();
  void rejectnewest_memlimit();

private:
  // No data members yet.
};

void
testConcurrentQueue::setUp()
{ 
}

void
testConcurrentQueue::tearDown()
{ 
}

void 
testConcurrentQueue::default_q_is_empty()
{
  std::cerr << "\nConcurrentQueue_t::default_q_is_empty\n";
  queue_t q;
  CPPUNIT_ASSERT(q.empty());
  CPPUNIT_ASSERT(!q.full());
}

void
testConcurrentQueue::queue_is_fifo()
{
  std::cerr << "\nConcurrentQueue_t::queue_is_fifo\n";
  queue_t q;
  q.enqNowait(1);
  q.enqNowait(2);
  q.enqNowait(3);
  queue_t::ValueType value;
  CPPUNIT_ASSERT(q.deqNowait(value));
  CPPUNIT_ASSERT(value == 1);
  CPPUNIT_ASSERT(q.deqNowait(value));
  CPPUNIT_ASSERT(value == 2);
  CPPUNIT_ASSERT(q.deqNowait(value));
  CPPUNIT_ASSERT(value == 3);
  CPPUNIT_ASSERT(q.empty());
  CPPUNIT_ASSERT(!q.full());
}

void
testConcurrentQueue::enq_and_deq()
{
  std::cerr << "\nConcurrentQueue_t::enq_and_deq\n";
  boost::shared_ptr<queue_t> q(new queue_t);
  unsigned int delay = 0;
  unsigned int num_items = 10000;
  boost::thread producer(FillQueue(q, delay, num_items));
  sleep(1); // give the producer a head-start
  boost::thread consumer(DrainQueue(q, delay));
  producer.join();
  consumer.join();
  CPPUNIT_ASSERT(q->size() == 0);
}

void
testConcurrentQueue::many_fillers()
{
  std::cerr << "\nConcurrentQueue_t::many_fillers\n";
  size_t num_fillers(3);
  unsigned int num_items(10);
  boost::shared_ptr<queue_t> q(new queue_t(num_items*num_fillers+1));
  
  boost::thread_group producers;
  typedef std::vector<FillQueue> fillers_t;
  fillers_t  fillers(num_fillers, FillQueue(q, 0, num_items));
  for (fillers_t::iterator
         i = fillers.begin(),
         e = fillers.end();
       i != e;
       ++i)
    {
      using boost::bind;
      using boost::thread;
      producers.add_thread(new thread(bind(&FillQueue::waiting_fill, 
                                           &*i)));
    }
  producers.join_all();
  CPPUNIT_ASSERT(q->size() == num_items * num_fillers);
}

void
testConcurrentQueue::enq_timing()
{
  std::cerr << "\nConcurrentQueue_t::enq_timing "
            << "(this may take up to 30 seconds)\n";
  queue_t q(1);

  // Queue is initially empty, so the first call should succeed.
  CPPUNIT_ASSERT_NO_THROW(q.enqNowait(1));
  CPPUNIT_ASSERT(q.size() == 1);
  CPPUNIT_ASSERT(q.capacity() == 1);
  CPPUNIT_ASSERT(q.full());

  // The queue is now full. The next enq should fail.
  edm::CPUTimer t;
  t.start();
  CPPUNIT_ASSERT_THROW(q.enqNowait(1), exception_t);
  t.stop();
  // We somewhat arbitrarily choose 100 milliseconds as "immediately
  // enough".
  CPPUNIT_ASSERT(t.realTime() < 0.1);  

  // Now test the timeout version, with a range of timeouts.
  for (unsigned long wait_time = 0; wait_time < 3; ++wait_time)
    {
      t.reset();
      CPPUNIT_ASSERT(q.size() == 1);
      t.start();
      CPPUNIT_ASSERT(!q.enqTimedWait(1, boost::posix_time::seconds(wait_time)));
      t.stop();
      // We somewhat arbitrarily choose 10 milliseconds as "good enough
      // resolution".
      CPPUNIT_ASSERT(fabs(t.realTime()-wait_time) < 0.01);
    }

  // Now test the version that waits indefinitiely. We fill the queue,
  // start a draining thread that delays before each deq, and then
  // make sure do eventually return from the call to enqWait.
  boost::shared_ptr<queue_t> qptr(new queue_t(1));
  CPPUNIT_ASSERT(qptr->capacity() == 1);
  CPPUNIT_ASSERT_NO_THROW(qptr->enqNowait(1));
  CPPUNIT_ASSERT(qptr->size() == 1);

  int delay = 2;
  boost::thread consumer(DrainQueue(qptr,delay));

  qptr->enqWait(delay);
  consumer.join();
  CPPUNIT_ASSERT(qptr->empty());  
}

void
testConcurrentQueue::change_capacity()
{
  std::cerr << "\nConcurrentQueue_t::change_capacity\n";
  queue_t q(1);
  CPPUNIT_ASSERT_NO_THROW(q.enqNowait(1));
  CPPUNIT_ASSERT_THROW(q.enqNowait(1), exception_t);
  CPPUNIT_ASSERT(!q.setCapacity(2));                 // did not reset
  CPPUNIT_ASSERT_THROW(q.enqNowait(3), exception_t); // ... so this fails.

  q.clear();
  CPPUNIT_ASSERT(q.setCapacity(2));
  CPPUNIT_ASSERT_NO_THROW(q.enqNowait(1));
  CPPUNIT_ASSERT_NO_THROW(q.enqNowait(2));
  CPPUNIT_ASSERT_THROW(q.enqNowait(3), exception_t);
  CPPUNIT_ASSERT(q.size() == 2);
  CPPUNIT_ASSERT(q.capacity() == 2);  
}

void
testConcurrentQueue::failiffull()
{
  std::cerr << "\nConcurrentQueue_t::failiffull\n";
  queue_t q(1);  
  CPPUNIT_ASSERT_NO_THROW(q.enqNowait(1));
  CPPUNIT_ASSERT_THROW(q.enqNowait(2), exception_t);
  CPPUNIT_ASSERT(q.size() == 1);
  queue_t::ValueType value;
  CPPUNIT_ASSERT(q.deqNowait(value));
  CPPUNIT_ASSERT(value==1);
  CPPUNIT_ASSERT(q.empty());
  CPPUNIT_ASSERT(q.used() == 0);
}

void
testConcurrentQueue::failiffull_memlimit()
{
  std::cerr << "\nConcurrentQueue_t::failiffull_memlimit\n";
  queue_t q(5,sizeof(int)); //memory for one int only
  CPPUNIT_ASSERT_NO_THROW(q.enqNowait(1));
  CPPUNIT_ASSERT_THROW(q.enqNowait(2), exception_t);
  CPPUNIT_ASSERT(q.size() == 1);
  CPPUNIT_ASSERT(q.used() == sizeof(int));
  queue_t::ValueType value;
  CPPUNIT_ASSERT(q.deqNowait(value));
  CPPUNIT_ASSERT(value==1);
  CPPUNIT_ASSERT(q.empty());
  CPPUNIT_ASSERT(q.used() == 0);
}

void
testConcurrentQueue::keepnewest()
{
  std::cerr << "\nConcurrentQueue_t::keepnewest\n";
  keepnewest_t q(1);
  CPPUNIT_ASSERT(q.enqNowait(1) == 0);
  CPPUNIT_ASSERT(q.enqNowait(2) == 1);
  CPPUNIT_ASSERT(q.size() == 1);
  keepnewest_t::ValueType value;
  CPPUNIT_ASSERT(q.deqNowait(value));
  CPPUNIT_ASSERT(value.first == 2);
  CPPUNIT_ASSERT(value.second == 1);
  CPPUNIT_ASSERT(q.empty());
  CPPUNIT_ASSERT(q.used() == 0);
}

void
testConcurrentQueue::keepnewest_memlimit()
{
  std::cerr << "\nConcurrentQueue_t::keepnewest_memlimit\n";
  keepnewest_t q(5,sizeof(int)); //memory for one int only
  CPPUNIT_ASSERT(q.enqNowait(1) == 0);
  CPPUNIT_ASSERT(q.enqNowait(2) == 1);
  CPPUNIT_ASSERT(q.size() == 1);
  CPPUNIT_ASSERT(q.used() == sizeof(int));
  keepnewest_t::ValueType value;
  CPPUNIT_ASSERT(q.deqNowait(value));
  CPPUNIT_ASSERT(value.first == 2);
  CPPUNIT_ASSERT(value.second == 1);
  CPPUNIT_ASSERT(q.empty());
  CPPUNIT_ASSERT(q.used() == 0);
}

void
testConcurrentQueue::keepnewest_memlimit2()
{
  std::cerr << "\nConcurrentQueue_t::keepnewest_memlimit2\n";
  keepnewest_t q(5,3*sizeof(uint32_t));
  CPPUNIT_ASSERT(q.enqNowait(QueueElement((uint32_t)1)) == 0);
  CPPUNIT_ASSERT(q.enqNowait(QueueElement((uint32_t)2)) == 0);
  CPPUNIT_ASSERT(q.enqNowait(QueueElement((uint32_t)3)) == 0);
  CPPUNIT_ASSERT(q.enqNowait(QueueElement((uint32_t)4)) == 1);
  CPPUNIT_ASSERT(q.size() == 3);
  CPPUNIT_ASSERT(q.used() == 3*sizeof(uint32_t));
  CPPUNIT_ASSERT(q.enqNowait(QueueElement((uint64_t)5)) == 2);
  CPPUNIT_ASSERT(q.size() == 2);
  CPPUNIT_ASSERT(q.used() == sizeof(uint32_t) + sizeof(uint64_t));

  keepnewest_t::ValueType value;
  CPPUNIT_ASSERT(q.deqNowait(value));
  CPPUNIT_ASSERT(value.first == (uint32_t)4);
  CPPUNIT_ASSERT(value.second == 3);

  CPPUNIT_ASSERT(q.deqNowait(value));
  CPPUNIT_ASSERT(value.first == (uint64_t)5);
  CPPUNIT_ASSERT(value.second == 0);

  CPPUNIT_ASSERT(q.empty());
  CPPUNIT_ASSERT(q.used() == 0);
}

void
testConcurrentQueue::rejectnewest()
{
  std::cerr << "\nConcurrentQueue_t::rejectnewest\n";
  rejectnewest_t q(1);
  CPPUNIT_ASSERT(q.enqNowait(1) == 0);
  CPPUNIT_ASSERT(q.enqNowait(2) == 1);
  CPPUNIT_ASSERT(q.size() == 1);
  rejectnewest_t::ValueType value;
  CPPUNIT_ASSERT(q.deqNowait(value));
  CPPUNIT_ASSERT(value.first == 1);
  CPPUNIT_ASSERT(value.second == 1);
  CPPUNIT_ASSERT(q.empty());
  CPPUNIT_ASSERT(q.enqNowait(3) == 0);
  CPPUNIT_ASSERT(q.size() == 1);
  CPPUNIT_ASSERT(q.deqNowait(value));
  CPPUNIT_ASSERT(value.first == 3);
  CPPUNIT_ASSERT(value.second == 0);
  CPPUNIT_ASSERT(q.empty());
  CPPUNIT_ASSERT(q.used() == 0);
}

void
testConcurrentQueue::rejectnewest_memlimit()
{
  std::cerr << "\nConcurrentQueue_t::rejectnewest_memlimit\n";
  rejectnewest_t q(5,sizeof(int)); //memory for one int only
  CPPUNIT_ASSERT(q.enqNowait(1) == 0);
  CPPUNIT_ASSERT(q.enqNowait(2) == 1);
  CPPUNIT_ASSERT(q.size() == 1);
  CPPUNIT_ASSERT(q.used() == sizeof(int));
  rejectnewest_t::ValueType value;
  CPPUNIT_ASSERT(q.deqNowait(value));
  CPPUNIT_ASSERT(value.first == 1);
  CPPUNIT_ASSERT(value.second == 1);
  CPPUNIT_ASSERT(q.empty());
  CPPUNIT_ASSERT(q.used() == 0);
  CPPUNIT_ASSERT(q.enqNowait(3) == 0);
  CPPUNIT_ASSERT(q.size() == 1);
  CPPUNIT_ASSERT(q.deqNowait(value));
  CPPUNIT_ASSERT(value.first == 3);
  CPPUNIT_ASSERT(value.second == 0);
  CPPUNIT_ASSERT(q.empty());
  CPPUNIT_ASSERT(q.used() == 0);
}

// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testConcurrentQueue);

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

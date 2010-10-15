#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "EventFilter/StorageManager/interface/ConcurrentQueue.h"
#include "FWCore/Utilities/interface/CPUTimer.h"
#include "boost/thread.hpp"
#include "boost/shared_ptr.hpp"
#include "boost/bind.hpp"

#include <math.h>
#include <vector>

typedef stor::ConcurrentQueue<int> queue_t;

class FillQueue
{
public:
  FillQueue(boost::shared_ptr<queue_t>& p, 
	    unsigned int delay,
	    unsigned int nEntries):
    _sharedQueue(p),
    _delay(delay),
    _counter(nEntries+1)
  { }  

  void operator()();

  void waiting_fill();
  
private:
  boost::shared_ptr<queue_t> _sharedQueue;
  unsigned int               _delay;
  unsigned int               _counter;
};


void FillQueue::operator()()
{
  while(--_counter)
    {
      sleep(_delay);
      _sharedQueue->enq_nowait(_counter);
    }
}

void FillQueue::waiting_fill()
{
  while(--_counter) _sharedQueue->enq_wait(_counter);
}

class DrainQueue
{
public:
  DrainQueue(boost::shared_ptr<queue_t>& p, unsigned int delay) :
    _sharedQueue(p),
    _delay(delay),
    _counter(0)
  { }  

  void operator()();
  unsigned int count() const;

private:
  boost::shared_ptr<queue_t> _sharedQueue;
  unsigned int               _delay;
  unsigned int               _counter;
};

void DrainQueue::operator()()
{
  queue_t::value_type val;
  while(true)
    {
      sleep(_delay);
      if (_sharedQueue->deq_nowait(val)) ++_counter;
      else return;
    }
}

unsigned int DrainQueue::count() const
{
  return _counter;
}


/*
  DrainTimedQueue is used for testing the timed-wait version of the
  dequeue functionality.
 */

class DrainTimedQueue
{
public:
  DrainTimedQueue(boost::shared_ptr<queue_t>& p, unsigned int delay) :
    _sharedQueue(p),
    _delay(delay),
    _counter(0)
  { }  

  void operator()();
  unsigned int count() const;

private:
  boost::shared_ptr<queue_t> _sharedQueue;
  unsigned int               _delay;
  unsigned int               _counter;
};

void DrainTimedQueue::operator()()
{
  queue_t::value_type val;
  while(true)
    {
      sleep(_delay);
      if (_sharedQueue->deq_nowait(val)) ++_counter;
      else return;
    }
}

unsigned int DrainTimedQueue::count() const
{
  return _counter;
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
  stor::ConcurrentQueue<int> q;
  CPPUNIT_ASSERT(q.empty());
  CPPUNIT_ASSERT(!q.full());
}

void
testConcurrentQueue::queue_is_fifo()
{
  std::cerr << "\nConcurrentQueue_t::queue_is_fifo\n";
  stor::ConcurrentQueue<int> q;
  q.enq_nowait(1);
  q.enq_nowait(2);
  q.enq_nowait(3);
  int value(0);
  CPPUNIT_ASSERT(q.deq_nowait(value));
  CPPUNIT_ASSERT(value == 1);
  CPPUNIT_ASSERT(q.deq_nowait(value));
  CPPUNIT_ASSERT(value == 2);
  CPPUNIT_ASSERT(q.deq_nowait(value));
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
  CPPUNIT_ASSERT(q.enq_nowait(1));
  CPPUNIT_ASSERT(q.size() == 1);
  CPPUNIT_ASSERT(q.capacity() == 1);
  CPPUNIT_ASSERT(q.full());

  // The queue is now full. The next enq should fail.
  edm::CPUTimer t;
  t.start();
  CPPUNIT_ASSERT(!q.enq_nowait(1));
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
      CPPUNIT_ASSERT(!q.enq_timed_wait(1, wait_time));
      t.stop();
      // We somewhat arbitrarily choose 10 milliseconds as "good enough
      // resolution".
      CPPUNIT_ASSERT(fabs(t.realTime()-wait_time) < 0.01);
    }

  // Now test the version that waits indefinitiely. We fill the queue,
  // start a draining thread that delays before each deq, and then
  // make sure do eventually return from the call to enq_wait.
  boost::shared_ptr<queue_t> qptr(new queue_t(1));
  CPPUNIT_ASSERT(qptr->capacity() == 1);
  CPPUNIT_ASSERT(qptr->enq_nowait(1));
  CPPUNIT_ASSERT(qptr->size() == 1);

  unsigned long delay = 2;
  boost::thread consumer(DrainQueue(qptr,delay));

  qptr->enq_wait(delay);
  consumer.join();
  CPPUNIT_ASSERT(qptr->empty());  
}

void
testConcurrentQueue::change_capacity()
{
  std::cerr << "\nConcurrentQueue_t::change_capacity\n";
  queue_t q(1);
  CPPUNIT_ASSERT(q.enq_nowait(1));
  CPPUNIT_ASSERT(!q.enq_nowait(1));
  CPPUNIT_ASSERT(!q.set_capacity(2));  // did not reset
  CPPUNIT_ASSERT(!q.enq_nowait(3));    // ... so this fails.

  q.clear();
  CPPUNIT_ASSERT(q.set_capacity(2));
  CPPUNIT_ASSERT(q.enq_nowait(1));
  CPPUNIT_ASSERT(q.enq_nowait(2));
  CPPUNIT_ASSERT(!q.enq_nowait(3));
  CPPUNIT_ASSERT(q.size() == 2);
  CPPUNIT_ASSERT(q.capacity() == 2);  
}

void
testConcurrentQueue::failiffull()
{
  std::cerr << "\nConcurrentQueue_t::failiffull\n";
  stor::ConcurrentQueue<int, stor::FailIfFull<int> > q(1);  
  CPPUNIT_ASSERT(q.enq_nowait(1));
  CPPUNIT_ASSERT(!q.enq_nowait(2));
  CPPUNIT_ASSERT(q.size() == 1);
  int value;
  CPPUNIT_ASSERT(q.deq_nowait(value));
  CPPUNIT_ASSERT(value==1);
}

void
testConcurrentQueue::failiffull_memlimit()
{
  std::cerr << "\nConcurrentQueue_t::failiffull_memlimit\n";
  stor::ConcurrentQueue<int, stor::FailIfFull<int> > q(5,sizeof(int)); //memory for one int only
  CPPUNIT_ASSERT(q.enq_nowait(1));
  CPPUNIT_ASSERT(!q.enq_nowait(2));
  CPPUNIT_ASSERT(q.size() == 1);
  CPPUNIT_ASSERT(q.used() == sizeof(int));
  int value;
  CPPUNIT_ASSERT(q.deq_nowait(value));
  CPPUNIT_ASSERT(value==1);
}

void
testConcurrentQueue::keepnewest()
{
  std::cerr << "\nConcurrentQueue_t::keepnewest\n";
  stor::ConcurrentQueue<int, stor::KeepNewest<int> > q(1);
  q.enq_nowait(1);
  q.enq_nowait(2);
  CPPUNIT_ASSERT(q.size() == 1);
  int value;
  CPPUNIT_ASSERT(q.deq_nowait(value));
  CPPUNIT_ASSERT(value == 2);
}

void
testConcurrentQueue::keepnewest_memlimit()
{
  std::cerr << "\nConcurrentQueue_t::keepnewest_memlimit\n";
  stor::ConcurrentQueue<int, stor::KeepNewest<int> > q(5,sizeof(int)); //memory for one int only
  q.enq_nowait(1);
  q.enq_nowait(2);
  CPPUNIT_ASSERT(q.size() == 1);
  CPPUNIT_ASSERT(q.used() == sizeof(int));
  int value;
  CPPUNIT_ASSERT(q.deq_nowait(value));
  CPPUNIT_ASSERT(value == 2);
}

void
testConcurrentQueue::rejectnewest()
{
  std::cerr << "\nConcurrentQueue_t::rejectnewest\n";
  stor::ConcurrentQueue<int, stor::RejectNewest<int> > q(1);
  q.enq_nowait(1);
  q.enq_nowait(2);
  CPPUNIT_ASSERT(q.size() == 1);
  int value;
  CPPUNIT_ASSERT(q.deq_nowait(value));
  CPPUNIT_ASSERT(value == 1);
}

void
testConcurrentQueue::rejectnewest_memlimit()
{
  std::cerr << "\nConcurrentQueue_t::rejectnewest_memlimit\n";
  stor::ConcurrentQueue<int, stor::RejectNewest<int> > q(5,sizeof(int)); //memory for one int only
  q.enq_nowait(1);
  q.enq_nowait(2);
  CPPUNIT_ASSERT(q.size() == 1);
  CPPUNIT_ASSERT(q.used() == sizeof(int));
  int value;
  CPPUNIT_ASSERT(q.deq_nowait(value));
  CPPUNIT_ASSERT(value == 1);
}

// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testConcurrentQueue);

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "EventFilter/StorageManager/interface/ExpirableQueue.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/Utils.h"

#include "EventFilter/StorageManager/test/TestHelper.h"

#include <cstdlib> // for size_t

typedef stor::ExpirableQueue<stor::I2OChain, stor::RejectNewest<stor::I2OChain> > DN_q_t;
typedef stor::ExpirableQueue<stor::I2OChain, stor::KeepNewest<stor::I2OChain> > DO_q_t;

using stor::utils::Duration_t;
using stor::testhelper::allocate_frame_with_sample_header;

using namespace stor;

//  -------------------------------------------------------------------
//  The following function templates are used in the tests below, to
//  assure uniformity in the testing of both instantiations of
//  ExpirableQueue.
//  -------------------------------------------------------------------

template <class Q>
void
test_default()
{
  // Default constructed queues are empty.
  Q q;
  CPPUNIT_ASSERT(q.empty());
  CPPUNIT_ASSERT(!q.full());

  // Clearing the queue should have no effect.
  q.clear();
  CPPUNIT_ASSERT(q.empty());
  CPPUNIT_ASSERT(!q.full());
}

template <class Q>
void
test_fill_and_go_stale()
{
  // Make a queue with a short time-to-staleness and small capacity,
  // so we can see it fill up.
  size_t capacity(10);
  Duration_t seconds_to_stale = boost::posix_time::seconds(2);
  Q q(capacity, seconds_to_stale);
  CPPUNIT_ASSERT(!q.full());

  // Pump in enough events to fill the queue.
  for (size_t i = 0; i < capacity; ++i)
    {
      I2OChain event(allocate_frame_with_sample_header(0,1,1));
      CPPUNIT_ASSERT(q.enqNowait(event) == 0);
    }
  CPPUNIT_ASSERT(q.full());

  // An immediate call to clearIfStale should do nothing.
  size_t clearedEvents;
  CPPUNIT_ASSERT(!q.clearIfStale(utils::getCurrentTime(),clearedEvents));
  CPPUNIT_ASSERT(q.full());
  CPPUNIT_ASSERT(clearedEvents == 0);

  // After waiting for our staleness interval, the queue should still
  // be full. But then a call to clearIfStale should clear the queue.
  utils::sleep(seconds_to_stale);

  CPPUNIT_ASSERT(q.full());
  CPPUNIT_ASSERT(!q.empty());

  CPPUNIT_ASSERT(q.clearIfStale(utils::getCurrentTime(),clearedEvents));

  CPPUNIT_ASSERT(!q.full());
  CPPUNIT_ASSERT(q.empty());
  CPPUNIT_ASSERT(clearedEvents == capacity);
}

template <class Q>
void
test_pop_freshens_queue()
{
  size_t capacity(2);
  Duration_t seconds_to_stale = boost::posix_time::seconds(1);
  Q q(capacity, seconds_to_stale);

  // Push in an event, then let the queue go stale.
  CPPUNIT_ASSERT(q.enqNowait(I2OChain(allocate_frame_with_sample_header(0,1,1))) == 0);
  CPPUNIT_ASSERT(!q.empty());
  utils::sleep(seconds_to_stale);

  // Verify the queue has gone stale.
  size_t clearedEvents;
  CPPUNIT_ASSERT(q.clearIfStale(utils::getCurrentTime(),clearedEvents));
  CPPUNIT_ASSERT(q.empty());
  CPPUNIT_ASSERT(clearedEvents == 1);

  // A queue that has gone stale should reject a new event
  CPPUNIT_ASSERT(q.enqNowait(I2OChain(allocate_frame_with_sample_header(0,1,1))) == 1);
  CPPUNIT_ASSERT(q.empty());

  // Popping from the queue should make it non-stale. This must be
  // true *even if the queue is empty*, so that popping does not get
  // an event.
  typename Q::ValueType popped;
  CPPUNIT_ASSERT(!q.deqNowait(popped));
  CPPUNIT_ASSERT(q.empty());
  CPPUNIT_ASSERT(!q.clearIfStale(utils::getCurrentTime(),clearedEvents));
  CPPUNIT_ASSERT(clearedEvents == 0);
}

//  -------------------------------------------------------------------

class testExpirableQueue : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testExpirableQueue);

  CPPUNIT_TEST(default_queue);
  CPPUNIT_TEST(fill_and_go_stale);
  CPPUNIT_TEST(pop_freshens_queue);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp();
  void tearDown();

  void default_queue();
  void fill_and_go_stale();
  void pop_freshens_queue();

private:

};

void
testExpirableQueue::setUp()
{
  CPPUNIT_ASSERT(g_factory);
  CPPUNIT_ASSERT(g_alloc);
  CPPUNIT_ASSERT(g_pool);
}

void
testExpirableQueue::tearDown()
{ 
}

void 
testExpirableQueue::default_queue()
{
  test_default<DN_q_t>();
  test_default<DO_q_t>();
}

void
testExpirableQueue::fill_and_go_stale()
{
  test_fill_and_go_stale<DN_q_t>();
  test_fill_and_go_stale<DO_q_t>();
}

void
testExpirableQueue::pop_freshens_queue()
{
  test_pop_freshens_queue<DN_q_t>();
  test_pop_freshens_queue<DO_q_t>();
}

// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testExpirableQueue);


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

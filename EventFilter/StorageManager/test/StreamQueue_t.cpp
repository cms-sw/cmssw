#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "EventFilter/StorageManager/interface/StreamQueue.h"

#include "EventFilter/StorageManager/test/TestHelper.h"

using stor::testhelper::outstanding_bytes;
using stor::testhelper::allocate_frame;
using stor::testhelper::allocate_frame_with_basic_header;
using namespace stor;


class testStreamQueue : public CppUnit::TestFixture
{
  typedef toolbox::mem::Reference Reference;
  CPPUNIT_TEST_SUITE(testStreamQueue);
  CPPUNIT_TEST(enq_deq);
  CPPUNIT_TEST(enq_deq_memlimit);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp();
  void tearDown();

  void enq_deq();
  void enq_deq_memlimit();

private:
  size_t memory_consumed_by_one_frame;
};

void
testStreamQueue::setUp()
{
  I2OChain frag(allocate_frame());
  memory_consumed_by_one_frame = outstanding_bytes();
}

void
testStreamQueue::tearDown()
{
}

void
testStreamQueue::enq_deq()
{
  StreamQueue sq;
  sq.setCapacity(1);
  
  CPPUNIT_ASSERT(sq.capacity() == 1);
  CPPUNIT_ASSERT(sq.empty());

  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  Reference* ref = allocate_frame_with_basic_header(I2O_SM_DATA, 0, 1);
  I2OChain c1(ref);
  CPPUNIT_ASSERT(!c1.empty());
  CPPUNIT_ASSERT_NO_THROW(sq.enqNowait(c1));

  CPPUNIT_ASSERT(sq.size() == 1);
  CPPUNIT_ASSERT(sq.used() == memory_consumed_by_one_frame);
  CPPUNIT_ASSERT(sq.full());

  I2OChain c2;
  CPPUNIT_ASSERT(c2.empty());
  CPPUNIT_ASSERT(sq.deqNowait(c2));
  CPPUNIT_ASSERT(!c2.empty());

  CPPUNIT_ASSERT(sq.empty());
  CPPUNIT_ASSERT(sq.used() == 0);
}

void
testStreamQueue::enq_deq_memlimit()
{
  StreamQueue sq(10, memory_consumed_by_one_frame);
  
  CPPUNIT_ASSERT(sq.capacity() == 10);
  CPPUNIT_ASSERT(sq.memory() == memory_consumed_by_one_frame);
  CPPUNIT_ASSERT(sq.empty());

  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  Reference* ref = allocate_frame_with_basic_header(I2O_SM_DATA, 0, 1);
  I2OChain c1(ref);
  CPPUNIT_ASSERT(!c1.empty());
  CPPUNIT_ASSERT(c1.complete());
  CPPUNIT_ASSERT_NO_THROW(sq.enqNowait(c1));

  CPPUNIT_ASSERT(sq.size() == 1);
  CPPUNIT_ASSERT(sq.used() == memory_consumed_by_one_frame);
  CPPUNIT_ASSERT(sq.full());

  ref = allocate_frame_with_basic_header(I2O_SM_DATA, 0, 2);
  I2OChain c2(ref);
  CPPUNIT_ASSERT(!c2.empty());
  CPPUNIT_ASSERT(!c2.complete());
  CPPUNIT_ASSERT(outstanding_bytes() == 2*memory_consumed_by_one_frame);
  CPPUNIT_ASSERT_THROW(sq.enqNowait(c2),stor::FailIfFull<I2OChain>::QueueIsFull);
  CPPUNIT_ASSERT(sq.size() == 1);
  CPPUNIT_ASSERT(sq.used() == memory_consumed_by_one_frame);
  CPPUNIT_ASSERT(sq.full());

  I2OChain c3;
  CPPUNIT_ASSERT(c3.empty());
  CPPUNIT_ASSERT(sq.deqNowait(c3));
  CPPUNIT_ASSERT(!c3.empty());
  CPPUNIT_ASSERT(c3.complete());

  CPPUNIT_ASSERT(sq.empty());
  CPPUNIT_ASSERT(sq.used() == 0);
}

// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testStreamQueue);

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

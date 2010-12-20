#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "EventFilter/StorageManager/interface/ConsumerID.h"
#include "EventFilter/StorageManager/interface/EnquingPolicyTag.h"
#include "EventFilter/StorageManager/interface/EventConsumerMonitorCollection.h"
#include "EventFilter/StorageManager/interface/EventConsumerRegistrationInfo.h"
#include "EventFilter/StorageManager/interface/EventQueueCollection.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/QueueID.h"
#include "EventFilter/StorageManager/interface/Utils.h"

#include "EventFilter/StorageManager/test/TestHelper.h"

#include "boost/thread.hpp"
#include "boost/shared_ptr.hpp"
#include "boost/bind.hpp"

#include <algorithm>

using stor::ConsumerID;
using stor::EventQueueCollection;
using stor::EventConsumerRegistrationInfo;
using stor::I2OChain;
using stor::QueueID;
using stor::EventConsRegPtr;
using stor::testhelper::allocate_frame_with_sample_header;
using stor::testhelper::outstanding_bytes;

using stor::enquing_policy::DiscardOld;
using stor::enquing_policy::DiscardNew;
using stor::enquing_policy::FailIfFull;


using std::binary_search;
using std::sort;

class testEventQueueCollection : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testEventQueueCollection);

  CPPUNIT_TEST(create_queues);
  CPPUNIT_TEST(pop_event_from_non_existing_queue);
  CPPUNIT_TEST(add_and_pop);
  CPPUNIT_TEST(invalid_queueid);
  CPPUNIT_TEST(clear_all_queues);
  CPPUNIT_TEST(remove_all_queues);
  CPPUNIT_TEST(shared_queues);

  CPPUNIT_TEST_SUITE_END();

public:
  testEventQueueCollection();

  void setUp();
  void tearDown();

  void create_queues();
  void pop_event_from_non_existing_queue();
  void add_and_pop();
  void invalid_queueid();
  void clear_all_queues();
  void remove_all_queues();
  void shared_queues();

private:
  stor::EventConsumerMonitorCollection _ecmc;
};

testEventQueueCollection::testEventQueueCollection() :
_ecmc(boost::posix_time::seconds(1))
{}

void
testEventQueueCollection::setUp()
{ 
  CPPUNIT_ASSERT(g_factory);
  CPPUNIT_ASSERT(g_alloc);
  CPPUNIT_ASSERT(g_pool);
}

void
testEventQueueCollection::tearDown()
{}

void 
testEventQueueCollection::create_queues()
{
  // Default collection should have no queues.
  EventQueueCollection c(_ecmc);
  CPPUNIT_ASSERT(c.size() == 0);

  // Make sure that the different types of queue are both counted
  // correctly.
  ConsumerID cid1;
  cid1.value = 109;
  EventConsRegPtr ecri1( 
    new EventConsumerRegistrationInfo(
      "cid1", "", Strings(), "", 1, false, 10, DiscardNew, boost::posix_time::seconds(120), ""));
  ecri1->setConsumerId(cid1);
  QueueID id1 = c.createQueue(ecri1);
  CPPUNIT_ASSERT(c.size() == 1);
  CPPUNIT_ASSERT(id1.policy() == DiscardNew);
  CPPUNIT_ASSERT(id1.index() == 0);


  ConsumerID cid2;
  cid2.value = 9234;
  EventConsRegPtr ecri2( 
    new EventConsumerRegistrationInfo(
      "cid2", "", Strings(), "", 1, false, 20, DiscardOld, boost::posix_time::seconds(120), ""));
  ecri2->setConsumerId(cid2);
  QueueID id2 = c.createQueue(ecri2);
  CPPUNIT_ASSERT(c.size() == 2);
  CPPUNIT_ASSERT(id2.policy() == DiscardOld);
  CPPUNIT_ASSERT(id2.index() == 0);

  ConsumerID cid3;
  cid3.value = 2;
  EventConsRegPtr ecri3( 
    new EventConsumerRegistrationInfo(
      "cid3", "", Strings(), "", 1, false, 20, DiscardOld, boost::posix_time::seconds(120), ""));
  ecri3->setConsumerId(cid3);
  QueueID id3 = c.createQueue(ecri3);
  CPPUNIT_ASSERT(c.size() == 3);
  CPPUNIT_ASSERT(id3.index() == 1);

  // Other policies should not allow creation
  ConsumerID cid4;
  cid4.value = 16;
  EventConsRegPtr ecri4( 
    new EventConsumerRegistrationInfo(
      "cid4", "", Strings(), "", 1, false, 1, FailIfFull, boost::posix_time::seconds(120), ""));
  ecri4->setConsumerId(cid4);
  QueueID id4 = c.createQueue(ecri4);
  CPPUNIT_ASSERT(c.size() == 3);
  CPPUNIT_ASSERT(!id4.isValid());

  // An invalid ConsumerID should not allow creation
  ConsumerID invalid;
  invalid.value = 0;
  CPPUNIT_ASSERT(!invalid.isValid());
  ecri3->setConsumerId(invalid);
  QueueID id5 = c.createQueue(ecri3);
  CPPUNIT_ASSERT(c.size() == 3);
  CPPUNIT_ASSERT(!id5.isValid());
}

void 
testEventQueueCollection::pop_event_from_non_existing_queue()
{
  // Attemping to pop and event from a non-existent queue should
  // result in an exception.
  EventQueueCollection c(_ecmc);
  CPPUNIT_ASSERT(c.size() == 0);
  QueueID invalid_id;
  CPPUNIT_ASSERT(!invalid_id.isValid());

  I2OChain chain;
  CPPUNIT_ASSERT(chain.empty());
  CPPUNIT_ASSERT_THROW(chain = c.popEvent(invalid_id), 
                       stor::exception::UnknownQueueId);
  CPPUNIT_ASSERT(chain.empty());
}

void add_and_pop_helper(boost::shared_ptr<EventQueueCollection> pcoll);
void create_queues_helper(boost::shared_ptr<EventQueueCollection> pcoll);

void
testEventQueueCollection::add_and_pop()
{
  using namespace boost;
  shared_ptr<EventQueueCollection> pcoll(new EventQueueCollection(_ecmc));

  boost::thread t1(bind(add_and_pop_helper, pcoll));
  boost::thread t2(bind(create_queues_helper, pcoll));
  boost::thread t3(bind(create_queues_helper, pcoll));
  boost::thread t4(bind(create_queues_helper, pcoll));
  boost::thread t5(bind(create_queues_helper, pcoll));

  t1.join();
  t2.join();
  t3.join();
  t4.join();
  t5.join();
}

void
create_queues_helper(boost::shared_ptr<EventQueueCollection> pcoll)
{
  ConsumerID cid;
  cid.value = 0;
  EventConsRegPtr ecriNew(
    new EventConsumerRegistrationInfo(
      "cid1", "", Strings(), "", 1, false, 10, DiscardNew, boost::posix_time::seconds(120), ""));
  EventConsRegPtr ecriOld(
    new EventConsumerRegistrationInfo(
      "cid1", "", Strings(), "", 1, false, 10, DiscardOld, boost::posix_time::seconds(120), ""));
  for (int i = 0; i < 1000; ++i)
    {
      ::usleep(2000); // 2000 microseconds
      ecriNew->setConsumerId(++cid);
      ecriOld->setConsumerId(++cid);
      pcoll->createQueue(ecriNew);
      pcoll->createQueue(ecriOld);
    }
}

void
add_and_pop_helper(boost::shared_ptr<EventQueueCollection> pcoll)
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  EventQueueCollection& coll = *pcoll;
  // We want events to go bad very rapidly.
  stor::utils::duration_t expiration_interval =  boost::posix_time::seconds(5);

  // Make some queues of each flavor, with very little capacity.  We want
  // them to fill rapidly.
  stor::utils::time_point_t now = stor::utils::getCurrentTime();

  ConsumerID cid;
  EventConsRegPtr ecri1(
    new EventConsumerRegistrationInfo(
      "cid1", "", Strings(), "", 1, false, 2, DiscardOld, expiration_interval, ""));
  ecri1->setConsumerId(++cid);
  QueueID q1 = coll.createQueue(ecri1);
  CPPUNIT_ASSERT(q1.isValid());

  EventConsRegPtr ecri2(
    new EventConsumerRegistrationInfo(
      "cid2", "", Strings(), "", 1, false, 2, DiscardNew, expiration_interval, ""));
  ecri2->setConsumerId(++cid);
  QueueID q2 = coll.createQueue(ecri2);
  CPPUNIT_ASSERT(q2.isValid());

  EventConsRegPtr ecri3(
    new EventConsumerRegistrationInfo(
      "cid3", "", Strings(), "", 1, false, 1, DiscardNew, expiration_interval, ""));
  ecri3->setConsumerId(++cid);
  QueueID q3 = coll.createQueue(ecri3);
  CPPUNIT_ASSERT(q3.isValid());

  // Make some chains, tagging them, and inserting them into the
  // collection. We use many more chains than we have slots in the
  // queues, to make sure that we don't block when the queues are full.
  const int num_chains = 100;

  for (int i = 0; i != num_chains; ++i)
    {
      I2OChain event(allocate_frame_with_sample_header(0,1,1));
      CPPUNIT_ASSERT(event.totalDataSize() != 0);
      unsigned char* payload = event.dataLocation(0);
      CPPUNIT_ASSERT(payload);
      payload[0] = i;      // event now carries an index as data.
      event.tagForEventConsumer(q1);
      if (i % 2 == 0) event.tagForEventConsumer(q2);
      if (i % 3 == 0) event.tagForEventConsumer(q3);
      coll.addEvent(event);
      CPPUNIT_ASSERT(outstanding_bytes() != 0);
    }
  // None of our queues should be empty; all should be full.
  CPPUNIT_ASSERT(!coll.empty(q1));
  CPPUNIT_ASSERT(coll.full(q1));

  CPPUNIT_ASSERT(!coll.empty(q2));
  CPPUNIT_ASSERT(coll.full(q2));

  CPPUNIT_ASSERT(!coll.empty(q3));
  CPPUNIT_ASSERT(coll.full(q3));

  // Queue with id q1 should contain "new" events; q2 and q3 should
  // contain "old" events.
  CPPUNIT_ASSERT(coll.popEvent(q1).dataLocation(0)[0] > num_chains/2);
  CPPUNIT_ASSERT(coll.popEvent(q2).dataLocation(0)[0] < num_chains/2);
  CPPUNIT_ASSERT(coll.popEvent(q3).dataLocation(0)[0] < num_chains/2);

  // Queues 1 and 2 should not be empty (because each contains one
  // event), but q3 should be empty (it has a capacity of one, and we
  // popped that one off).
  CPPUNIT_ASSERT(!coll.empty(q1));
  CPPUNIT_ASSERT(!coll.empty(q2));
  CPPUNIT_ASSERT(coll.empty(q3));
  
  // Now sleep for the expiration interval.
  // Our queues should have all become stale;
  // they should also all be empty.
  stor::utils::sleep(expiration_interval);
  std::vector<QueueID> stale_queues;
  coll.clearStaleQueues(stale_queues);
  //CPPUNIT_ASSERT(stale_queues.size() == coll.size());
  sort(stale_queues.begin(), stale_queues.end());
  CPPUNIT_ASSERT(binary_search(stale_queues.begin(), stale_queues.end(), q1));
  CPPUNIT_ASSERT(binary_search(stale_queues.begin(), stale_queues.end(), q2));
  CPPUNIT_ASSERT(binary_search(stale_queues.begin(), stale_queues.end(), q3));
  CPPUNIT_ASSERT(coll.empty(q1));
  CPPUNIT_ASSERT(coll.empty(q2));
  CPPUNIT_ASSERT(coll.empty(q3));

  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

void
testEventQueueCollection::invalid_queueid()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  EventQueueCollection coll(_ecmc);
  // Make sure none of the interface functions cause a failure. Many
  // do not return any status we can test; we just run the function
  // and observe that we do *not* crash or throw any exception.
  QueueID id1(DiscardNew, 0);
  QueueID id2(DiscardOld, 0);

  coll.setExpirationInterval(id1, boost::posix_time::seconds(2));
  coll.setExpirationInterval(id2, boost::posix_time::seconds(2));

  CPPUNIT_ASSERT(coll.getExpirationInterval(id1) == boost::posix_time::seconds(0));
  CPPUNIT_ASSERT(coll.getExpirationInterval(id2) == boost::posix_time::seconds(0));

  {
    I2OChain event(allocate_frame_with_sample_header(0,1,1));
    event.tagForEventConsumer(id1);
    event.tagForEventConsumer(id2);
    CPPUNIT_ASSERT(!event.empty());
    coll.addEvent(event);
    CPPUNIT_ASSERT(outstanding_bytes() != 0);
  }
  // Trying to pop an event off an nonexistent queue should give an
  // empty event.
  I2OChain event;
  CPPUNIT_ASSERT(event.empty());
  event = coll.popEvent(id1);
  CPPUNIT_ASSERT(event.empty());
  event = coll.popEvent(id2);
  CPPUNIT_ASSERT(event.empty());

  coll.clearQueue(id1); // should have no effect
  coll.clearQueue(id2); // should have no effect
  CPPUNIT_ASSERT(coll.empty(id1)); // nonexistent queue is empty.
  CPPUNIT_ASSERT(coll.full(id1));  // nonexistent queue is also full.
  CPPUNIT_ASSERT(coll.empty(id2)); // nonexistent queue is empty.
  CPPUNIT_ASSERT(coll.full(id2));  // nonexistent queue is also full.

  
  std::vector<QueueID> stale_queues;
  coll.clearStaleQueues(stale_queues);
  CPPUNIT_ASSERT(stale_queues.empty());
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

void
testEventQueueCollection::clear_all_queues()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  EventQueueCollection coll(_ecmc);
  ConsumerID cid;

  EventConsRegPtr ecri1( 
    new EventConsumerRegistrationInfo(
      "cid1", "", Strings(), "", 1, false, 10, DiscardNew, boost::posix_time::seconds(120), ""));
  ecri1->setConsumerId(++cid);
  QueueID q1 = coll.createQueue(ecri1);

  EventConsRegPtr ecri2( 
    new EventConsumerRegistrationInfo(
      "cid2", "", Strings(), "", 1, false, 10, DiscardOld, boost::posix_time::seconds(120), ""));
  ecri2->setConsumerId(++cid);
  QueueID q2 = coll.createQueue(ecri2);

  EventConsRegPtr ecri3( 
    new EventConsumerRegistrationInfo(
      "cid3", "", Strings(), "", 1, false, 10, DiscardOld, boost::posix_time::seconds(120), ""));
  ecri3->setConsumerId(++cid);
  QueueID q3 = coll.createQueue(ecri3);

  EventConsRegPtr ecri4( 
    new EventConsumerRegistrationInfo(
      "cid4", "", Strings(), "", 1, false, 10, DiscardNew, boost::posix_time::seconds(120), ""));
  ecri4->setConsumerId(++cid);
  QueueID q4 = coll.createQueue(ecri4);

  CPPUNIT_ASSERT(coll.size() == 4);
  
  for (int i = 0; i < 100; ++i)
    {
      I2OChain event(allocate_frame_with_sample_header(0,1,1));
      event.tagForEventConsumer(q1);
      if (i%2 == 0) event.tagForEventConsumer(q2);
      if (i%3 == 0) event.tagForEventConsumer(q3);
      if (i%4 == 0) event.tagForEventConsumer(q4);
      coll.addEvent(event);
      CPPUNIT_ASSERT(outstanding_bytes() != 0);
    }
  CPPUNIT_ASSERT(!coll.empty(q1));
  CPPUNIT_ASSERT(!coll.empty(q2));
  CPPUNIT_ASSERT(!coll.empty(q3));
  CPPUNIT_ASSERT(!coll.empty(q4));

  CPPUNIT_ASSERT(!coll.popEvent(cid).empty());
  
  coll.clearQueues();
  CPPUNIT_ASSERT(coll.size() == 4);
  CPPUNIT_ASSERT(coll.empty(q1));
  CPPUNIT_ASSERT(coll.empty(q2));
  CPPUNIT_ASSERT(coll.empty(q3));
  CPPUNIT_ASSERT(coll.empty(q4));
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

void
testEventQueueCollection::remove_all_queues()
{
  EventQueueCollection coll(_ecmc);
  ConsumerID cid;

  EventConsRegPtr ecri1( 
    new EventConsumerRegistrationInfo(
      "cid1", "", Strings(), "", 1, false, 10, DiscardNew, boost::posix_time::seconds(120), ""));
  ecri1->setConsumerId(++cid);
  QueueID q1 = coll.createQueue(ecri1);

  EventConsRegPtr ecri2( 
    new EventConsumerRegistrationInfo(
      "cid2", "", Strings(), "", 1, false, 10, DiscardOld, boost::posix_time::seconds(120), ""));
  ecri2->setConsumerId(++cid);
  QueueID q2 = coll.createQueue(ecri2);

  EventConsRegPtr ecri3( 
    new EventConsumerRegistrationInfo(
      "cid3", "", Strings(), "", 1, false, 10, DiscardOld, boost::posix_time::seconds(120), ""));
  ecri3->setConsumerId(++cid);
  QueueID q3 = coll.createQueue(ecri3);

  EventConsRegPtr ecri4( 
    new EventConsumerRegistrationInfo(
      "cid4", "", Strings(), "", 1, false, 10, DiscardNew, boost::posix_time::seconds(120), ""));
  ecri4->setConsumerId(++cid);
  QueueID q4 = coll.createQueue(ecri4);

  CPPUNIT_ASSERT(coll.size() == 4);
  coll.removeQueues();
  CPPUNIT_ASSERT(coll.size() == 0);  
}


void
testEventQueueCollection::shared_queues()
{
  EventQueueCollection coll(_ecmc);
  ConsumerID cid;

  EventConsRegPtr ecri1( 
    new EventConsumerRegistrationInfo(
      "cid1", "", Strings(), "", 1, true, 10, DiscardNew, boost::posix_time::seconds(120), ""));
  ecri1->setConsumerId(++cid);
  QueueID q1 = coll.createQueue(ecri1);
  CPPUNIT_ASSERT(coll.size() == 1);

  // different policy
  EventConsRegPtr ecri2( 
    new EventConsumerRegistrationInfo(
      "cid2", "", Strings(), "", 1, true, 10, DiscardOld, boost::posix_time::seconds(120), ""));
  ecri2->setConsumerId(++cid);
  QueueID q2 = coll.createQueue(ecri2);
  CPPUNIT_ASSERT(coll.size() == 2);
  CPPUNIT_ASSERT(q1 != q2);

  // shared with q1
  EventConsRegPtr ecri3( 
    new EventConsumerRegistrationInfo(
      "cid3", "", Strings(), "", 1, true, 10, DiscardNew, boost::posix_time::seconds(120), ""));
  ecri3->setConsumerId(++cid);
  QueueID q3 = coll.createQueue(ecri3);
  CPPUNIT_ASSERT(coll.size() == 2);
  CPPUNIT_ASSERT(q1 == q3);

  // shared with q2
  EventConsRegPtr ecri4( 
    new EventConsumerRegistrationInfo(
      "cid4", "", Strings(), "", 1, true, 10, DiscardOld, boost::posix_time::seconds(120), ""));
  ecri4->setConsumerId(++cid);
  QueueID q4 = coll.createQueue(ecri4);
  CPPUNIT_ASSERT(coll.size() == 2);
  CPPUNIT_ASSERT(q2 == q4);

  // different size
  EventConsRegPtr ecri5( 
    new EventConsumerRegistrationInfo(
      "cid5", "", Strings(), "", 1, true, 20, DiscardNew, boost::posix_time::seconds(120), ""));
  ecri5->setConsumerId(++cid);
  QueueID q5 = coll.createQueue(ecri5);
  CPPUNIT_ASSERT(coll.size() == 3);
  CPPUNIT_ASSERT(q4 != q5);

  // different timeout
  EventConsRegPtr ecri6( 
    new EventConsumerRegistrationInfo(
      "cid6", "", Strings(), "", 1, true, 10, DiscardNew, boost::posix_time::seconds(20), ""));
  ecri6->setConsumerId(++cid);
  QueueID q6 = coll.createQueue(ecri6);
  CPPUNIT_ASSERT(coll.size() == 4);
  CPPUNIT_ASSERT(q4 != q6);

  // same as queue q5
  EventConsRegPtr ecri7( 
    new EventConsumerRegistrationInfo(
      "cid7", "", Strings(), "", 1, true, 20, DiscardNew, boost::posix_time::seconds(120), ""));
  ecri7->setConsumerId(++cid);
  QueueID q7 = coll.createQueue(ecri7);
  CPPUNIT_ASSERT(coll.size() == 4);
  CPPUNIT_ASSERT(q5 == q7);

  // different prescale
  EventConsRegPtr ecri8( 
    new EventConsumerRegistrationInfo(
      "cid8", "", Strings(), "", 10, true, 20, DiscardNew, boost::posix_time::seconds(120), ""));
  ecri8->setConsumerId(++cid);
  QueueID q8 = coll.createQueue(ecri8);
  CPPUNIT_ASSERT(coll.size() == 5);
  CPPUNIT_ASSERT(q7 != q8);

  coll.removeQueues();
  CPPUNIT_ASSERT(coll.size() == 0);  
}


// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testEventQueueCollection);

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

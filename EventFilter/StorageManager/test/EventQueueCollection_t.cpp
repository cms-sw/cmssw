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
  stor::EventConsumerMonitorCollection ecmc_;
};

testEventQueueCollection::testEventQueueCollection() :
ecmc_(boost::posix_time::seconds(1))
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
  EventQueueCollection c(ecmc_);
  CPPUNIT_ASSERT(c.size() == 0);

  // Make sure that the different types of queue are both counted
  // correctly.
  ConsumerID cid1;
  cid1.value = 109;
  edm::ParameterSet pset;
  pset.addUntrackedParameter<std::string>("SelectHLTOutput", "hltOutputDQM");
  pset.addUntrackedParameter<std::string>("queuePolicy", "DiscardNew");
  EventConsRegPtr ecri1(new EventConsumerRegistrationInfo(pset));
  ecri1->setConsumerId(cid1);
  QueueID id1 = c.createQueue(ecri1);
  CPPUNIT_ASSERT(c.size() == 1);
  CPPUNIT_ASSERT(id1.policy() == DiscardNew);
  CPPUNIT_ASSERT(id1.index() == 0);


  ConsumerID cid2;
  cid2.value = 9234;
  pset.addUntrackedParameter<std::string>("queuePolicy", "DiscardOld");
  EventConsRegPtr ecri2(new EventConsumerRegistrationInfo(pset));
  ecri2->setConsumerId(cid2);
  QueueID id2 = c.createQueue(ecri2);
  CPPUNIT_ASSERT(c.size() == 2);
  CPPUNIT_ASSERT(id2.policy() == DiscardOld);
  CPPUNIT_ASSERT(id2.index() == 0);

  ConsumerID cid3;
  cid3.value = 2;
  EventConsRegPtr ecri3(new EventConsumerRegistrationInfo(pset));
  ecri3->setConsumerId(cid3);
  QueueID id3 = c.createQueue(ecri3);
  CPPUNIT_ASSERT(c.size() == 3);
  CPPUNIT_ASSERT(id3.index() == 1);

  // Other policies should not allow creation
  ConsumerID cid4;
  cid4.value = 16;
  edm::ParameterSet pset2;
  pset2.addUntrackedParameter<std::string>("SelectHLTOutput", "hltOutputDQM");
  EventConsRegPtr ecri4(new EventConsumerRegistrationInfo(pset2));
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
  EventQueueCollection c(ecmc_);
  CPPUNIT_ASSERT(c.size() == 0);
  QueueID invalid_id;
  CPPUNIT_ASSERT(!invalid_id.isValid());

  EventQueueCollection::ValueType event;
  CPPUNIT_ASSERT(event.first.empty());
  CPPUNIT_ASSERT_THROW(event = c.popEvent(invalid_id), 
                       stor::exception::UnknownQueueId);
  CPPUNIT_ASSERT(event.first.empty());
}

void
testEventQueueCollection::add_and_pop()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  EventQueueCollection coll(ecmc_);

  // We want events to go bad very rapidly.
  double expiration_interval = 5;

  // Make some queues of each flavor, with very little capacity.  We want
  // them to fill rapidly.
  stor::utils::TimePoint_t now = stor::utils::getCurrentTime();

  ConsumerID cid;
  edm::ParameterSet pset;
  pset.addUntrackedParameter<std::string>("SelectHLTOutput", "hltOutputDQM");
  pset.addUntrackedParameter<std::string>("queuePolicy", "DiscardOld");
  pset.addUntrackedParameter<double>("consumerTimeOut", expiration_interval);
  pset.addUntrackedParameter<int>("queueSize", 2);

  EventConsRegPtr ecri1(new EventConsumerRegistrationInfo(pset));
  ecri1->setConsumerId(++cid);
  QueueID q1 = coll.createQueue(ecri1);
  CPPUNIT_ASSERT(q1.isValid());

  pset.addUntrackedParameter<std::string>("queuePolicy", "DiscardNew");
  EventConsRegPtr ecri2(new EventConsumerRegistrationInfo(pset));
  ecri2->setConsumerId(++cid);
  QueueID q2 = coll.createQueue(ecri2);
  CPPUNIT_ASSERT(q2.isValid());

  pset.addUntrackedParameter<int>("queueSize", 1);
  EventConsRegPtr ecri3(new EventConsumerRegistrationInfo(pset));
  ecri3->setConsumerId(++cid);
  QueueID q3 = coll.createQueue(ecri3);
  CPPUNIT_ASSERT(q3.isValid());

  CPPUNIT_ASSERT(coll.size() == 3);

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
  CPPUNIT_ASSERT(coll.popEvent(q1).first.dataLocation(0)[0] > num_chains/2);
  CPPUNIT_ASSERT(coll.popEvent(q2).first.dataLocation(0)[0] < num_chains/2);
  CPPUNIT_ASSERT(coll.popEvent(q3).first.dataLocation(0)[0] < num_chains/2);

  // Queues 1 and 2 should not be empty (because each contains one
  // event), but q3 should be empty (it has a capacity of one, and we
  // popped that one off).
  CPPUNIT_ASSERT(!coll.empty(q1));
  CPPUNIT_ASSERT(!coll.empty(q2));
  CPPUNIT_ASSERT(coll.empty(q3));

  // Queues should not be cleared, as not stale, yet
  CPPUNIT_ASSERT(!coll.clearStaleQueues(now));
  CPPUNIT_ASSERT(!coll.stale(q1,now));
  CPPUNIT_ASSERT(!coll.stale(q2,now));
  CPPUNIT_ASSERT(!coll.stale(q3,now));
  CPPUNIT_ASSERT(!coll.allQueuesStale(now));
  CPPUNIT_ASSERT(!coll.empty(q1));
  CPPUNIT_ASSERT(!coll.empty(q2));
  CPPUNIT_ASSERT(coll.empty(q3));

  // Now sleep for the expiration interval.
  // Our queues should have all become stale;
  // they should also all be empty.
  ::sleep(expiration_interval);
  now = stor::utils::getCurrentTime();
  CPPUNIT_ASSERT(coll.stale(q1,now));
  CPPUNIT_ASSERT(coll.stale(q2,now));
  CPPUNIT_ASSERT(coll.stale(q3,now));
  CPPUNIT_ASSERT(coll.allQueuesStale(now));
  CPPUNIT_ASSERT(coll.clearStaleQueues(now));
  CPPUNIT_ASSERT(coll.empty(q1));
  CPPUNIT_ASSERT(coll.empty(q2));
  CPPUNIT_ASSERT(coll.empty(q3));

  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

void
testEventQueueCollection::invalid_queueid()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  EventQueueCollection coll(ecmc_);

  QueueID id1(DiscardNew, 0);
  QueueID id2(DiscardOld, 0);
  
  CPPUNIT_ASSERT_THROW(coll.setExpirationInterval(id1, boost::posix_time::seconds(2)),
    stor::exception::UnknownQueueId);
  CPPUNIT_ASSERT_THROW(coll.setExpirationInterval(id2, boost::posix_time::seconds(2)),
    stor::exception::UnknownQueueId);
 
  CPPUNIT_ASSERT_THROW(coll.getExpirationInterval(id1), stor::exception::UnknownQueueId);
  CPPUNIT_ASSERT_THROW(coll.getExpirationInterval(id2), stor::exception::UnknownQueueId);

  {
    I2OChain event(allocate_frame_with_sample_header(0,1,1));
    event.tagForEventConsumer(id1);
    event.tagForEventConsumer(id2);
    CPPUNIT_ASSERT(!event.empty());
    CPPUNIT_ASSERT_THROW(coll.addEvent(event), stor::exception::UnknownQueueId);
    CPPUNIT_ASSERT(outstanding_bytes() != 0);
  }
  // Trying to pop an event off an nonexistent queue should give an
  // empty event.
  EventQueueCollection::ValueType event;
  CPPUNIT_ASSERT(event.first.empty());
  CPPUNIT_ASSERT_THROW(event = coll.popEvent(id1), stor::exception::UnknownQueueId);
  CPPUNIT_ASSERT(event.first.empty());
  CPPUNIT_ASSERT_THROW(event = coll.popEvent(id2), stor::exception::UnknownQueueId);
  CPPUNIT_ASSERT(event.first.empty());

  CPPUNIT_ASSERT_THROW(coll.clearQueue(id1), stor::exception::UnknownQueueId);
  CPPUNIT_ASSERT_THROW(coll.clearQueue(id2), stor::exception::UnknownQueueId);
  CPPUNIT_ASSERT_THROW(coll.empty(id1), stor::exception::UnknownQueueId);
  CPPUNIT_ASSERT_THROW(coll.full(id1), stor::exception::UnknownQueueId);
  CPPUNIT_ASSERT_THROW(coll.empty(id2), stor::exception::UnknownQueueId);
  CPPUNIT_ASSERT_THROW(coll.full(id2), stor::exception::UnknownQueueId);
  
  stor::utils::TimePoint_t now = stor::utils::getCurrentTime();
  CPPUNIT_ASSERT(!coll.clearStaleQueues(now));
  CPPUNIT_ASSERT_THROW(coll.stale(id1,now), stor::exception::UnknownQueueId);
  CPPUNIT_ASSERT_THROW(coll.stale(id2,now), stor::exception::UnknownQueueId);
  CPPUNIT_ASSERT(coll.allQueuesStale(now));

  CPPUNIT_ASSERT(outstanding_bytes() == 0);
}

void
testEventQueueCollection::clear_all_queues()
{
  CPPUNIT_ASSERT(outstanding_bytes() == 0);
  EventQueueCollection coll(ecmc_);
  ConsumerID cid;

  edm::ParameterSet pset;
  pset.addUntrackedParameter<std::string>("SelectHLTOutput", "hltOutputDQM");
  pset.addUntrackedParameter<std::string>("queuePolicy", "DiscardNew");
  pset.addUntrackedParameter<double>("consumerTimeOut", 120);
  pset.addUntrackedParameter<int>("queueSize", 10);

  EventConsRegPtr ecri1(new EventConsumerRegistrationInfo(pset));
  ecri1->setConsumerId(++cid);
  QueueID q1 = coll.createQueue(ecri1);

  pset.addUntrackedParameter<std::string>("queuePolicy", "DiscardOld");
  EventConsRegPtr ecri2(new EventConsumerRegistrationInfo(pset));
  ecri2->setConsumerId(++cid);
  QueueID q2 = coll.createQueue(ecri2);

  EventConsRegPtr ecri3(new EventConsumerRegistrationInfo(pset));
  ecri3->setConsumerId(++cid);
  QueueID q3 = coll.createQueue(ecri3);

  pset.addUntrackedParameter<std::string>("queuePolicy", "DiscardNew");
  EventConsRegPtr ecri4(new EventConsumerRegistrationInfo(pset));
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

  CPPUNIT_ASSERT(!coll.popEvent(cid).first.empty());
  
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
  EventQueueCollection coll(ecmc_);
  ConsumerID cid;

  edm::ParameterSet pset;
  pset.addUntrackedParameter<std::string>("SelectHLTOutput", "hltOutputDQM");
  pset.addUntrackedParameter<std::string>("queuePolicy", "DiscardNew");
  pset.addUntrackedParameter<double>("consumerTimeOut", 120);
  pset.addUntrackedParameter<int>("queueSize", 10);

  EventConsRegPtr ecri1(new EventConsumerRegistrationInfo(pset));
  ecri1->setConsumerId(++cid);
  QueueID q1 = coll.createQueue(ecri1);

  pset.addUntrackedParameter<std::string>("queuePolicy", "DiscardOld");
  EventConsRegPtr ecri2(new EventConsumerRegistrationInfo(pset));
  ecri2->setConsumerId(++cid);
  QueueID q2 = coll.createQueue(ecri2);

  EventConsRegPtr ecri3(new EventConsumerRegistrationInfo(pset));
  ecri3->setConsumerId(++cid);
  QueueID q3 = coll.createQueue(ecri3);

  pset.addUntrackedParameter<std::string>("queuePolicy", "DiscardNew");
  EventConsRegPtr ecri4(new EventConsumerRegistrationInfo(pset));
  ecri4->setConsumerId(++cid);
  QueueID q4 = coll.createQueue(ecri4);

  CPPUNIT_ASSERT(coll.size() == 4);
  coll.removeQueues();
  CPPUNIT_ASSERT(coll.size() == 0);  
}


void
testEventQueueCollection::shared_queues()
{
  EventQueueCollection coll(ecmc_);
  ConsumerID cid;

  edm::ParameterSet pset1;
  pset1.addUntrackedParameter<std::string>("SelectHLTOutput", "hltOutputDQM");
  pset1.addUntrackedParameter<std::string>("consumerName", "Consumer 1");
  pset1.addUntrackedParameter<std::string>("queuePolicy", "DiscardNew");
  pset1.addUntrackedParameter<double>("consumerTimeOut", 120);
  pset1.addUntrackedParameter<int>("queueSize", 10);
  pset1.addUntrackedParameter<bool>("uniqueEvents", true);

  EventConsRegPtr ecri1(new EventConsumerRegistrationInfo(pset1));
  ecri1->setConsumerId(++cid);
  QueueID q1 = coll.createQueue(ecri1);
  CPPUNIT_ASSERT(coll.size() == 1);

  // different policy
  edm::ParameterSet pset2 = pset1;
  pset2.addUntrackedParameter<std::string>("consumerName", "Consumer 2");
  pset2.addUntrackedParameter<std::string>("queuePolicy", "DiscardOld");
  EventConsRegPtr ecri2(new EventConsumerRegistrationInfo(pset2));
  ecri2->setConsumerId(++cid);
  QueueID q2 = coll.createQueue(ecri2);
  CPPUNIT_ASSERT(coll.size() == 2);
  CPPUNIT_ASSERT(q1 != q2);

  // shared with q1
  edm::ParameterSet pset3 = pset1;
  pset3.addUntrackedParameter<std::string>("consumerName", "Consumer 3");
  EventConsRegPtr ecri3(new EventConsumerRegistrationInfo(pset3));
  ecri3->setConsumerId(++cid);
  QueueID q3 = coll.createQueue(ecri3);
  CPPUNIT_ASSERT(coll.size() == 2);
  CPPUNIT_ASSERT(q1 == q3);

  // shared with q2
  edm::ParameterSet pset4 = pset2;
  pset4.addUntrackedParameter<std::string>("consumerName", "Consumer 4");
  EventConsRegPtr ecri4(new EventConsumerRegistrationInfo(pset4));
  ecri4->setConsumerId(++cid);
  QueueID q4 = coll.createQueue(ecri4);
  CPPUNIT_ASSERT(coll.size() == 2);
  CPPUNIT_ASSERT(q2 == q4);

  // different size
  edm::ParameterSet pset5 = pset4;
  pset5.addUntrackedParameter<std::string>("consumerName", "Consumer 5");
  pset5.addUntrackedParameter<int>("queueSize", 20);
  EventConsRegPtr ecri5(new EventConsumerRegistrationInfo(pset5));
  ecri5->setConsumerId(++cid);
  QueueID q5 = coll.createQueue(ecri5);
  CPPUNIT_ASSERT(coll.size() == 3);
  CPPUNIT_ASSERT(q4 != q5);

  // different timeout
  edm::ParameterSet pset6 = pset4;
  pset6.addUntrackedParameter<std::string>("consumerName", "Consumer 6");
  pset6.addUntrackedParameter<double>("consumerTimeOut", 20);
  EventConsRegPtr ecri6(new EventConsumerRegistrationInfo(pset6));
  ecri6->setConsumerId(++cid);
  QueueID q6 = coll.createQueue(ecri6);
  CPPUNIT_ASSERT(coll.size() == 4);
  CPPUNIT_ASSERT(q4 != q6);

  // same as queue q5
  edm::ParameterSet pset7 = pset5;
  pset7.addUntrackedParameter<std::string>("consumerName", "Consumer 7");
  EventConsRegPtr ecri7(new EventConsumerRegistrationInfo(pset7));
  ecri7->setConsumerId(++cid);
  QueueID q7 = coll.createQueue(ecri7);
  CPPUNIT_ASSERT(coll.size() == 4);
  CPPUNIT_ASSERT(q5 == q7);

  // different prescale
  edm::ParameterSet pset8 = pset7;
  pset8.addUntrackedParameter<std::string>("consumerName", "Consumer 8");
  pset8.addUntrackedParameter<int>("prescale", 10);
  EventConsRegPtr ecri8(new EventConsumerRegistrationInfo(pset8));
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

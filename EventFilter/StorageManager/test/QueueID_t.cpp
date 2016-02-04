#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "EventFilter/StorageManager/interface/QueueID.h"

using stor::QueueID;


class testQueueID : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testQueueID);

  CPPUNIT_TEST(default_queueid);
  CPPUNIT_TEST(all_policies);
  CPPUNIT_TEST(sorting);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp();
  void tearDown();

  void default_queueid();
  void all_policies();
  void sorting();

private:

};

void
testQueueID::setUp()
{ 
}

void
testQueueID::tearDown()
{ 
}

void 
testQueueID::default_queueid()
{
  QueueID id1;
  QueueID id2;
  CPPUNIT_ASSERT( !id1.isValid() );
  CPPUNIT_ASSERT(id1 == id2);
  CPPUNIT_ASSERT( ! (id1 != id2) );
  CPPUNIT_ASSERT(! (id1 < id1) );
  CPPUNIT_ASSERT(! (id1 < id2) );
  CPPUNIT_ASSERT(! (id2 < id1) );

  QueueID invalid1(stor::enquing_policy::Max, 0UL);
  CPPUNIT_ASSERT(invalid1 == id1);
}

void
testQueueID::all_policies()
{
  QueueID dnew(stor::enquing_policy::DiscardNew, 10UL);
  CPPUNIT_ASSERT(dnew.isValid());
  CPPUNIT_ASSERT(dnew.policy() == stor::enquing_policy::DiscardNew);
  CPPUNIT_ASSERT(dnew.index() == 10UL);

  QueueID dold(stor::enquing_policy::DiscardOld, 20UL);
  CPPUNIT_ASSERT(dold.isValid());
  CPPUNIT_ASSERT(dold.policy() == stor::enquing_policy::DiscardOld);
  CPPUNIT_ASSERT(dold.index() == 20UL);

  QueueID ff(stor::enquing_policy::FailIfFull,   30UL);
  CPPUNIT_ASSERT(ff.isValid());
  CPPUNIT_ASSERT(ff.policy() == stor::enquing_policy::FailIfFull);
  CPPUNIT_ASSERT(ff.index() == 30UL);

  QueueID invalid1(stor::enquing_policy::Max, 0UL);
  CPPUNIT_ASSERT(invalid1.policy() == stor::enquing_policy::Max);
  CPPUNIT_ASSERT(invalid1.index() == 0UL);

  QueueID invalid2(stor::enquing_policy::Max, 10UL);
  CPPUNIT_ASSERT(invalid1 != invalid2);
  CPPUNIT_ASSERT(! invalid1.isValid() );
  CPPUNIT_ASSERT(! invalid2.isValid() );
}

void
testQueueID::sorting()
{
  using namespace stor;
  // Test sorting of items with the same index.
  CPPUNIT_ASSERT( QueueID(enquing_policy::DiscardNew, 5UL) <
                  QueueID(enquing_policy::DiscardOld, 5UL) );

  CPPUNIT_ASSERT( QueueID(enquing_policy::DiscardNew, 5UL) <
                  QueueID(enquing_policy::FailIfFull, 5UL) );

  CPPUNIT_ASSERT( QueueID(enquing_policy::DiscardNew, 5UL) <
                  QueueID(enquing_policy::Max, 5UL) );

  CPPUNIT_ASSERT( QueueID(enquing_policy::DiscardOld, 5UL) <
                  QueueID(enquing_policy::FailIfFull, 5UL) );

  CPPUNIT_ASSERT( QueueID(enquing_policy::DiscardOld, 5UL) <
                  QueueID(enquing_policy::Max, 5UL) );

  CPPUNIT_ASSERT( QueueID(enquing_policy::FailIfFull, 5UL) <
                  QueueID(enquing_policy::Max, 5UL) );

  // Test sorting of items with the same policy.
  CPPUNIT_ASSERT (QueueID(enquing_policy::DiscardOld, 0UL) <
                  QueueID(enquing_policy::DiscardOld, 1UL) );

  CPPUNIT_ASSERT (QueueID(enquing_policy::DiscardNew, 0UL) <
                  QueueID(enquing_policy::DiscardNew, 1UL) );

  CPPUNIT_ASSERT (QueueID(enquing_policy::FailIfFull, 0UL) <
                  QueueID(enquing_policy::FailIfFull, 1UL) );

  CPPUNIT_ASSERT (QueueID(enquing_policy::Max, 0UL) <
                  QueueID(enquing_policy::Max, 1UL) );

                  
}


// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testQueueID);


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

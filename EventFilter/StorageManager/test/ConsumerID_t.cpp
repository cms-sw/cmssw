#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "EventFilter/StorageManager/interface/ConsumerID.h"

#include <sstream>
#include <string>

using stor::ConsumerID;
using namespace std;

class testConsumerID : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testConsumerID);
  CPPUNIT_TEST(invalid_id);
  CPPUNIT_TEST(valid_id);
  CPPUNIT_TEST(output);
  CPPUNIT_TEST(preincrement);
  CPPUNIT_TEST(postincrement);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp();
  void tearDown();

  void invalid_id();
  void valid_id();
  void output();
  void preincrement();
  void postincrement();

private:
  // No data members yet.
};

void
testConsumerID::setUp()
{ 
}

void
testConsumerID::tearDown()
{ 
}


void 
testConsumerID::invalid_id()
{
  ConsumerID invalid;
  CPPUNIT_ASSERT(!invalid.isValid());
}

void 
testConsumerID::valid_id()
{
  ConsumerID valid(3);
  CPPUNIT_ASSERT(valid.isValid());
}

void
testConsumerID::output()
{
  ostringstream out;
  ConsumerID id(18);
  out << id << std::endl;
  CPPUNIT_ASSERT(out.str() == "18\n");
}

void
testConsumerID::preincrement()
{
  ConsumerID id;
  CPPUNIT_ASSERT((++id).value == 1);
  CPPUNIT_ASSERT(id.value == 1);
  CPPUNIT_ASSERT((++id).value == 2);
  CPPUNIT_ASSERT(id.value == 2);
}


void
testConsumerID::postincrement()
{
  ConsumerID id;
  id++;
  CPPUNIT_ASSERT(id.value == 1);
  ConsumerID other = id++;
  CPPUNIT_ASSERT(id.value == 2);
  CPPUNIT_ASSERT(other.value == 1);  
}


// This macro writes the 'main' for this test.
CPPUNIT_TEST_SUITE_REGISTRATION(testConsumerID);


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

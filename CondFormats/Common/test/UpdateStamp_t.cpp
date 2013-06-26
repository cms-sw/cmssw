#include <cppunit/extensions/HelperMacros.h>

//#include "boost/intrusive_ptr.hpp"

#define private public
#include "CondFormats/Common/interface/UpdateStamp.h"
#undef private
#include "CondFormats/Common/interface/TimeConversions.h"

namespace {

   class Test: public CppUnit::TestFixture
   {
	 CPPUNIT_TEST_SUITE(Test);
	 CPPUNIT_TEST(construct);
	 CPPUNIT_TEST(stamp);
	 CPPUNIT_TEST_SUITE_END();
      public:
	 void setUp(){}
	 void tearDown(){}
	 void construct();
	 void stamp();

   };

  void Test::construct() {
    cond::UpdateStamp object;
    CPPUNIT_ASSERT(-1==object.m_revision);
    CPPUNIT_ASSERT(0==object.m_timestamp);
    CPPUNIT_ASSERT("not stamped"==object.m_comment);
  }

  void Test::stamp() {
    cond::UpdateStamp object;
    cond::Time_t otime = cond::time::now();
    {
      cond::Time_t btime = cond::time::now();
      object.stamp("V0");
      cond::Time_t atime = cond::time::now();
      CPPUNIT_ASSERT(0==object.m_revision);
      CPPUNIT_ASSERT(object.m_timestamp>=otime);
      CPPUNIT_ASSERT(object.m_timestamp>=btime);
      CPPUNIT_ASSERT(atime>=object.m_timestamp);
      CPPUNIT_ASSERT("V0"==object.m_comment);
    }
    {
      cond::Time_t btime = cond::time::now();
      object.stamp("V1");
      cond::Time_t atime = cond::time::now();
      CPPUNIT_ASSERT(1==object.m_revision);
      CPPUNIT_ASSERT(object.m_timestamp>=otime);
      CPPUNIT_ASSERT(object.m_timestamp>=btime);
      CPPUNIT_ASSERT(atime>=object.m_timestamp);
      CPPUNIT_ASSERT("V1"==object.m_comment);
    }



  }

}

CPPUNIT_TEST_SUITE_REGISTRATION(Test);
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"




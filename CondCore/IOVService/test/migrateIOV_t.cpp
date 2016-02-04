#include <cppunit/extensions/HelperMacros.h>

#define private public
#include "CondFormats/Common/interface/IOVSequence.h"
#include "CondCore/IOVService/src/IOV.h"
#undef private

#include "CondCore/IOVService/interface/migrateIOV.h"

namespace {

   class Test: public CppUnit::TestFixture
   {
     CPPUNIT_TEST_SUITE(Test);
     CPPUNIT_TEST(migrate);
     CPPUNIT_TEST(backport);
     CPPUNIT_TEST_SUITE_END();
   public:
     Test();
     void setUp(){}
     void tearDown(){}
     void migrate();
     void backport();

     cond::IOV m_iov;
     cond::IOVSequence m_sequence;

   };

  Test::Test() :
    m_iov(0,1),m_sequence(0,40,"")
  {
    m_iov.add(10,"a");
    m_iov.add(20,"b");
    m_iov.add(30,"c");
    m_iov.add(40,"d");
    //
    m_sequence.add(1,"a");
    m_sequence.add(11,"b");
    m_sequence.add(21,"c");
    m_sequence.add(31,"d");
  }

  void Test::migrate() {
    cond::IOVSequence * sequence = migrateIOV(m_iov);
    CPPUNIT_ASSERT((*sequence).m_timetype==m_sequence.m_timetype);
    CPPUNIT_ASSERT((*sequence).m_lastTill==m_sequence.m_lastTill);
    CPPUNIT_ASSERT((*sequence).m_notOrdered==m_sequence.m_notOrdered);
    CPPUNIT_ASSERT((*sequence).m_metadata==m_sequence.m_metadata);
    CPPUNIT_ASSERT((*sequence).iovs()==m_sequence.iovs());
    delete sequence;
  }

  void Test::backport() {
    cond::IOV * iov = backportIOV(m_sequence);
    CPPUNIT_ASSERT((*iov).timetype==m_iov.timetype);
    CPPUNIT_ASSERT((*iov).firstsince==m_iov.firstsince);
    CPPUNIT_ASSERT((*iov).iov==m_iov.iov);
    delete iov;
  }

}

CPPUNIT_TEST_SUITE_REGISTRATION(Test);
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"



 

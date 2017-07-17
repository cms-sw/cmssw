/*----------------------------------------------------------------------

Test program for edm::TypeID class.
Changed by Viji on 29-06-2005

 ----------------------------------------------------------------------*/

#include <cassert>
#include <cppunit/extensions/HelperMacros.h>
#include "FWCore/Utilities/interface/CallOnceNoWait.h"
#include "FWCore/Utilities/interface/CallNTimesNoWait.h"

#include <thread>
#include <atomic>

class testCallXNoWait: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testCallXNoWait);
  
  CPPUNIT_TEST(onceTest);
  CPPUNIT_TEST(nTimesTest);
  CPPUNIT_TEST(onceThreadedTest);
  CPPUNIT_TEST(nTimesThreadedTest);
  
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}
  
  void onceTest();
  void nTimesTest();
  
  void onceThreadedTest();
  void nTimesThreadedTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testCallXNoWait);

void testCallXNoWait::onceTest()
{

  edm::CallOnceNoWait guard;
  
  unsigned int iCount=0;
  
  guard([&iCount](){ ++iCount; });
  
  CPPUNIT_ASSERT(iCount == 1);

  guard([&iCount](){ ++iCount; });
  CPPUNIT_ASSERT(iCount == 1);

}

void testCallXNoWait::nTimesTest()
{
  edm::CallNTimesNoWait guard{3};
  
  unsigned int iCount=0;
  
  for(unsigned int i=0; i<6; ++i) {
    guard([&iCount](){ ++iCount; });
    if(i<3) {
      CPPUNIT_ASSERT(iCount == i+1);
    } else {
      CPPUNIT_ASSERT(iCount == 3);
    }
  }
}


void testCallXNoWait::onceThreadedTest()
{
  
  edm::CallOnceNoWait guard;
  
  std::atomic<unsigned int> iCount{0};
  
  std::vector<std::thread> threads;
  
  std::atomic<bool> start{false};

  for(unsigned int i=0; i<4; ++i) {
    threads.emplace_back([&guard,&iCount,&start](){
      while(not start) {}
      guard([&iCount](){ ++iCount; });
    });
  }
  CPPUNIT_ASSERT(iCount == 0);
  
  start = true;
  for(auto& t : threads) {
    t.join();
  }
  CPPUNIT_ASSERT(iCount == 1);
}


void testCallXNoWait::nTimesThreadedTest()
{
  const unsigned short kMaxTimes=3;
  edm::CallNTimesNoWait guard(kMaxTimes);
  
  std::atomic<unsigned int> iCount{0};
  
  std::vector<std::thread> threads;
  
  std::atomic<bool> start{false};
  
  for(unsigned int i=0; i<2*kMaxTimes; ++i) {
    threads.emplace_back([&guard,&iCount,&start](){
      while(not start) {}
      guard([&iCount](){ ++iCount; });
    });
  }
  CPPUNIT_ASSERT(iCount == 0);
  
  start = true;
  for(auto& t : threads) {
    t.join();
  }
  CPPUNIT_ASSERT(iCount == kMaxTimes);
}


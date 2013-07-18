/*----------------------------------------------------------------------

Test program for edm::TypeIDBase class.
Changed by Viji on 29-06-2005

$Id: cputimer_t.cppunit.cpp,v 1.3 2007/11/07 03:06:26 wmtan Exp $
 ----------------------------------------------------------------------*/

#include <cassert>
#include <iostream>
#include <string>
#include <unistd.h>
#include <sys/resource.h>
#include <cppunit/extensions/HelperMacros.h>
#include "FWCore/Utilities/interface/CPUTimer.h"

class testCPUTimer: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testCPUTimer);

CPPUNIT_TEST(testTiming);

CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}

  void testTiming();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testCPUTimer);

using std::cerr;
using std::endl;

void testCPUTimer::testTiming()

{
  edm::CPUTimer timer;
  CPPUNIT_ASSERT(timer.realTime() == 0.0);
  CPPUNIT_ASSERT(timer.cpuTime()==0.0);
  
  timer.start();
  sleep(2);
  timer.stop();
  //Consider good enough if times are within 1%. Closer than that causes test failures
  if((timer.realTime()<=2.0) or (timer.cpuTime()+2.0-0.02 > timer.realTime())) {
    std::cerr <<"real "<<timer.realTime()<<" cpu "<<timer.cpuTime()<< std::endl;
  }
  CPPUNIT_ASSERT(timer.realTime() > 2.0);
  CPPUNIT_ASSERT(timer.cpuTime()+2.0-0.02 <= timer.realTime());

  timer.start();
  sleep(2);
  if(timer.realTime() <= 4.0 ) {
    std::cerr <<"real "<<timer.realTime()<<" cpu "<<timer.cpuTime()<< std::endl;
  }
  CPPUNIT_ASSERT(timer.realTime() > 4.0);
  //this should do nothing
  timer.start();
  CPPUNIT_ASSERT(timer.realTime() > 4.0);
  
  sleep(2);

  timer.stop();
  
  double real = timer.realTime();
  double cpu = timer.cpuTime();

  
  //this should do nothing
  timer.stop();
  CPPUNIT_ASSERT(timer.realTime()==real);
  CPPUNIT_ASSERT(timer.cpuTime()==cpu);

  timer.reset();
  CPPUNIT_ASSERT(timer.realTime() == 0.0);
  CPPUNIT_ASSERT(timer.cpuTime()==0.0);
  
  rusage theUsage;
  getrusage(RUSAGE_SELF, &theUsage) ;
  struct timeval startTime;
  startTime.tv_sec =theUsage.ru_utime.tv_sec;
  startTime.tv_usec =theUsage.ru_utime.tv_usec;
  
  timer.start();
  struct timeval nowTime;
  do {
    rusage theUsage2;
    getrusage(RUSAGE_SELF, &theUsage2) ;
    nowTime.tv_sec =theUsage2.ru_utime.tv_sec;
    nowTime.tv_usec =theUsage2.ru_utime.tv_usec;
  }while(nowTime.tv_sec -startTime.tv_sec +1E-6*(nowTime.tv_usec-startTime.tv_usec) <1);
  timer.stop();

  if( (timer.realTime() < 1.0) or (timer.cpuTime() <1.0)) {
    std::cerr <<"real "<<timer.realTime()<<" cpu "<<timer.cpuTime()<< std::endl;
  }
  CPPUNIT_ASSERT(timer.realTime() >= 1.0);
  CPPUNIT_ASSERT(timer.cpuTime()>=1.0);

}


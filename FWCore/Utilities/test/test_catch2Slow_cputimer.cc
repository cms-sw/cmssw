/*----------------------------------------------------------------------
Test program for edm::TypeIDBase class.
 ----------------------------------------------------------------------*/

#include <cassert>
#include <iostream>
#include <string>
#include <unistd.h>
#include <sys/resource.h>
#include <catch2/catch_all.hpp>
#include "FWCore/Utilities/interface/CPUTimer.h"

using std::cerr;
using std::endl;

TEST_CASE("CPUTimer", "[CPUTimer]") {
  SECTION("testTiming") {
    edm::CPUTimer timer;
    REQUIRE(timer.realTime() == 0.0);
    REQUIRE(timer.cpuTime() == 0.0);
    timer.start();
    sleep(2);
    timer.stop();
    if ((timer.realTime() <= 2.0) || (timer.cpuTime() + 2.0 - 0.02 > timer.realTime())) {
      std::cerr << "real " << timer.realTime() << " cpu " << timer.cpuTime() << std::endl;
    }
    REQUIRE(timer.realTime() > 2.0);
    REQUIRE(timer.cpuTime() + 2.0 - 0.02 <= timer.realTime());
    timer.start();
    sleep(2);
    if (timer.realTime() <= 4.0) {
      std::cerr << "real " << timer.realTime() << " cpu " << timer.cpuTime() << std::endl;
    }
    REQUIRE(timer.realTime() > 4.0);
    timer.start();
    REQUIRE(timer.realTime() > 4.0);
    sleep(2);
    timer.stop();
    double real = timer.realTime();
    double cpu = timer.cpuTime();
    timer.stop();
    REQUIRE(timer.realTime() == real);
    REQUIRE(timer.cpuTime() == cpu);
    timer.reset();
    REQUIRE(timer.realTime() == 0.0);
    REQUIRE(timer.cpuTime() == 0.0);
    rusage theUsage;
    getrusage(RUSAGE_SELF, &theUsage);
    struct timeval startTime;
    startTime.tv_sec = theUsage.ru_utime.tv_sec;
    startTime.tv_usec = theUsage.ru_utime.tv_usec;
    timer.start();
    struct timeval nowTime;
    do {
      rusage theUsage2;
      getrusage(RUSAGE_SELF, &theUsage2);
      nowTime.tv_sec = theUsage2.ru_utime.tv_sec;
      nowTime.tv_usec = theUsage2.ru_utime.tv_usec;
    } while (nowTime.tv_sec - startTime.tv_sec + 1E-6 * (nowTime.tv_usec - startTime.tv_usec) < 1);
    timer.stop();
    if ((timer.realTime() < 1.0) || (timer.cpuTime() < 1.0)) {
      std::cerr << "real " << timer.realTime() << " cpu " << timer.cpuTime() << std::endl;
    }
    REQUIRE(timer.realTime() >= 1.0);
    REQUIRE(timer.cpuTime() >= 1.0);
  }
}

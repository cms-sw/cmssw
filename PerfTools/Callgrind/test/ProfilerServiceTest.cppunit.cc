#define private public
#include "PerfTools/Callgrind/interface/ProfilerService.h"
#undef private


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include <boost/assign/std/vector.hpp>
#include <boost/assign/list_of.hpp>
using namespace boost::assign;


#include <limits>
#include<vector>
#include<string>



#include <cppunit/extensions/HelperMacros.h>


class TestProfilerService : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestProfilerService);
  CPPUNIT_TEST_SUITE(check_constr);
  CPPUNIT_TEST_SUITE(check_config);
  CPPUNIT_TEST_SUITE(check_Instrumentation);
  CPPUNIT_TEST_SUITE(check_Event);
  CPPUNIT_TEST_SUITE(check_Path);
  CPPUNIT_TEST_SUITE(check_Nesting);

public:
  TestProfilerService();
  void setUp();
  void tearDown();
  void check_constr();
  void check_config();
  void check_Instrumentation();
  void check_Event();
  void check_Path();
  void check_Nesting();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestProfilerService);

TestProfilerService::TestProfilerService(){}

void TestProfilerService::setUp() {}

void TestProfilerService::tearDown() {}

void TestProfilerService::check_constr() {
  edm::ParameterSet pset;
  edm::ActivityRegistry activity;
  ProfilerService ps(pset,activity);
  CPPUNIT_ASSERT(ps.m_firstEvent==0);
  CPPUNIT_ASSERT(ps.m_lastEvent==std::numeric_limits<int>::max());
  CPPUNIT_ASSERT(ps.m_paths.empty());
}

void TestProfilerService::check_config() {
  int fe=2;
  int le=10;
  std::vector<std::string> paths; 
  paths += "p1","p2","p3";
  edm::ParameterSet pset;
  pset.setUntrackedParameter<int>("firstEvent",fe);
  pset.setUntrackedParameter<int>("lastEvent",le);
  pset.setUntrackedParameter<std::vector<std::string> >("paths",paths);
  edm::ActivityRegistry activity;
  ProfilerService ps(pset,activity);
  CPPUNIT_ASSERT(ps.m_firstEvent==fe);
  CPPUNIT_ASSERT(ps.m_lastEvent==le);
  CPPUNIT_ASSERT(ps.m_paths==paths);
}

void TestProfilerService::check_Instrumentation() {
  int fe=2;
  int le=10;
  edm::ParameterSet pset;
  pset.setUntrackedParameter<int>("firstEvent",fe);
  pset.setUntrackedParameter<int>("lastEvent",le);
  edm::ActivityRegistry activity;
  ProfilerService ps(pset,activity);
  ps.beginEvent();
  CPPUNIT_ASSERT(m_active==0);
  CPPUNIT_ASSERT(!ps.doEvent());
  CPPUNIT_ASSERT(!ps.startInstrumentation());
  CPPUNIT_ASSERT(!ps.stopInstrumentation());
  CPPUNIT_ASSERT(!ps.forceStopInstrumentation());

  ps.beginEvent();
  CPPUNIT_ASSERT(m_active==0);
  CPPUNIT_ASSERT(ps.doEvent());
  CPPUNIT_ASSERT(ps.startInstrumentation());
  CPPUNIT_ASSERT(ps.stopInstrumentation());
  CPPUNIT_ASSERT(!ps.forceStopInstrumentation());

  CPPUNIT_ASSERT(ps.startInstrumentation());
  CPPUNIT_ASSERT(!ps.startInstrumentation());
  CPPUNIT_ASSERT(!ps.stopInstrumentation());
  CPPUNIT_ASSERT(ps.stopInstrumentation());
  CPPUNIT_ASSERT(!ps.stopInstrumentation());


  for(int i=2;i<10;i++) ps.beginEvent();
  CPPUNIT_ASSERT(ps.m_evtCount==10);
  CPPUNIT_ASSERT(ps.doEvent());
  CPPUNIT_ASSERT(ps.startInstrumentation());
  CPPUNIT_ASSERT(ps.stopInstrumentation());
  ps.beginEvent();
  CPPUNIT_ASSERT(ps.m_evtCount==11);
  CPPUNIT_ASSERT(!ps.doEvent());
  CPPUNIT_ASSERT(!ps.startInstrumentation());
  CPPUNIT_ASSERT(!ps.stopInstrumentation());

}

void TestProfilerService::check_Event() {

}

void TestProfilerService::check_Path() {

}
void TestProfilerService::check_Nesting() {

}


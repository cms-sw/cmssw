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
  CPPUNIT_TEST(check_constr);
  CPPUNIT_TEST(check_config);
  CPPUNIT_TEST(check_Instrumentation);
  CPPUNIT_TEST(check_Event);
  CPPUNIT_TEST(check_Path);
  CPPUNIT_TEST(check_Nesting);
  CPPUNIT_TEST_SUITE_END();
  
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
  pset.addUntrackedParameter<int>("firstEvent",fe);
  pset.addUntrackedParameter<int>("lastEvent",le);
  pset.addUntrackedParameter<std::vector<std::string> >("paths",paths);
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
  pset.addUntrackedParameter<int>("firstEvent",fe);
  pset.addUntrackedParameter<int>("lastEvent",le);
  edm::ActivityRegistry activity;
  ProfilerService ps(pset,activity);
  ps.beginEvent();
  CPPUNIT_ASSERT(ps.m_active==0);
  CPPUNIT_ASSERT(!ps.doEvent());
  CPPUNIT_ASSERT(!ps.startInstrumentation());
  CPPUNIT_ASSERT(!ps.stopInstrumentation());
  CPPUNIT_ASSERT(!ps.forceStopInstrumentation());
  
  ps.beginEvent();
  CPPUNIT_ASSERT(ps.m_active==0);
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
   int fe=2;
  int le=10;
  std::vector<std::string> paths; 
  paths += "ALL";
  edm::ParameterSet pset;
  pset.addUntrackedParameter<int>("firstEvent",fe);
  pset.addUntrackedParameter<int>("lastEvent",le);
  pset.addUntrackedParameter<std::vector<std::string> >("paths",paths);
  edm::ActivityRegistry activity;
  ProfilerService ps(pset,activity);

  ps.beginEvent();
  CPPUNIT_ASSERT(ps.m_active==0);
  CPPUNIT_ASSERT(!ps.doEvent());
  ps.endEvent();

  ps.beginEvent();
  CPPUNIT_ASSERT(ps.m_active==1);
  CPPUNIT_ASSERT(ps.doEvent());
  ps.endEvent();
  CPPUNIT_ASSERT(ps.m_active==0);
  for(int i=2;i<10;i++) {
    ps.beginEvent();
    ps.endEvent();
  }
  CPPUNIT_ASSERT(ps.m_evtCount==10);
  CPPUNIT_ASSERT(ps.doEvent()); // who cares?

  ps.beginEvent();
  CPPUNIT_ASSERT(ps.m_active==0);
  CPPUNIT_ASSERT(!ps.doEvent());
  ps.endEvent();
  CPPUNIT_ASSERT(ps.m_active==0);
 

}

struct CheckPaths {
  CheckPaths(ProfilerService & ips, std::vector<std::string> const & iselpaths, int ibase=0 ) : 
    ps(ips), selpaths(iselpaths),base(ibase), done(0){}
  ProfilerService & ps;
  std::vector<std::string> const & selpaths;
  int base;

  mutable int done;
  
  void operator()(std::string const & path) const {
    bool ok = ps.doEvent() && (std::find(selpaths.begin(),selpaths.end(),path) != selpaths.end());
    noselPath();
    ps.beginPath(path);
    if (ok) selPath(path);
    else noselPath();
    ps.endPath(path);
    noselPath();
  }

  void selPath(const std::string & path) const {
    CPPUNIT_ASSERT(ps.m_active==base+1);
    CPPUNIT_ASSERT(ps.m_activePath==path);
    ++done;
  }

  void noselPath() const {
   CPPUNIT_ASSERT(ps.m_active==base);
   CPPUNIT_ASSERT(ps.m_activePath.empty());
  }

};

void TestProfilerService::check_Path() {
  int fe=2;
  int le=10;
  std::vector<std::string> paths; 
  paths += "p1","p2","p3";
  edm::ParameterSet pset;
  pset.addUntrackedParameter<int>("firstEvent",fe);
  pset.addUntrackedParameter<int>("lastEvent",le);
  pset.addUntrackedParameter<std::vector<std::string> >("paths",paths);
  edm::ActivityRegistry activity;
  ProfilerService ps(pset,activity);

  std::vector<std::string> allPaths; 
  allPaths += "p1","p21","p22","p3";
  CheckPaths cp(ps,paths);
  CPPUNIT_ASSERT(std::find(paths.begin(),paths.end(),allPaths[0]) != paths.end());
  CPPUNIT_ASSERT(std::find(paths.begin(),paths.end(),allPaths[1]) == paths.end());
  CPPUNIT_ASSERT(std::find(paths.begin(),paths.end(),allPaths[2]) == paths.end());
  CPPUNIT_ASSERT(std::find(paths.begin(),paths.end(),allPaths[3]) != paths.end());

  ps.beginEvent();
  CPPUNIT_ASSERT(ps.m_active==0);
  CPPUNIT_ASSERT(!ps.doEvent());
  std::for_each(allPaths.begin(),allPaths.end(),cp);
  CPPUNIT_ASSERT(cp.done==0);
  ps.endEvent();

  ps.beginEvent();
  CPPUNIT_ASSERT(ps.m_active==0);
  CPPUNIT_ASSERT(ps.doEvent());
  std::for_each(allPaths.begin(),allPaths.end(),cp);
  CPPUNIT_ASSERT(cp.done==2);
  ps.endEvent();

}

void TestProfilerService::check_Nesting() {
  int fe=2;
  int le=10;
  std::vector<std::string> paths; 
  paths += "ALL", "p1","p2","p3";
  edm::ParameterSet pset;
  pset.addUntrackedParameter<int>("firstEvent",fe);
  pset.addUntrackedParameter<int>("lastEvent",le);
  pset.addUntrackedParameter<std::vector<std::string> >("paths",paths);
  edm::ActivityRegistry activity;
  ProfilerService ps(pset,activity);

  std::vector<std::string> allPaths; 
  allPaths += "p1","p21","p22","p3";
  CheckPaths cp(ps,paths,1);

  ps.beginEvent();
  CPPUNIT_ASSERT(ps.m_active==0);
  CPPUNIT_ASSERT(!ps.doEvent());
  std::for_each(allPaths.begin(),allPaths.end(),cp);
  CPPUNIT_ASSERT(cp.done==0);
  ps.endEvent();
  CPPUNIT_ASSERT(ps.m_active==0);

  ps.beginEvent();
  CPPUNIT_ASSERT(ps.m_active==1);
  CPPUNIT_ASSERT(ps.doEvent());
  std::for_each(allPaths.begin(),allPaths.end(),cp);
  CPPUNIT_ASSERT(cp.done==2);
  ps.endEvent();
  CPPUNIT_ASSERT(ps.m_active==0);
 
}


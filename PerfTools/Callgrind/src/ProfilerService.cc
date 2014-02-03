#include "PerfTools/Callgrind/interface/ProfilerService.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <limits>

#include "valgrind/callgrind.h"

ProfilerService::ProfilerService(edm::ParameterSet const& pset, 
				 edm::ActivityRegistry  & activity) :
  
  m_firstEvent(pset.getUntrackedParameter<int>("firstEvent",0 )),
  m_lastEvent(pset.getUntrackedParameter<int>("lastEvent",std::numeric_limits<int>::max())),
  m_dumpInterval(pset.getUntrackedParameter<int>("dumpInterval",100)),
  m_paths(pset.getUntrackedParameter<std::vector<std::string> >("paths",std::vector<std::string>() )),
  m_excludedPaths(pset.getUntrackedParameter<std::vector<std::string> >("excludePaths",std::vector<std::string>() )),
  m_allPaths(false),
  m_evtCount(0),
  m_counts(0),
  m_doEvent(false),
  m_active(0),
  m_paused(false) {
  static std::string const allPaths("ALL");
  m_allPaths = std::find(m_paths.begin(),m_paths.end(),allPaths) != m_paths.end();
  
  // either FullEvent or selected path
  static std::string const fullEvent("FullEvent");
  if (std::find(m_paths.begin(),m_paths.end(),fullEvent) != m_paths.end())
    activity.watchPostSource(this,&ProfilerService::preSourceI);
  else {
    activity.watchPreProcessEvent(this,&ProfilerService::beginEventI);
    activity.watchPostProcessEvent(this,&ProfilerService::endEventI);
    activity.watchPreProcessPath(this,&ProfilerService::beginPathI);
    activity.watchPostProcessPath(this,&ProfilerService::endPathI);
  }
}

ProfilerService::~ProfilerService(){
  dumpStat();
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
bool ProfilerService::startInstrumentation(){
  // FIXME here or in client?
  if (!doEvent()) return false;


  if (m_active==0) {
    CALLGRIND_START_INSTRUMENTATION;
    if (m_counts%m_dumpInterval==0) dumpStat();
    ++m_counts;
  }
  // support nested start/stop
  ++m_active;
  return m_active==1;
}

bool ProfilerService::stopInstrumentation() {
  if (m_active==0) return false;
  --m_active;
  if (m_active==0)
    CALLGRIND_STOP_INSTRUMENTATION;
  return m_active==0;
}

bool ProfilerService::forceStopInstrumentation() {
  if (m_active==0) return false;
  // FIXME report something if appens;
  CALLGRIND_STOP_INSTRUMENTATION;
  m_active=0;
  return true;
}

bool ProfilerService::pauseInstrumentation() {
   if (m_active==0) return false;
   CALLGRIND_STOP_INSTRUMENTATION;
   m_paused=true;
   return true;
}   

bool ProfilerService::resumeInstrumentation() {
  if (m_active==0 || (!m_paused)) return false;
  CALLGRIND_START_INSTRUMENTATION;
  if (m_counts%m_dumpInterval==0) dumpStat();
  ++m_counts;
  m_paused=false;
  return true;
}

void ProfilerService::dumpStat() const {
     CALLGRIND_DUMP_STATS;
}
#pragma GCC diagnostic pop

void ProfilerService::newEvent() {
  ++m_evtCount;
  m_doEvent = m_evtCount >= m_firstEvent && m_evtCount <= m_lastEvent;
}


void ProfilerService::fullEvent() {
  newEvent();
  if(m_doEvent&&m_active==0)
    startInstrumentation();
  if ( (!m_doEvent) && m_active!=0) {
    stopInstrumentation();
    // force, a nested instrumentation may fail to close in presence of filters
    forceStopInstrumentation();
    dumpStat();
  }
}

void  ProfilerService::beginEvent() {
  newEvent();
  //  static std::string const fullEvent("FullEvent");
  //  if (std::find(m_paths.begin(),m_paths.end(),fullEvent) != m_paths.end())
  if (m_allPaths) 
    startInstrumentation();
}

void  ProfilerService::endEvent() {
  stopInstrumentation();
  // force, a nested instrumentation may fail to close in presence of filters
  forceStopInstrumentation();
}

void  ProfilerService::beginPath(std::string const & path) {
  if (!doEvent()) return;
  // assume less than 5-6 path to instrument or to exclude
  if (std::find(m_excludedPaths.begin(),m_excludedPaths.end(),path) != m_excludedPaths.end()) {
    pauseInstrumentation();
    return; 
  }
  if (std::find(m_paths.begin(),m_paths.end(),path) == m_paths.end()) return; 
  m_activePath=path;
  startInstrumentation();
}

void  ProfilerService::endPath(std::string const & path) {
  resumeInstrumentation();
  if (m_activePath==path) {
    stopInstrumentation();
    m_activePath.clear();
  }
  // do to force, a nested instrumentation may fail to close in presence of filters  
  // forceStopInstrumentation();
}

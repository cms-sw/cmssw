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
  m_paths(pset.getUntrackedParameter<std::vector<std::string> >("paths",std::vector<std::string>() )),
  m_allPaths(false),
  m_evtCount(0),
  m_doEvent(false),
  m_active(0){
  static std::string const allPaths("ALL");
  m_allPaths = std::find(m_paths.begin(),m_paths.end(),allPaths) != m_paths.end();
 

    activity.watchPreProcessEvent(this,&ProfilerService::beginEventI);
    activity.watchPostProcessEvent(this,&ProfilerService::endEventI);
    activity.watchPreProcessPath(this,&ProfilerService::beginPathI);
    activity.watchPostProcessPath(this,&ProfilerService::endPathI);

}

ProfilerService::~ProfilerService(){}

bool ProfilerService::startInstrumentation(){
  // FIXME here or in client?
  if (!doEvent()) return false;


  if (m_active==0) {
    CALLGRIND_START_INSTRUMENTATION;
    CALLGRIND_DUMP_STATS;
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

void ProfilerService::dumpStat() {
     CALLGRIND_DUMP_STATS;
}

void  ProfilerService::beginEvent() {
  ++m_evtCount;
  m_doEvent = m_evtCount >= m_firstEvent && m_evtCount <= m_lastEvent;
  static std::string const fullEvent("FullEvent");
  if (std::find(m_paths.begin(),m_paths.end(),fullEvent) != m_paths.end())
    startInstrumentation();
}

void  ProfilerService::endEvent() {
  stopInstrumentation();
  // force, a nested instrumentation may fail to close in presence of filters
  forceStopInstrumentation();
}

void  ProfilerService::beginPath(std::string const & path) {
  if (!doEvent()) return;
  // assume less than 5-6 path to instrument ....
  if ( (!m_allPaths) && std::find(m_paths.begin(),m_paths.end(),path) == m_paths.end()) return; 
  m_activePath=path;
  startInstrumentation();
}

void  ProfilerService::endPath(std::string const & path) {
  if ( m_allPaths || m_activePath==path) {
    stopInstrumentation();
    m_activePath.clear();
  }
  // do to force, a nested instrumentation may fail to close in presence of filters  
  // forceStopInstrumentation();
}

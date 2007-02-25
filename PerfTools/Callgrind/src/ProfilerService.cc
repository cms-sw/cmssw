#include "PerfTools/Callgrind/interface/ProfilerService.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>

#include "valgrind/callgrind.h"

ProfilerService::ProfilerService(edm::ParameterSet const& pset, 
				 edm::ActivityRegistry  & activity) :
  
  m_firstEvent(pset.getUntrackedParameter<int>("FirstEvent",0 )),
  m_lastEvent(pset.getUntrackedParameter<int>("LastEvent",-1)),
  m_paths(pset.getUntrackedParameter<std::vector<std::string> >("Paths",std::vector<std::string>() )),
  m_evtCount(0),
  m_doEvent(false),
  m_active(0){
    activity.watchPreProcessEvent(this,&ProfilerService::beginEvent);
    activity.watchPostProcessEvent(this,&ProfilerService::endEvent);
    activity.watchPreProcessPath(this,&ProfilerService::beginPath);
    activity.watchPostProcessPath(this,&ProfilerService::endPath);

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


bool ProfilerService::dumpStat() {
     CALLGRIND_DUMP_STATS;
}

void  ProfilerService::beginEvent(const edm::EventID&, const edm::Timestamp&) {
  ++m_evtCount;
  m_doEvent = m_evtCount >= m_firstEvent && m_evtCount <= m_lastEvent;
  static std::string const allPaths("ALL");
  if (std::find(m_paths.begin(),m_paths.end(),allPaths) != m_paths.end())
    startInstrumentation();
}

void  ProfilerService::endEvent(const edm::Event&, const edm::EventSetup&) {
  stopInstrumentation();
  // force, a nested instrumentation may fail to close in presence of filters
  forceStopInstrumentation();
}

void  ProfilerService::beginPath(std::string const & path) {
  if (!doEvent()) return;
  // assume less than 5-6 path to instrument ....
  if (std::find(m_paths.begin(),m_paths.end(),path) == m_paths.end()) return; 
    m_activePath=path;
    startInstrumentation();
}

void  ProfilerService::endPath(std::string const & path,  const edm::HLTPathStatus&) {
  if ( m_activePath==path) {
    stopInstrumentation();
    m_activePath.clear();
  }
  // do to force, a nested instrumentation may fail to close in presence of filters  
  // forceStopInstrumentation();
}

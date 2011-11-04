//system headers
#include <time.h>

// C++ headers
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <map>
#include <cmath>

// boost headers
#include <boost/foreach.hpp>
// for forward compatibility with boost 1.47
#define BOOST_FILESYSTEM_VERSION 3
#include <boost/filesystem/path.hpp>

// CMSSW headers
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/HLTPathStatus.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "HLTrigger/Timer/interface/FastTimerService.h"
#include "HLTrigger/Timer/interface/CPUAffinity.h"


FastTimerService::FastTimerService(const edm::ParameterSet & config, edm::ActivityRegistry & registry) :
  m_timer_id(               config.getUntrackedParameter<bool>(        "useRealTimeClock",     false) ? CLOCK_REALTIME : CLOCK_THREAD_CPUTIME_ID),
  m_is_cpu_bound(           false ),
  m_enable_timing_modules(  config.getUntrackedParameter<bool>(        "enableTimingModules",  false) ),
  m_enable_timing_paths(    config.getUntrackedParameter<bool>(        "enableTimingPaths",    false) ),
  m_enable_timing_summary(  config.getUntrackedParameter<bool>(        "enableTimingSummary",  false) ), 
  m_enable_dqm(             config.getUntrackedParameter<bool>(        "enableDQM",            false) ), 
  m_enable_dqm_bylumi(      config.getUntrackedParameter<bool>(        "enableDQMbyLumi",      false) ), 
  m_dqm_time_range(         config.getUntrackedParameter<double>(      "dqmTimeRange",         1000.) ),   // ms
  m_dqm_time_resolution(    config.getUntrackedParameter<double>(      "dqmTimeResolution",       5.) ),   // ms
  m_dqm_path(               config.getUntrackedParameter<std::string>( "dqmPath",     "TimerService") ),
  m_first_path(0),          // these are initialized at prePathBeginRun(), 
  m_last_path(0),           // to make sure we cache the correct pointers
  m_first_endpath(0),
  m_last_endpath(0),
  m_event(0.),
  m_source(0.),
  m_all_paths(0.),
  m_all_endpaths(0.),
  m_paths(),
  m_modules(),
  m_summary_events(0),
  m_summary_event(0.),
  m_summary_source(0.),
  m_summary_all_paths(0.),
  m_summary_all_endpaths(0.),
  m_summary_paths(),
  m_summary_modules(),
  m_dqms(0),                // these are initialized at postBeginJob(),
  m_dqm_event(0),           // to make sure the DQM services have been loaded
  m_dqm_source(0),
  m_dqm_all_paths(0),
  m_dqm_all_endpaths(0)
{
  registry.watchPostBeginJob(      this, & FastTimerService::postBeginJob );
  registry.watchPostEndJob(        this, & FastTimerService::postEndJob );
  registry.watchPrePathBeginRun(   this, & FastTimerService::prePathBeginRun) ;
  registry.watchPreProcessEvent(   this, & FastTimerService::preProcessEvent );
  registry.watchPostProcessEvent(  this, & FastTimerService::postProcessEvent );
  registry.watchPreSource(         this, & FastTimerService::preSource );
  registry.watchPostSource(        this, & FastTimerService::postSource );
  // watch per-path events 
  registry.watchPreProcessPath(    this, & FastTimerService::preProcessPath );
  registry.watchPostProcessPath(   this, & FastTimerService::postProcessPath );
  // watch per-module events o if enabled
  if (m_enable_timing_modules) {
    registry.watchPreModuleBeginJob( this, & FastTimerService::preModuleBeginJob );
    registry.watchPreModule(         this, & FastTimerService::preModule );
    registry.watchPostModule(        this, & FastTimerService::postModule );
  }
}

FastTimerService::~FastTimerService()
{
}

void FastTimerService::postBeginJob() {
  // check if the process is bound to a single CPU.
  // otherwise, the results of the CLOCK_THREAD_CPUTIME_ID timer might be inaccurate
  m_is_cpu_bound = CPUAffinity::isCpuBound();
  if ((m_timer_id != CLOCK_REALTIME) and not m_is_cpu_bound)
    // the process is NOT bound to a single CPU
    edm::LogError("FastTimerService") << "this process is NOT bound to a single CPU, the results of the FastTimerService may be undefined";

  edm::service::TriggerNamesService & tns = * edm::Service<edm::service::TriggerNamesService>();
  BOOST_FOREACH(std::string const & name, tns.getTrigPaths())
    m_paths[name] = 0.;
  BOOST_FOREACH(std::string const & name, tns.getEndPaths())
    m_paths[name] = 0.;

  if (m_enable_dqm)
    // load the DQM store
    m_dqms = edm::Service<DQMStore>().operator->();

  if (m_dqms) {
    if (m_enable_dqm_bylumi) {
      // by-lumi plots use a different second-level directory: 
      //    TimerService   --> TimerService/EventInfo
      //    HLT/Timer      --> HLT/EventInfo/Timer
      // etc.

      // assume the path to be relative, and to have at least an item
      boost::filesystem::path::iterator item = m_dqm_path.begin();
      boost::filesystem::path path = * item++;
      path /= "EventInfo";
      while (item != m_dqm_path.end())
        path /= * item++;
      m_dqm_path = path;
    }

    // book MonitorElement's
    int bins = (int) std::ceil(m_dqm_time_range / m_dqm_time_resolution);
    m_dqms->setCurrentFolder(m_dqm_path.generic_string());
    m_dqm_event         = m_dqms->book1D("event",        "Event",    bins, 0., m_dqm_time_range);
    m_dqm_source        = m_dqms->book1D("source",       "Source",   bins, 0., m_dqm_time_range);
    m_dqm_all_paths     = m_dqms->book1D("all_paths",    "Paths",    bins, 0., m_dqm_time_range);
    m_dqm_all_endpaths  = m_dqms->book1D("all_endpaths", "EndPaths", bins, 0., m_dqm_time_range);
    if (m_enable_timing_paths) {
      m_dqms->setCurrentFolder((m_dqm_path / "Paths").generic_string());
      BOOST_FOREACH(PathMap<double>::value_type const & path, m_paths)
        m_dqm_paths[path.first] = m_dqms->book1D(path.first, path.first, bins, 0., m_dqm_time_range);
    }
    if (m_enable_timing_modules) {
      m_dqms->setCurrentFolder((m_dqm_path / "Modules").generic_string());
      BOOST_FOREACH(ModuleMap<double>::value_type const & module, m_modules)
        m_dqm_modules[module.first] = m_dqms->book1D(module.first->moduleLabel(), module.first->moduleLabel(), bins, 0., m_dqm_time_range);
    }
  }
}

void FastTimerService::postEndJob() {
  if (not m_enable_timing_summary)
    return;

  edm::service::TriggerNamesService & tns = * edm::Service<edm::service::TriggerNamesService>();

  // spacing is set to mimic TimeReport
  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  out << "FastReport " << std::right << std::setw(10) << (m_timer_id == CLOCK_REALTIME ? "Real" : "CPU") << '\n';
  out << "FastReport " << std::right << std::setw(10) << m_summary_source       / (double) m_summary_events << "                                  Source"        << '\n';
  out << "FastReport " << std::right << std::setw(10) << m_summary_event        / (double) m_summary_events << "                                  Event"         << '\n';
  out << "FastReport " << std::right << std::setw(10) << m_summary_all_paths    / (double) m_summary_events << "                                  all Paths"     << '\n';
  out << "FastReport " << std::right << std::setw(10) << m_summary_all_endpaths / (double) m_summary_events << "                                  all EndPaths"  << '\n';
  if (m_enable_timing_paths) {
    out << '\n';
    out << "FastReport " << std::right << std::setw(10) << (m_timer_id == CLOCK_REALTIME ? "Real" : "CPU")    << "                                  Path"          << '\n';
    BOOST_FOREACH(std::string const & name, tns.getTrigPaths())
      out << "FastReport " << std::right << std::setw(10) << m_summary_paths[name]  / (double) m_summary_events << "                                  " << name << '\n';
    out << '\n';
    out << "FastReport " << std::right << std::setw(10) << (m_timer_id == CLOCK_REALTIME ? "Real" : "CPU")    << "                                  EndPath"       << '\n';
    BOOST_FOREACH(std::string const & name, tns.getEndPaths())
      out << "FastReport " << std::right << std::setw(10) << m_summary_paths[name]  / (double) m_summary_events << "                                  " << name << '\n';
  }
  edm::LogVerbatim("FastReport") << out.str();
}

void FastTimerService::preModuleBeginJob(edm::ModuleDescription const & module) {
  // allocate a counter for each module
  m_modules.insert( std::make_pair(& module, (double) 0.) );
}

void FastTimerService::preProcessEvent(edm::EventID const & id, edm::Timestamp const & stamp) {
  // new event, reset the per-event counter
  start(m_timer_event);
}

void FastTimerService::postProcessEvent(edm::Event const & event, edm::EventSetup const & setup) {
  // stop the per-event timer, and account event time
  stop(m_timer_event);
  m_event = delta(m_timer_event);
  if (m_enable_timing_summary)
    m_summary_event += m_event;
  if (m_dqms)
    m_dqm_event->Fill(m_event * 1000.);                         // convert to ms
}

void FastTimerService::preSource() {
  start(m_timer_source);

  // keep track of the total number of events
  ++m_summary_events;
}

void FastTimerService::postSource() {
  stop(m_timer_source);
  m_source = delta(m_timer_source);
  if (m_enable_timing_summary)
    m_summary_source += m_source;
  if (m_dqms)
    m_dqm_source->Fill(m_source * 1000.);                       // convert to ms
}

void FastTimerService::prePathBeginRun(std::string const & path ) {
  // cache the pointers to the names of the first and last path and endpath
  edm::service::TriggerNamesService & tns = * edm::Service<edm::service::TriggerNamesService>();
  if (path == tns.getTrigPaths().front())
    m_first_path = & path;
  if (path == tns.getTrigPaths().back())
    m_last_path = & path;
  if (path == tns.getEndPaths().front())
    m_first_endpath = & path;
  if (path == tns.getEndPaths().back())
    m_last_endpath = & path;
}

void FastTimerService::preProcessPath(std::string const & path ) {
  // time each (end)path
  start(m_timer_path);

  if (& path == m_first_path) {
    // this is the first path, start the "all paths" counter
    m_timer_paths.first = m_timer_path.first;
  } else if (& path == m_first_endpath) {
    // this is the first endpath, start the "all paths" counter
    m_timer_endpaths.first = m_timer_path.first;
  }
}

void FastTimerService::postProcessPath(std::string const & path, edm::HLTPathStatus const & status) {
  // time each (end)path
  stop(m_timer_path);

  if (& path == m_last_path) {
    // this is the last path, stop and account the "all paths" counter
    m_timer_paths.second = m_timer_path.second;
    m_all_paths = delta(m_timer_paths);
    if (m_enable_timing_summary)
      m_summary_all_paths += m_all_paths;
    if (m_dqms)
      m_dqm_all_paths->Fill(m_all_paths * 1000.);               // convert to ms
  } else if (& path == m_last_endpath) {
    // this is the last endpath, stop and account the "all endpaths" counter
    m_timer_endpaths.second = m_timer_path.second;
    m_all_endpaths = delta(m_timer_endpaths);
    if (m_enable_timing_summary)
      m_summary_all_endpaths += m_all_endpaths;
    if (m_dqms)
      m_dqm_all_endpaths->Fill(m_all_endpaths * 1000.);         // convert to ms
  }

  // if enabled, account each (end)path
  if (m_enable_timing_paths) {
    PathMap<double>::iterator keyval;
    if ((keyval = m_paths.find(path)) != m_paths.end()) {
      keyval->second = delta(m_timer_path);
      if (m_enable_timing_summary)
        m_summary_paths[path] += keyval->second;
      if (m_dqms)
        m_dqm_paths[path]->Fill(keyval->second * 1000.);        // convert to ms
    } else {
      // should never get here
      edm::LogError("FastTimerService") << "FastTimerService::postProcessPath: unexpected path " << path;
    }
  }
}

void FastTimerService::preModule(edm::ModuleDescription const & module) {
  // time each module
  start(m_timer_module);
}

void FastTimerService::postModule(edm::ModuleDescription const & module) {
  // time and account each module
  stop(m_timer_module);
  ModuleMap<double>::iterator keyval = m_modules.find(& module);
  if (keyval != m_modules.end()) {
    keyval->second = delta(m_timer_module);
    if (m_enable_timing_summary)
      m_summary_modules[& module] += keyval->second;
    if (m_dqms)
      m_dqm_modules[& module]->Fill(keyval->second * 1000.);    // convert to ms
  } else {
    // should never get here
    edm::LogError("FastTimerService") << "FastTimerService::postModule: unexpected module " << module.moduleLabel();
  }
}


// declare FastTimerService as a framework Service
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(FastTimerService);

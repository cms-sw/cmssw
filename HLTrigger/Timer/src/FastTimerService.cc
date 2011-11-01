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


/*
procesing time is diveded into
 - source
 - pre-event processing overhead
 - event processing
 - post-event processing overhead

until lumi-processing and run-processing are taken into account, they will count as overhead

event processing time is diveded into
 - trigger processing (from the begin of the first path to the end of the last path)
 - trigger overhead
 - endpath processing (from the begin of the first endpath to the end of the last endpath)
 - endpath overhead
*/

/*
Timer per-call overhead (on lxplus, SLC 5.7):

Test results for "clock_gettime(CLOCK_THREAD_CPUTIME_ID, & value)"
 - average time per call: 340 ns
 - resolution:              ? ns

Test results for "clock_gettime(CLOCK_PROCESS_CPUTIME_ID, & value)"
 - average time per call: 360 ns
 - resolution:              ? ns

Test results for "clock_gettime(CLOCK_REALTIME, & value)"
 - average time per call: 320 ns
 - resolution:              1 us

Test results for "getrusage(RUSAGE_SELF, & value)"
 - average time per call: 380 ns
 - resolution:              1 ms

Test results for "gettimeofday(& value, NULL)"
 - average time per call:  65 ns
 - resolution:              1 us

Assuming an HLT process with ~2500 modules and ~500 paths, tracking each step 
with clock_gettime(CLOCK_THREAD_CPUTIME_ID) gives a per-event overhead of 2 ms
*/

class FastTimerService {
public:
  FastTimerService(const edm::ParameterSet &, edm::ActivityRegistry & );
  ~FastTimerService();

  // query the current module/path/event
  // Note: these functions incur in a "timer per-call overhead" (see above), currently of the order of 340ns
  double currentModuleTime() const;         // return the time spent since the last preModule() event
  double currentPathTime() const;           // return the time spent since the last preProcessPath() event
  double currentEventTime() const;          // return the time spent since the last preProcessEvent() event

  // query the time spent in a module/path the last time it has run
  double queryModuleTime(const edm::ModuleDescription &) const;
  double queryPathTime(const std::string &) const;

  // query the time spent in the current (or last) event's
  //  - source        (available during event processing)
  //  - all paths     (available during endpaths)
  //  - all endpaths  (available after all endpaths have run, usually returns the last event's value)
  //  - processing    (available after the event has been processed, usually returns the last event's value)
  double querySourceTime() const;
  double queryPathsTime() const;
  double queryEndPathsTime() const;
  double queryEventTime() const;

  // try to assess the overhead which may not be included in the source, paths and event timers
  double queryPreSourceOverhead() const;    // time spent after the previous event's postProcessEvent and this event's preSource
  double queryPreEventOverhead() const;     // time spent after this event's postSource and preProcessEvent
  double queryPreEndPathsOverhead() const;  // time spent after the last path's postProcessPath and the first endpath's preProcessPath


private:
  void postBeginJob();
  void postEndJob();
  void preModuleBeginJob( edm::ModuleDescription const & );
  void preProcessEvent( edm::EventID const &, edm::Timestamp const & );
  void postProcessEvent( edm::Event const &, edm::EventSetup const & );
  void preSource();
  void postSource();
  void prePathBeginRun( std::string const & );
  void preProcessPath(  std::string const & );
  void postProcessPath( std::string const &, edm::HLTPathStatus const & );
  void preModule(  edm::ModuleDescription const & );
  void postModule( edm::ModuleDescription const & );

private:
  template <typename T> class PathMap   : public std::map<std::string, T> {};
  template <typename T> class ModuleMap : public std::map<edm::ModuleDescription const *, T> {};

  // configuration
  const clockid_t                               m_timer_id;             // the default is to use CLOCK_THREAD_CPUTIME_ID, unless useRealTimeClock is set, which will use CLOCK_REALTIME
  const bool                                    m_enable_timing_modules;
  const bool                                    m_enable_timing_paths;
  const bool                                    m_enable_timing_summary;
  const bool                                    m_enable_dqm;
  const bool                                    m_enable_dqm_bylumi;
  const double                                  m_dqm_time_range;
  const double                                  m_dqm_time_resolution;
  boost::filesystem::path                       m_dqm_path;

  const std::string *                           m_first_path;           // the framework does not provide a pre-paths or pre-endpaths signal,
  const std::string *                           m_last_path;            // so we emulate them keeping track of the first and last path and endpath
  const std::string *                           m_first_endpath;
  const std::string *                           m_last_endpath;

  // per-event accounting
  double                                        m_event;
  double                                        m_source;
  double                                        m_all_paths;
  double                                        m_all_endpaths;
  PathMap<double>                               m_paths;
  ModuleMap<double>                             m_modules;              // this assumes that ModuleDescription are stored in the same object through the whole job,
                                                                        // which is true only *after* the edm::Worker constructors have run
  // per-job summary
  unsigned int                                  m_summary_events;       // number of events
  double                                        m_summary_event;
  double                                        m_summary_source;
  double                                        m_summary_all_paths;
  double                                        m_summary_all_endpaths;
  PathMap<double>                               m_summary_paths;
  ModuleMap<double>                             m_summary_modules;      // see the comment for m_modules

  // DQM
  DQMStore *                                    m_dqms;
  MonitorElement *                              m_dqm_event;
  MonitorElement *                              m_dqm_source;
  MonitorElement *                              m_dqm_all_paths;
  MonitorElement *                              m_dqm_all_endpaths;
  PathMap<MonitorElement *>                     m_dqm_paths;
  ModuleMap<MonitorElement *>                   m_dqm_modules;          // see the comment for m_modules

  // timers
  std::pair<struct timespec, struct timespec>   m_timer_event;          // track time spent in each event
  std::pair<struct timespec, struct timespec>   m_timer_source;         // track time spent in the source
  std::pair<struct timespec, struct timespec>   m_timer_paths;          // track time spent in all paths
  std::pair<struct timespec, struct timespec>   m_timer_endpaths;       // track time spent in all endpaths
  std::pair<struct timespec, struct timespec>   m_timer_path;           // track time spent in each path
  std::pair<struct timespec, struct timespec>   m_timer_module;         // track time spent in each module

  void gettime(struct timespec & stamp)
  {
    clock_gettime(m_timer_id, & stamp);
  }

  void start(std::pair<struct timespec, struct timespec> & times)
  {
    gettime(times.first);
  }

  void stop(std::pair<struct timespec, struct timespec> & times)
  {
    gettime(times.second);
  }

  static
  double delta(const struct timespec & first, const struct timespec & second)
  {
    if (second.tv_nsec > first.tv_nsec)
      return (double) (second.tv_sec - first.tv_sec) + (double) (second.tv_nsec - first.tv_nsec) / (double) 1e9;
    else
      return (double) (second.tv_sec - first.tv_sec) - (double) (first.tv_nsec - second.tv_nsec) / (double) 1e9;
  }

  static
  double delta(const std::pair<struct timespec, struct timespec> & times)
  {
    return delta(times.first, times.second);
  }

};


FastTimerService::FastTimerService(const edm::ParameterSet & config, edm::ActivityRegistry & registry) :
  m_timer_id(               config.getUntrackedParameter<bool>(        "useRealTimeClock",     false) ? CLOCK_REALTIME : CLOCK_THREAD_CPUTIME_ID),
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

      /*
      // assume the path to be relative, and to have at least an item
      boost::filesystem::path::iterator item = m_dqm_path.begin();
      boost::filesystem::path path = * item++;
      path /= "EventInfo";
      while (item != m_dqm_path.end())
        path /= * item++;
      m_dqm_path = path;
      */
      m_dqm_path /= "EventInfo";
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
  out << "FastReport " << std::setw(10) << m_summary_source       / (double) m_summary_events << "                                  source"        << '\n';
  out << "FastReport " << std::setw(10) << m_summary_event        / (double) m_summary_events << "                                  event"         << '\n';
  out << "FastReport " << std::setw(10) << m_summary_all_paths    / (double) m_summary_events << "                                  all paths"     << '\n';
  out << "FastReport " << std::setw(10) << m_summary_all_endpaths / (double) m_summary_events << "                                  all endpaths"  << '\n';
  out << '\n';
  BOOST_FOREACH(std::string const & name, tns.getTrigPaths())
    out << "FastReport " << std::setw(10) << m_summary_paths[name]  / (double) m_summary_events << "                                  " << name << '\n';
  out << '\n';
  BOOST_FOREACH(std::string const & name, tns.getEndPaths())
    out << "FastReport " << std::setw(10) << m_summary_paths[name]  / (double) m_summary_events << "                                  " << name << '\n';
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

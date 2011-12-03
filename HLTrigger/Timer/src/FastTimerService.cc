//system headers
#ifdef __linux
#include <time.h>
#else
typedef int clockid_t;
#define CLOCK_REALTIME               0                                                                                                                                                                                        
#define CLOCK_MONOTONIC              1                                                                                                                                                                                        
#define CLOCK_PROCESS_CPUTIME_ID     2                                                                                                                                                                                        
#define CLOCK_THREAD_CPUTIME_ID      3                                                                                                                                                                                        
#endif

// C++ headers
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <set>
#include <map>

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
  // configuration
  m_timer_id(               config.getUntrackedParameter<bool>(        "useRealTimeClock",     false) ? CLOCK_REALTIME : CLOCK_THREAD_CPUTIME_ID),
  m_is_cpu_bound(           false ),
  m_enable_timing_modules(  config.getUntrackedParameter<bool>(        "enableTimingModules",  false) ),
  m_enable_timing_paths(    config.getUntrackedParameter<bool>(        "enableTimingPaths",    false) ),
  m_enable_timing_summary(  config.getUntrackedParameter<bool>(        "enableTimingSummary",  false) ), 
  m_enable_dqm(             config.getUntrackedParameter<bool>(        "enableDQM",            false) ), 
  m_enable_dqm_bylumi(      config.getUntrackedParameter<bool>(        "enableDQMbyLumi",      false) ), 
  // dqm configuration 
  m_dqm_time_range(         config.getUntrackedParameter<double>(      "dqmTimeRange",         1000.) ),   // ms
  m_dqm_time_resolution(    config.getUntrackedParameter<double>(      "dqmTimeResolution",       5.) ),   // ms
  m_dqm_path(               config.getUntrackedParameter<std::string>( "dqmPath",     "TimerService") ),
  // caching
  m_first_path(0),          // these are initialized at prePathBeginRun(), 
  m_last_path(0),           // to make sure we cache the correct pointers
  m_first_endpath(0),
  m_last_endpath(0),
  m_is_first_module(false),
  // per-event accounting
  m_event(0.),
  m_source(0.),
  m_all_paths(0.),
  m_all_endpaths(0.),
  // per-job summary
  m_summary_events(0),
  m_summary_event(0.),
  m_summary_source(0.),
  m_summary_all_paths(0.),
  m_summary_all_endpaths(0.),
  // DQM
  m_dqms(0),                // these are initialized at postBeginJob(),
  m_dqm_event(0),           // to make sure the DQM service has been loaded
  m_dqm_source(0),
  m_dqm_all_paths(0),
  m_dqm_all_endpaths(0),
  // per-path and per-module accounting
  m_current_path(0),
  m_paths(),
  m_modules()
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
  // watch per-module events if enabled
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
    m_paths[name];
  BOOST_FOREACH(std::string const & name, tns.getEndPaths())
    m_paths[name];

  // fill pathmap
  if (m_enable_timing_modules) {
    for (size_t i = 0; i < tns.getTrigPaths().size(); ++i) {
      std::string              const & name    = tns.getTrigPath(i);
      std::vector<std::string> const & modules = tns.getTrigPathModules(i);
      std::vector<ModuleInfo *>      & pathmap = m_paths[name].modules;
      pathmap.reserve( modules.size() );
      std::set<edm::ModuleDescription const *> pool;        // keep track of modules already inserted
      BOOST_FOREACH( std::string const & module, modules) {
        edm::ModuleDescription const * md = findModuleDescription(module);
        if (pool.find(md) != pool.end()) {
          // new module
          pool.insert(md);
          pathmap.push_back( & m_modules[md] );
        } else {
          // duplicate module
          pathmap.push_back( 0 );
        }
      }
    }
    for (size_t i = 0; i < tns.getEndPaths().size(); ++i) {
      std::string              const & name    = tns.getEndPath(i);
      std::vector<std::string> const & modules = tns.getEndPathModules(i);
      std::vector<ModuleInfo *>      & pathmap = m_paths[name].modules;
      pathmap.reserve( modules.size() );
      std::set<edm::ModuleDescription const *> pool;        // keep track of modules already inserted
      BOOST_FOREACH( std::string const & module, modules) {
        edm::ModuleDescription const * md = findModuleDescription(module);
        if (pool.find(md) != pool.end()) {
          // new module
          pool.insert(md);
          pathmap.push_back( & m_modules[md] );
        } else {
          // duplicate module
          pathmap.push_back( 0 );
        }
      }
    }

  }

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
    m_dqm_event         = m_dqms->book1D("event",        "Event",    bins, 0., m_dqm_time_range)->getTH1F();
    m_dqm_event         ->StatOverflows(true);
    m_dqm_source        = m_dqms->book1D("source",       "Source",   bins, 0., m_dqm_time_range)->getTH1F();
    m_dqm_source        ->StatOverflows(true);
    m_dqm_all_paths     = m_dqms->book1D("all_paths",    "Paths",    bins, 0., m_dqm_time_range)->getTH1F();
    m_dqm_all_paths     ->StatOverflows(true);
    m_dqm_all_endpaths  = m_dqms->book1D("all_endpaths", "EndPaths", bins, 0., m_dqm_time_range)->getTH1F();
    m_dqm_all_endpaths  ->StatOverflows(true);
    if (m_enable_timing_paths) {
      m_dqms->setCurrentFolder((m_dqm_path / "Paths").generic_string());
      BOOST_FOREACH(PathMap<PathInfo>::value_type & keyval, m_paths) {
        std::string const & pathname = keyval.first;
        PathInfo          & pathinfo = keyval.second;
        pathinfo.dqm_active = m_dqms->book1D(pathname, pathname, bins, 0., m_dqm_time_range)->getTH1F();
        pathinfo.dqm_active->StatOverflows(true);
      }
    }
    if (m_enable_timing_paths and m_enable_timing_modules) {
      m_dqms->setCurrentFolder((m_dqm_path / "Paths").generic_string());
      BOOST_FOREACH(PathMap<PathInfo>::value_type & keyval, m_paths) {
        std::string const & pathname = keyval.first;
        PathInfo          & pathinfo = keyval.second;
        pathinfo.dqm_premodules   = m_dqms->book1D(pathname + "_premodules",   pathname + " pre-modules overhead",   bins, 0., m_dqm_time_range)->getTH1F();
        pathinfo.dqm_premodules  ->StatOverflows(true);
        pathinfo.dqm_intermodules = m_dqms->book1D(pathname + "_intermodules", pathname + " inter-modules overhead", bins, 0., m_dqm_time_range)->getTH1F();
        pathinfo.dqm_intermodules->StatOverflows(true);
        pathinfo.dqm_postmodules  = m_dqms->book1D(pathname + "_postmodules",  pathname + " post-modules overhead",  bins, 0., m_dqm_time_range)->getTH1F();
        pathinfo.dqm_postmodules ->StatOverflows(true);
        pathinfo.dqm_total        = m_dqms->book1D(pathname + "_total",        pathname + " total time",             bins, 0., m_dqm_time_range)->getTH1F();
        pathinfo.dqm_total       ->StatOverflows(true);
      }
    }
    if (m_enable_timing_modules) {
      m_dqms->setCurrentFolder((m_dqm_path / "Modules").generic_string());
      BOOST_FOREACH(ModuleMap<ModuleInfo>::value_type & keyval, m_modules) {
        std::string const & label  = keyval.first->moduleLabel();
        ModuleInfo        & module = keyval.second;
        module.dqm_active = m_dqms->book1D(label, label, bins, 0., m_dqm_time_range)->getTH1F();
        module.dqm_active->StatOverflows(true);
      }
    }
  }
}

void FastTimerService::postEndJob() {
  if (not m_enable_timing_summary)
    return;

  edm::service::TriggerNamesService & tns = * edm::Service<edm::service::TriggerNamesService>();

  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  out << "FastReport " << (m_timer_id == CLOCK_REALTIME ? "(real time) " : "(CPU time)  ") << '\n';
  out << "FastReport              " << std::right << std::setw(10) << m_summary_source       / (double) m_summary_events << "  Source"        << '\n';
  out << "FastReport              " << std::right << std::setw(10) << m_summary_event        / (double) m_summary_events << "  Event"         << '\n';
  out << "FastReport              " << std::right << std::setw(10) << m_summary_all_paths    / (double) m_summary_events << "  all Paths"     << '\n';
  out << "FastReport              " << std::right << std::setw(10) << m_summary_all_endpaths / (double) m_summary_events << "  all EndPaths"  << '\n';
  if (m_enable_timing_paths and not m_enable_timing_modules) {
    out << '\n';
    out << "FastReport " << (m_timer_id == CLOCK_REALTIME ? "(real time) " : "(CPU time)  ")    << "     Active Path" << '\n';
    BOOST_FOREACH(std::string const & name, tns.getTrigPaths())
      out << "FastReport              " 
          << std::right << std::setw(10) << m_paths[name].summary_active  / (double) m_summary_events << "  " 
          << name << '\n';
    out << '\n';
    out << "FastReport " << (m_timer_id == CLOCK_REALTIME ? "(real time) " : "(CPU time)  ")    << "     Active EndPath" << '\n';
    BOOST_FOREACH(std::string const & name, tns.getEndPaths())
      out << "FastReport              " 
          << std::right << std::setw(10) << m_paths[name].summary_active  / (double) m_summary_events << "  " 
          << name << '\n';
  } else if (m_enable_timing_paths and m_enable_timing_modules) {
    out << '\n';
    out << "FastReport " << (m_timer_id == CLOCK_REALTIME ? "(real time) " : "(CPU time)  ")    << "   Pre-mods Inter-mods  Post-mods     Active      Total  Path" << '\n';
    BOOST_FOREACH(std::string const & name, tns.getTrigPaths())
      out << "FastReport              " 
          << std::right << std::setw(10) << m_paths[name].summary_premodules    / (double) m_summary_events << " "
          << std::right << std::setw(10) << m_paths[name].summary_intermodules  / (double) m_summary_events << " "
          << std::right << std::setw(10) << m_paths[name].summary_postmodules   / (double) m_summary_events << " "
          << std::right << std::setw(10) << m_paths[name].summary_active        / (double) m_summary_events << " "
          << std::right << std::setw(10) << m_paths[name].summary_total         / (double) m_summary_events << "  "
          << name << '\n';
    out << '\n';
    out << "FastReport " << (m_timer_id == CLOCK_REALTIME ? "(real time) " : "(CPU time)  ")    << "   Pre-mods Inter-mods  Post-mods     Active      Total  EndPath" << '\n';
    BOOST_FOREACH(std::string const & name, tns.getEndPaths())
      out << "FastReport              " 
          << std::right << std::setw(10) << m_paths[name].summary_premodules    / (double) m_summary_events << " "
          << std::right << std::setw(10) << m_paths[name].summary_intermodules  / (double) m_summary_events << " "
          << std::right << std::setw(10) << m_paths[name].summary_postmodules   / (double) m_summary_events << " "
          << std::right << std::setw(10) << m_paths[name].summary_active        / (double) m_summary_events << " "
          << std::right << std::setw(10) << m_paths[name].summary_total         / (double) m_summary_events << "  "
          << name << '\n';
  }
  edm::LogVerbatim("FastReport") << out.str();
}

void FastTimerService::preModuleBeginJob(edm::ModuleDescription const & module) {
  // this is ever called only if m_enable_timing_modules = true
  assert(m_enable_timing_modules);

  // allocate a counter for each module
  m_modules[& module];
}

void FastTimerService::preProcessEvent(edm::EventID const & id, edm::Timestamp const & stamp) {
  // new event, reset the per-event counter
  start(m_timer_event);

  // clear the event counters
  m_event        = 0;
  m_source       = 0;
  m_all_paths    = 0;
  m_all_endpaths = 0;
  for (PathMap<PathInfo>::iterator path = m_paths.begin(); path != m_paths.end(); ++path) {
    path->second.time_active       = 0.;
    path->second.time_premodules   = 0.;
    path->second.time_intermodules = 0.;
    path->second.time_postmodules  = 0.;
    path->second.time_total        = 0.;
  }
  for (ModuleMap<ModuleInfo>::iterator module = m_modules.begin(); module != m_modules.end(); ++module) {
    module->second.time_active     = 0.;
  }
}

void FastTimerService::postProcessEvent(edm::Event const & event, edm::EventSetup const & setup) {
  // stop the per-event timer, and account event time
  stop(m_timer_event);
  m_event = delta(m_timer_event);
  m_summary_event += m_event;
  if (m_dqms)
    m_dqm_event->Fill(m_event * 1000.);     // convert to ms
}

void FastTimerService::preSource() {
  start(m_timer_source);

  // keep track of the total number of events
  ++m_summary_events;
}

void FastTimerService::postSource() {
  stop(m_timer_source);
  m_source = delta(m_timer_source);
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
  // prepare to measure the time spent between the beginning of the path and the execution of the first module
  m_is_first_module = true;

  PathMap<PathInfo>::iterator keyval = m_paths.find(path);
  if (keyval != m_paths.end()) {
    m_current_path = & keyval->second;

    if (m_enable_timing_modules) {
      // reset the status of this path's modules
      BOOST_FOREACH(ModuleInfo * module, m_current_path->modules)
        if (module)
          module->has_just_run = false;
    }
  } else {
    // should never get here
    m_current_path = 0; 
    edm::LogError("FastTimerService") << "FastTimerService::preProcessPath: unexpected path " << path;
  }

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
  double active = delta(m_timer_path);

  // if enabled, account each (end)path
  if (m_enable_timing_paths) {
    PathInfo & pathinfo = * m_current_path;
    pathinfo.time_active = active;
    pathinfo.summary_active += active;
    if (m_dqms)
      pathinfo.dqm_active->Fill(active * 1000.);        // convert to ms
  
    // measure the time spent between the execution of the last module and the end of the path
    if (m_enable_timing_modules) {
      double post    = 0.;                // time spent after the last active module
      double inter   = 0.;                // time spent between modules
      double current = 0.;                // time spent in modules active in the current path
      double other   = 0.;                // time spent in modules part of this path, but active in other paths
      double total   = 0.;                // total per-path time, including modules already run as part of other paths

      size_t last_run = status.index();   // index of the last module run in this path
      for (size_t i = 0; i <= last_run; ++i) {
        ModuleInfo * module = pathinfo.modules[i];
        if (module == 0)
          // this is a module occurring more than once in the same path, skip it after the first occurrence
          continue;
        if (module->has_just_run) {
          current += module->time_active;
        } else {
          other   += module->time_active;
        }
      }
      
      if (m_is_first_module) {
        // no modules were active duruing this path, account all the time as "post-modules"
        pathinfo.time_premodules = 0.;
        if (m_dqms)
          pathinfo.dqm_premodules ->Fill(0.);
        post  = active;
        inter = 0.;
      } else {
        // time spent after the last active module
        post  = delta(m_timer_module.second, m_timer_path.second);
        // time spent between modules
        inter = active - pathinfo.time_premodules - current - post;
        if (std::abs(inter) < 1e-9)
          // take care of numeric precision and rounding errors - the timer is less precise than nanosecond resolution
          inter = 0.;
      }
      // total per-path time, including modules already run as part of other paths
      total = active + other;

      pathinfo.time_intermodules     = inter;
      pathinfo.time_postmodules      = post;
      pathinfo.time_total            = total;
      pathinfo.summary_intermodules += inter;
      pathinfo.summary_postmodules  += post;
      pathinfo.summary_total        += total;
      if (m_dqms) {
        pathinfo.dqm_intermodules->Fill(inter * 1000.);   // convert to ms
        pathinfo.dqm_postmodules ->Fill(post  * 1000.);   // convert to ms
        pathinfo.dqm_total       ->Fill(total * 1000.);   // convert to ms
      }
    }
  }

  if (& path == m_last_path) {
    // this is the last path, stop and account the "all paths" counter
    m_timer_paths.second = m_timer_path.second;
    m_all_paths = delta(m_timer_paths);
    m_summary_all_paths += m_all_paths;
    if (m_dqms)
      m_dqm_all_paths->Fill(m_all_paths * 1000.);               // convert to ms
  } else if (& path == m_last_endpath) {
    // this is the last endpath, stop and account the "all endpaths" counter
    m_timer_endpaths.second = m_timer_path.second;
    m_all_endpaths = delta(m_timer_endpaths);
    m_summary_all_endpaths += m_all_endpaths;
    if (m_dqms)
      m_dqm_all_endpaths->Fill(m_all_endpaths * 1000.);         // convert to ms
  }

}

void FastTimerService::preModule(edm::ModuleDescription const & module) {
  // this is ever called only if m_enable_timing_modules = true
  assert(m_enable_timing_modules);

  // time each module
  start(m_timer_module);

  if (m_is_first_module) {
    // measure the time spent between the beginning of the path and the execution of the first module
    m_is_first_module = false;
    double time = delta(m_timer_path.first, m_timer_module.first);
    m_current_path->time_premodules     = time;
    m_current_path->summary_premodules += time;
    if (m_dqms)
      m_current_path->dqm_premodules->Fill(time * 1000.);       // convert to ms
  }
}

void FastTimerService::postModule(edm::ModuleDescription const & module) {
  // this is ever called only if m_enable_timing_modules = true
  assert(m_enable_timing_modules);

  // time and account each module
  stop(m_timer_module);

  ModuleMap<ModuleInfo>::iterator keyval = m_modules.find(& module);
  if (keyval != m_modules.end()) {
    ModuleInfo & module = keyval->second;
    double time = delta(m_timer_module);
    module.has_just_run    = true;
    module.time_active     = time;
    module.summary_active += time;
    if (m_dqms)
      module.dqm_active->Fill(time * 1000.);                    // convert to ms
  } else {
    // should never get here
    edm::LogError("FastTimerService") << "FastTimerService::postModule: unexpected module " << module.moduleLabel();
  }
}
  
// find the module description associated to a module, by label
edm::ModuleDescription const * FastTimerService::findModuleDescription(const std::string & label) const {
  BOOST_FOREACH(ModuleMap<ModuleInfo>::value_type const & keyval, m_modules)
    if (keyval.first->moduleLabel() == label)
      return keyval.first;
  // not found
  return 0;
}


// declare FastTimerService as a framework Service
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(FastTimerService);

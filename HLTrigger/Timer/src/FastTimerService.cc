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
  m_current_path(0),
  m_has_just_run(),
  m_is_first_module(false),
  // per-event accounting
  m_event(0.),
  m_source(0.),
  m_all_paths(0.),
  m_all_endpaths(0.),
  m_paths(),
  m_paths_premodules(),
  m_paths_intermodules(),
  m_paths_postmodules(),
  m_paths_total(),
  m_modules(),
  // per-job summary
  m_summary_events(0),
  m_summary_event(0.),
  m_summary_source(0.),
  m_summary_all_paths(0.),
  m_summary_all_endpaths(0.),
  m_summary_paths(),
  m_summary_paths_premodules(),
  m_summary_paths_intermodules(),
  m_summary_paths_postmodules(),
  m_summary_paths_total(),
  m_summary_modules(),
  // DQM
  m_dqms(0),                // these are initialized at postBeginJob(),
  m_dqm_event(0),           // to make sure the DQM services have been loaded
  m_dqm_source(0),
  m_dqm_all_paths(0),
  m_dqm_all_endpaths(0),
  m_dqm_paths(),
  m_dqm_paths_premodules(),
  m_dqm_paths_intermodules(),
  m_dqm_paths_postmodules(),
  m_dqm_paths_total(),
  m_dqm_modules()
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
    m_paths[name] = 0.;
  BOOST_FOREACH(std::string const & name, tns.getEndPaths())
    m_paths[name] = 0.;

  // fill pathmap
  if (m_enable_timing_modules) {
    for (size_t i = 0; i < tns.getTrigPaths().size(); ++i) {
      std::string               const & name    = tns.getTrigPath(i);
      std::vector<std::string>  const & modules = tns.getTrigPathModules(i);
      std::vector<edm::ModuleDescription const *> & pathmap = m_pathmap[name];
      pathmap.reserve( modules.size() );
      BOOST_FOREACH( std::string const & module, modules) {
        edm::ModuleDescription const * md = findModuleDescription(module);
        if (std::find(pathmap.begin(), pathmap.end(), md) == pathmap.end())
          // new module
          pathmap.push_back( md );
        else
          // duplicate module
          pathmap.push_back( 0 );
      }
    }
    for (size_t i = 0; i < tns.getEndPaths().size(); ++i) {
      std::string               const & name    = tns.getEndPath(i);
      std::vector<std::string>  const & modules = std::vector<std::string>();
    /*
      FIXME this requires a change to the TriggerNamesService
      std::vector<std::string>  const & modules = tns.getEndPathModules(i);
    */
      std::vector<edm::ModuleDescription const *> & pathmap = m_pathmap[name];
      pathmap.reserve( modules.size() );
      BOOST_FOREACH( std::string const & module, modules) {
        edm::ModuleDescription const * md = findModuleDescription(module);
        if (std::find(pathmap.begin(), pathmap.end(), md) == pathmap.end())
          // new module
          pathmap.push_back( md );
        else
          // duplicate module
          pathmap.push_back( 0 );
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
    m_dqm_event         = m_dqms->book1D("event",        "Event",    bins, 0., m_dqm_time_range);
    m_dqm_source        = m_dqms->book1D("source",       "Source",   bins, 0., m_dqm_time_range);
    m_dqm_all_paths     = m_dqms->book1D("all_paths",    "Paths",    bins, 0., m_dqm_time_range);
    m_dqm_all_endpaths  = m_dqms->book1D("all_endpaths", "EndPaths", bins, 0., m_dqm_time_range);
    if (m_enable_timing_paths) {
      m_dqms->setCurrentFolder((m_dqm_path / "Paths").generic_string());
      BOOST_FOREACH(PathMap<double>::value_type const & path, m_paths) {
        m_dqm_paths[path.first] = m_dqms->book1D(path.first, path.first, bins, 0., m_dqm_time_range);
        m_dqm_paths[path.first]->getTH1()->StatOverflows(true);
      }
    }
    if (m_enable_timing_paths and m_enable_timing_modules) {
      m_dqms->setCurrentFolder((m_dqm_path / "Paths").generic_string());
      BOOST_FOREACH(PathMap<double>::value_type const & path, m_paths) {
        m_dqm_paths_premodules  [path.first] = m_dqms->book1D(path.first + "_premodules",   path.first + " pre-modules overhead",   bins, 0., m_dqm_time_range);
        m_dqm_paths_premodules  [path.first]->getTH1()->StatOverflows(true);
        m_dqm_paths_intermodules[path.first] = m_dqms->book1D(path.first + "_intermodules", path.first + " inter-modules overhead", bins, 0., m_dqm_time_range);
        m_dqm_paths_intermodules[path.first]->getTH1()->StatOverflows(true);
        m_dqm_paths_postmodules [path.first] = m_dqms->book1D(path.first + "_postmodules",  path.first + " post-modules overhead",  bins, 0., m_dqm_time_range);
        m_dqm_paths_postmodules [path.first]->getTH1()->StatOverflows(true);
        m_dqm_paths_total       [path.first] = m_dqms->book1D(path.first + "_total",        path.first + " total time",             bins, 0., m_dqm_time_range);
        m_dqm_paths_total       [path.first]->getTH1()->StatOverflows(true);
      }
    }
    if (m_enable_timing_modules) {
      m_dqms->setCurrentFolder((m_dqm_path / "Modules").generic_string());
      BOOST_FOREACH(ModuleMap<double>::value_type const & module, m_modules) {
        m_dqm_modules[module.first] = m_dqms->book1D(module.first->moduleLabel(), module.first->moduleLabel(), bins, 0., m_dqm_time_range);
        m_dqm_modules[module.first]->getTH1()->StatOverflows(true);
      }
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
          << std::right << std::setw(10) << m_summary_paths[name]  / (double) m_summary_events << "  " 
          << name << '\n';
    out << '\n';
    out << "FastReport " << (m_timer_id == CLOCK_REALTIME ? "(real time) " : "(CPU time)  ")    << "     Active EndPath" << '\n';
    BOOST_FOREACH(std::string const & name, tns.getEndPaths())
      out << "FastReport              " 
          << std::right << std::setw(10) << m_summary_paths[name]  / (double) m_summary_events << "  " 
          << name << '\n';
  } else if (m_enable_timing_paths and m_enable_timing_modules) {
    out << '\n';
    out << "FastReport " << (m_timer_id == CLOCK_REALTIME ? "(real time) " : "(CPU time)  ")    << "   Pre-mods Inter-mods  Post-mods     Active      Total  Path" << '\n';
    BOOST_FOREACH(std::string const & name, tns.getTrigPaths())
      out << "FastReport              " 
          << std::right << std::setw(10) << m_summary_paths_premodules[name]    / (double) m_summary_events << " "
          << std::right << std::setw(10) << m_summary_paths_intermodules[name]  / (double) m_summary_events << " "
          << std::right << std::setw(10) << m_summary_paths_postmodules[name]   / (double) m_summary_events << " "
          << std::right << std::setw(10) << m_summary_paths[name]               / (double) m_summary_events << " "
          << std::right << std::setw(10) << m_summary_paths_total[name]         / (double) m_summary_events << "  "
          << name << '\n';
    out << '\n';
    out << "FastReport " << (m_timer_id == CLOCK_REALTIME ? "(real time) " : "(CPU time)  ")    << "   Pre-mods Inter-mods  Post-mods     Active      Total  EndPath" << '\n';
    BOOST_FOREACH(std::string const & name, tns.getEndPaths())
      out << "FastReport              " 
          << std::right << std::setw(10) << m_summary_paths_premodules[name]    / (double) m_summary_events << " "
          << std::right << std::setw(10) << m_summary_paths_intermodules[name]  / (double) m_summary_events << " "
          << std::right << std::setw(10) << m_summary_paths_postmodules[name]   / (double) m_summary_events << " "
          << std::right << std::setw(10) << m_summary_paths[name]               / (double) m_summary_events << " "
          << std::right << std::setw(10) << m_summary_paths_total[name]         / (double) m_summary_events << "  "
          << name << '\n';
  }
  edm::LogVerbatim("FastReport") << out.str();
}

// this is ever called only if m_enable_timing_modules = true
void FastTimerService::preModuleBeginJob(edm::ModuleDescription const & module) {
  // allocate a counter for each module
  m_modules.insert(      std::make_pair(& module, (double) 0.) );
  m_has_just_run.insert( std::make_pair(& module, false) );
}

void FastTimerService::preProcessEvent(edm::EventID const & id, edm::Timestamp const & stamp) {
  // new event, reset the per-event counter
  start(m_timer_event);

  // clear the event counters
  m_event        = 0;
  m_source       = 0;
  m_all_paths    = 0;
  m_all_endpaths = 0;
  BOOST_FOREACH(PathMap<double>::value_type & keyval, m_paths)
    keyval.second = 0.;
  BOOST_FOREACH(PathMap<double>::value_type & keyval, m_paths_premodules)
    keyval.second = 0.;
  BOOST_FOREACH(PathMap<double>::value_type & keyval, m_paths_intermodules)
    keyval.second = 0.;
  BOOST_FOREACH(PathMap<double>::value_type & keyval, m_paths_postmodules)
    keyval.second = 0.;
  BOOST_FOREACH(PathMap<double>::value_type & keyval, m_paths_total)
    keyval.second = 0.;
  BOOST_FOREACH(ModuleMap<double>::value_type & keyval, m_modules)
    keyval.second = 0.;
}

void FastTimerService::postProcessEvent(edm::Event const & event, edm::EventSetup const & setup) {
  // stop the per-event timer, and account event time
  stop(m_timer_event);
  m_event = delta(m_timer_event);
  if (m_enable_timing_summary)
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
  // prepare to measure the time spent between the beginning of the path and the execution of the first module
  m_is_first_module = true;
  m_current_path = & path;

  // time each (end)path
  start(m_timer_path);

  if (& path == m_first_path) {
    // this is the first path, start the "all paths" counter
    m_timer_paths.first = m_timer_path.first;
  } else if (& path == m_first_endpath) {
    // this is the first endpath, start the "all paths" counter
    m_timer_endpaths.first = m_timer_path.first;
  }

  // reset the status of the modules
  if (m_enable_timing_modules) {
  BOOST_FOREACH(ModuleMap<bool>::value_type & value, m_has_just_run)
    value.second = false;
  }
}

void FastTimerService::postProcessPath(std::string const & path, edm::HLTPathStatus const & status) {
  // time each (end)path
  stop(m_timer_path);
  double active = delta(m_timer_path);

  // if enabled, account each (end)path
  if (m_enable_timing_paths) {
    PathMap<double>::iterator keyval;
    if ((keyval = m_paths.find(path)) != m_paths.end()) {
      keyval->second = active;
      if (m_enable_timing_summary)
        m_summary_paths[path] += active;
      if (m_dqms)
        m_dqm_paths[path]->Fill(active * 1000.);    // convert to ms
    
      // measure the time spent between the execution of the last module and the end of the path
      if (m_enable_timing_modules) {
        double post    = 0.;                // time spent after the last active module
        double inter   = 0.;                // time spent between modules
        double current = 0.;                // time spent in modules active in the current path
        double other   = 0.;                // time spent in modules part of this path, but active in other paths
        double total   = 0.;                // total per-path time, including modules already run as part of other paths

        std::vector<edm::ModuleDescription const *> const & modules = m_pathmap[path];
        size_t last_run = status.index();   // index of the last module run in this path
        if (not modules.empty())            // FIXME EndPaths will appear empty until the TriggerNamesService is fixed
        for (size_t i = 0; i <= last_run; ++i) {
          edm::ModuleDescription const * module = modules[i];
          if (module == 0)
            // this is a module occurring more than once in the same path, skip it after the first occurrence
            continue;
          if (m_has_just_run[module]) {
            current += m_modules[module];
          } else {
            other   += m_modules[module];
          }
        }
        
        if (m_is_first_module) {
          // no modules were active duruing this path, assign all the path time as "post-modules"
          m_paths_premodules[* m_current_path] = 0.;
          if (m_dqms)
            m_dqm_paths_premodules[* m_current_path]->Fill(0.);
          post  = active;
          inter = 0.;
        } else {
          // time spent after the last active module
          post  = delta(m_timer_module.second, m_timer_path.second);
          // time spent between modules
          inter = active - m_paths_premodules[* m_current_path] - current - post;
          if (std::abs(inter) < 1e-9)
            // take care of numeric precision and rounding errors - no timer is more precise than nanosecond resolution
            inter = 0.;
        }
        // total per-path time, including modules already run as part of other paths
        total = active + other;

        m_paths_intermodules[* m_current_path] = inter;
        m_paths_postmodules [* m_current_path] = post;
        m_paths_total       [* m_current_path] = total;
        if (m_enable_timing_summary) {
          m_summary_paths_intermodules[* m_current_path] += inter;
          m_summary_paths_postmodules [* m_current_path] += post;
          m_summary_paths_total       [* m_current_path] += total;
        }
        if (m_dqms) {
          m_dqm_paths_intermodules[* m_current_path]->Fill(inter * 1000.);  // convert to ms
          m_dqm_paths_postmodules [* m_current_path]->Fill(post  * 1000.);  // convert to ms
          m_dqm_paths_total       [* m_current_path]->Fill(total * 1000.);  // convert to ms
        }
      }
    } else {
      // should never get here
      edm::LogError("FastTimerService") << "FastTimerService::postProcessPath: unexpected path " << path;
    }
  }

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
    m_paths_premodules[* m_current_path] = time;
    if (m_enable_timing_summary)
      m_summary_paths_premodules[* m_current_path] += time;
    if (m_dqms)
      m_dqm_paths_premodules[* m_current_path]->Fill(time * 1000.);     // convert to ms
  }
}

void FastTimerService::postModule(edm::ModuleDescription const & module) {
  // this is ever called only if m_enable_timing_modules = true
  assert(m_enable_timing_modules);

  // time and account each module
  stop(m_timer_module);

  ModuleMap<double>::iterator keyval = m_modules.find(& module);
  if (keyval != m_modules.end()) {
    keyval->second = delta(m_timer_module);
    m_has_just_run[& module] = true;
    if (m_enable_timing_summary)
      m_summary_modules[& module] += keyval->second;
    if (m_dqms)
      m_dqm_modules[& module]->Fill(keyval->second * 1000.);    // convert to ms
  } else {
    // should never get here
    edm::LogError("FastTimerService") << "FastTimerService::postModule: unexpected module " << module.moduleLabel();
  }
}
  
// find the module description associated to a module, by label
edm::ModuleDescription const * FastTimerService::findModuleDescription(const std::string & label) const {
  BOOST_FOREACH(ModuleMap<double>::value_type const & keyval, m_modules)
    if (keyval.first->moduleLabel() == label)
      return keyval.first;
  // not found
  return 0;
}


// declare FastTimerService as a framework Service
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(FastTimerService);

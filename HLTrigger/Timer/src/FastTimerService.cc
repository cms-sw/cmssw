// FIXME
// we are by-passing the ME's when filling the plots, so we might need to call the ME's update() by hand


// system headers
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
#include <tr1/unordered_set>
#include <tr1/unordered_map>

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
  m_skip_first_path(        config.getUntrackedParameter<bool>(        "skipFirstPath",        false) ),
  // dqm configuration
  m_dqm_eventtime_range(        config.getUntrackedParameter<double>(      "dqmEventTimeRange",         1000.) ),   // ms
  m_dqm_eventtime_resolution(   config.getUntrackedParameter<double>(      "dqmEventTimeResolution",       5.) ),   // ms
  m_dqm_pathtime_range(         config.getUntrackedParameter<double>(      "dqmPathTimeRange",           100.) ),   // ms
  m_dqm_pathtime_resolution(    config.getUntrackedParameter<double>(      "dqmPathTimeResolution",       0.5) ),   // ms
  m_dqm_moduletime_range(       config.getUntrackedParameter<double>(      "dqmModuleTimeRange",          40.) ),   // ms
  m_dqm_moduletime_resolution(  config.getUntrackedParameter<double>(      "dqmModuleTimeResolution",     0.2) ),   // ms

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
  m_modules(),
  m_cache_paths(),
  m_cache_modules()
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
  //edm::LogImportant("FastTimerService") << __func__ << "()";

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

  // cache all pathinfo objects
  if (m_enable_timing_paths) {
    m_cache_paths.reserve(m_paths.size());
    BOOST_FOREACH(PathMap<PathInfo>::value_type & keyval, m_paths)
      m_cache_paths.push_back(& keyval.second);
  }

  // cache all moduleinfo objects
  if (m_enable_timing_modules) {
    m_cache_modules.reserve(m_modules.size());
    BOOST_FOREACH(ModuleMap<ModuleInfo>::value_type & keyval, m_modules)
      m_cache_modules.push_back(& keyval.second);
  }

  // associate to each path all the modules it contains
  if (m_enable_timing_paths and m_enable_timing_modules) {
    for (size_t i = 0; i < tns.getTrigPaths().size(); ++i)
      fillPathMap( tns.getTrigPath(i), tns.getTrigPathModules(i) );
    for (size_t i = 0; i < tns.getEndPaths().size(); ++i)
      fillPathMap( tns.getEndPath(i), tns.getEndPathModules(i) );
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
      boost::filesystem::path dqm_path(m_dqm_path);
      boost::filesystem::path::iterator item = dqm_path.begin();
      boost::filesystem::path path = * item++;
      path /= "EventInfo";
      while (item != dqm_path.end())
        path /= * item++;
      m_dqm_path = path.generic_string();
    }

    // book MonitorElement's
    int eventbins = (int) std::ceil(m_dqm_eventtime_range / m_dqm_eventtime_resolution);
    int pathbins = (int) std::ceil(m_dqm_pathtime_range / m_dqm_pathtime_resolution);
    int modulebins = (int) std::ceil(m_dqm_moduletime_range / m_dqm_moduletime_resolution);

    m_dqms->setCurrentFolder(m_dqm_path);
    m_dqm_event         = m_dqms->book1D("event",        "Event",    eventbins, 0., m_dqm_eventtime_range)->getTH1F();
    m_dqm_event         ->StatOverflows(true);
    m_dqm_source        = m_dqms->book1D("source",       "Source",   pathbins, 0., m_dqm_pathtime_range)->getTH1F();
    m_dqm_source        ->StatOverflows(true);
    m_dqm_all_paths     = m_dqms->book1D("all_paths",    "Paths",    eventbins, 0., m_dqm_eventtime_range)->getTH1F();
    m_dqm_all_paths     ->StatOverflows(true);
    m_dqm_all_endpaths  = m_dqms->book1D("all_endpaths", "EndPaths", pathbins, 0., m_dqm_pathtime_range)->getTH1F();
    m_dqm_all_endpaths  ->StatOverflows(true);
    // these are actually filled in the harvesting step - but that may happen in a separate step, which no longer has all the information about the endpaths
    size_t size_p = tns.getTrigPaths().size();
    size_t size_e = tns.getEndPaths().size();
    size_t size = size_p + size_e;
    TH1F * path_active_time = m_dqms->book1D("path_active_time", "Additional time spent in each path", size, -0.5, size-0.5)->getTH1F();
    TH1F * path_total_time  = m_dqms->book1D("path_total_time",  "Total time spent in each path",      size, -0.5, size-0.5)->getTH1F();
    for (size_t i = 0; i < size_p; ++i) {
      std::string const & label = tns.getTrigPath(i);
      path_active_time->GetXaxis()->SetBinLabel(i + 1, label.c_str());
      path_total_time ->GetXaxis()->SetBinLabel(i + 1, label.c_str());
    }
    for (size_t i = 0; i < size_e; ++i) {
      std::string const & label = tns.getEndPath(i);
      path_active_time->GetXaxis()->SetBinLabel(i + size_p + 1, label.c_str());
      path_total_time ->GetXaxis()->SetBinLabel(i + size_p + 1, label.c_str());
    }

    if (m_enable_timing_paths) {
      m_dqms->setCurrentFolder((m_dqm_path + "/Paths"));
      BOOST_FOREACH(PathMap<PathInfo>::value_type & keyval, m_paths) {
        std::string const & pathname = keyval.first;
        PathInfo          & pathinfo = keyval.second;
        pathinfo.dqm_active = m_dqms->book1D(pathname + "_active", pathname + " active time", pathbins, 0., m_dqm_pathtime_range)->getTH1F();
        pathinfo.dqm_active->StatOverflows(true);
      }
    }

    if (m_enable_timing_modules) {
      m_dqms->setCurrentFolder((m_dqm_path + "/Modules"));
      BOOST_FOREACH(ModuleMap<ModuleInfo>::value_type & keyval, m_modules) {
        std::string const & label  = keyval.first->moduleLabel();
        ModuleInfo        & module = keyval.second;
        module.dqm_active = m_dqms->book1D(label, label, modulebins, 0., m_dqm_moduletime_range)->getTH1F();
        module.dqm_active->StatOverflows(true);
      }
    }

    if (m_enable_timing_paths and m_enable_timing_modules) {
      m_dqms->setCurrentFolder((m_dqm_path + "/Paths"));
      BOOST_FOREACH(PathMap<PathInfo>::value_type & keyval, m_paths) {
        std::string const & pathname = keyval.first;
        PathInfo          & pathinfo = keyval.second;
#ifdef FASTTIMERSERVICE_DETAILED_OVERHEAD_ACCOUNTING
        pathinfo.dqm_premodules   = m_dqms->book1D(pathname + "_premodules",   pathname + " pre-modules overhead",   modulebins, 0., m_dqm_moduletime_range)->getTH1F();
        pathinfo.dqm_premodules  ->StatOverflows(true);
        pathinfo.dqm_intermodules = m_dqms->book1D(pathname + "_intermodules", pathname + " inter-modules overhead", modulebins, 0., m_dqm_moduletime_range)->getTH1F();
        pathinfo.dqm_intermodules->StatOverflows(true);
        pathinfo.dqm_postmodules  = m_dqms->book1D(pathname + "_postmodules",  pathname + " post-modules overhead",  modulebins, 0., m_dqm_moduletime_range)->getTH1F();
        pathinfo.dqm_postmodules ->StatOverflows(true);
#else
        pathinfo.dqm_overhead     = m_dqms->book1D(pathname + "_overhead",     pathname + " overhead time",          pathbins, 0., m_dqm_pathtime_range)->getTH1F();
        pathinfo.dqm_overhead    ->StatOverflows(true);
#endif
        pathinfo.dqm_total        = m_dqms->book1D(pathname + "_total",        pathname + " total time",             pathbins, 0., m_dqm_pathtime_range)->getTH1F();
        pathinfo.dqm_total       ->StatOverflows(true);
        
        // book histograms for modules-in-paths statistics
        size_t id;
        std::vector<std::string> const & modules = ((id = tns.findTrigPath(pathname)) != tns.getTrigPaths().size()) ? tns.getTrigPathModules(id) :
                                                   ((id = tns.findEndPath(pathname))  != tns.getEndPaths().size())  ? tns.getEndPathModules(id)  :
                                                   std::vector<std::string>();
        pathinfo.dqm_module_counter = m_dqms->book1D(pathname + "_module_counter", pathname + " module counter", modules.size(), -0.5, modules.size() - 0.5)->getTH1F();
        pathinfo.dqm_module_active  = m_dqms->book1D(pathname + "_module_active",  pathname + " module active",  modules.size(), -0.5, modules.size() - 0.5)->getTH1F();
        pathinfo.dqm_module_total   = m_dqms->book1D(pathname + "_module_total",   pathname + " module total",   modules.size(), -0.5, modules.size() - 0.5)->getTH1F();
        // find module labels
        for (size_t i = 0; i < modules.size(); ++i) {
          if (pathinfo.modules[i]) {
            pathinfo.dqm_module_counter->GetXaxis()->SetBinLabel( i+1, modules[i].c_str() );
            pathinfo.dqm_module_active ->GetXaxis()->SetBinLabel( i+1, modules[i].c_str() );
            pathinfo.dqm_module_total  ->GetXaxis()->SetBinLabel( i+1, modules[i].c_str() );
          } else {
            pathinfo.dqm_module_counter->GetXaxis()->SetBinLabel( i+1, "(dup.)" );
            pathinfo.dqm_module_active ->GetXaxis()->SetBinLabel( i+1, "(dup.)" );
            pathinfo.dqm_module_total  ->GetXaxis()->SetBinLabel( i+1, "(dup.)" );
          }
        }
      }
    }

  }
}

void FastTimerService::postEndJob() {
  //edm::LogImportant("FastTimerService") << __func__ << "()";

  if (m_enable_timing_summary) {
    // print a timing sumary for the whle job
    edm::service::TriggerNamesService & tns = * edm::Service<edm::service::TriggerNamesService>();

    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "FastReport " << (m_timer_id == CLOCK_REALTIME ? "(real time) " : "(CPU time)  ") << '\n';
    out << "FastReport              " << std::right << std::setw(10) << m_summary_source       / (double) m_summary_events << "  Source"        << '\n';
    out << "FastReport              " << std::right << std::setw(10) << m_summary_event        / (double) m_summary_events << "  Event"         << '\n';
    out << "FastReport              " << std::right << std::setw(10) << m_summary_all_paths    / (double) m_summary_events << "  all Paths"     << '\n';
    out << "FastReport              " << std::right << std::setw(10) << m_summary_all_endpaths / (double) m_summary_events << "  all EndPaths"  << '\n';
    if (m_enable_timing_modules) {
      double modules_total = 0.;
      BOOST_FOREACH(ModuleMap<ModuleInfo>::value_type & keyval, m_modules)
        modules_total += keyval.second.summary_active;
      out << "FastReport              " << std::right << std::setw(10) << modules_total          / (double) m_summary_events << "  all Modules"   << '\n';
    }
    out << '\n';
    if (m_enable_timing_paths and not m_enable_timing_modules) {
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
#ifdef FASTTIMERSERVICE_DETAILED_OVERHEAD_ACCOUNTING
      out << "FastReport " << (m_timer_id == CLOCK_REALTIME ? "(real time) " : "(CPU time)  ")    << "     Active   Pre-mods Inter-mods  Post-mods      Total  Path" << '\n';
#else
      out << "FastReport " << (m_timer_id == CLOCK_REALTIME ? "(real time) " : "(CPU time)  ")    << "     Active   Overhead      Total  Path" << '\n';
#endif
      BOOST_FOREACH(std::string const & name, tns.getTrigPaths())
        out << "FastReport              "
            << std::right << std::setw(10) << m_paths[name].summary_active        / (double) m_summary_events << " "
#ifdef FASTTIMERSERVICE_DETAILED_OVERHEAD_ACCOUNTING
            << std::right << std::setw(10) << m_paths[name].summary_premodules    / (double) m_summary_events << " "
            << std::right << std::setw(10) << m_paths[name].summary_intermodules  / (double) m_summary_events << " "
            << std::right << std::setw(10) << m_paths[name].summary_postmodules   / (double) m_summary_events << " "
#else
            << std::right << std::setw(10) << m_paths[name].summary_overhead      / (double) m_summary_events << "  "
#endif
            << std::right << std::setw(10) << m_paths[name].summary_total         / (double) m_summary_events << "  "
            << name << '\n';
      out << '\n';
#ifdef FASTTIMERSERVICE_DETAILED_OVERHEAD_ACCOUNTING
      out << "FastReport " << (m_timer_id == CLOCK_REALTIME ? "(real time) " : "(CPU time)  ")    << "     Active   Pre-mods Inter-mods  Post-mods      Total  Path" << '\n';
#else
      out << "FastReport " << (m_timer_id == CLOCK_REALTIME ? "(real time) " : "(CPU time)  ")    << "     Active   Overhead      Total  Path" << '\n';
#endif
      BOOST_FOREACH(std::string const & name, tns.getEndPaths())
        out << "FastReport              "
            << std::right << std::setw(10) << m_paths[name].summary_active        / (double) m_summary_events << " "
#ifdef FASTTIMERSERVICE_DETAILED_OVERHEAD_ACCOUNTING
            << std::right << std::setw(10) << m_paths[name].summary_premodules    / (double) m_summary_events << " "
            << std::right << std::setw(10) << m_paths[name].summary_intermodules  / (double) m_summary_events << " "
            << std::right << std::setw(10) << m_paths[name].summary_postmodules   / (double) m_summary_events << " "
#else
            << std::right << std::setw(10) << m_paths[name].summary_overhead      / (double) m_summary_events << "  "
#endif
            << std::right << std::setw(10) << m_paths[name].summary_total         / (double) m_summary_events << "  "
            << name << '\n';
    }
    out << '\n';
    if (m_enable_timing_modules) {
      out << "FastReport " << (m_timer_id == CLOCK_REALTIME ? "(real time) " : "(CPU time)  ")    << "     Active  Module" << '\n';
      BOOST_FOREACH(ModuleMap<ModuleInfo>::value_type & keyval, m_modules) {
        std::string const & label  = keyval.first->moduleLabel();
        ModuleInfo  const & module = keyval.second;
        out << "FastReport              " << std::right << std::setw(10) << module.summary_active  / (double) m_summary_events << "  " << label << '\n';
      }
      out << "FastReport " << (m_timer_id == CLOCK_REALTIME ? "(real time) " : "(CPU time)  ")    << "     Active  Module" << '\n';
    }
    out << '\n';
    edm::LogVerbatim("FastReport") << out.str();
  }

  // needed for the DAQ when reconfiguring between runs
  reset();
}

void FastTimerService::reset() {
  // caching
  m_first_path = 0;          // these are initialized at prePathBeginRun(),
  m_last_path = 0;           // to make sure we cache the correct pointers
  m_first_endpath = 0;
  m_last_endpath = 0;
  m_is_first_module = false;
  // per-event accounting
  m_event = 0.;
  m_source = 0.;
  m_all_paths = 0.;
  m_all_endpaths = 0.;
  // per-job summary
  m_summary_events = 0;
  m_summary_event = 0.;
  m_summary_source = 0.;
  m_summary_all_paths = 0.;
  m_summary_all_endpaths = 0.;
  // DQM
  m_dqms = 0;
  // the DAQ destroys and re-creates the DQM and DQMStore services at each reconfigure, so we don't need to clean them up
  m_dqm_event = 0;
  m_dqm_source = 0;
  m_dqm_all_paths = 0;
  m_dqm_all_endpaths = 0;
  // per-path and per-module accounting
  m_current_path = 0;
  m_paths.clear();          // this should destroy all PathInfo objects and Reset the associated plots
  m_modules.clear();        // this should destroy all ModuleInfo objects and Reset the associated plots
  m_cache_paths.clear();
  m_cache_modules.clear();
}

void FastTimerService::preModuleBeginJob(edm::ModuleDescription const & module) {
  //edm::LogImportant("FastTimerService") << __func__ << "(" << & module << ")";
  //edm::LogImportant("FastTimerService") << "module " << module.moduleLabel() << " @ " << & module;

  // this is ever called only if m_enable_timing_modules = true
  assert(m_enable_timing_modules);

  // allocate a counter for each module
  m_modules[& module];
}

void FastTimerService::preProcessEvent(edm::EventID const & id, edm::Timestamp const & stamp) {
  //edm::LogImportant("FastTimerService") << __func__ << "(...)";

  // new event, reset the per-event counter
  start(m_timer_event);

  // clear the event counters
  m_event        = 0;
  m_all_paths    = 0;
  m_all_endpaths = 0;
  BOOST_FOREACH(PathInfo * path, m_cache_paths) {
    path->time_active       = 0.;
#ifdef FASTTIMERSERVICE_DETAILED_OVERHEAD_ACCOUNTING
    path->time_premodules   = 0.;
    path->time_intermodules = 0.;
    path->time_postmodules  = 0.;
#endif
    path->time_total        = 0.;
  }
  BOOST_FOREACH(ModuleInfo * module, m_cache_modules) {
    module->time_active     = 0.;
  }
}

void FastTimerService::postProcessEvent(edm::Event const & event, edm::EventSetup const & setup) {
  //edm::LogImportant("FastTimerService") << __func__ << "(...)";

  // stop the per-event timer, and account event time
  stop(m_timer_event);
  m_event = delta(m_timer_event);
  m_summary_event += m_event;
  if (m_dqms)
    m_dqm_event->Fill(m_event * 1000.);     // convert to ms
}

void FastTimerService::preSource() {
  //edm::LogImportant("FastTimerService") << __func__ << "()";

  start(m_timer_source);

  // clear the event counters
  m_source = 0;

  // keep track of the total number of events
  ++m_summary_events;
}

void FastTimerService::postSource() {
  //edm::LogImportant("FastTimerService") << __func__ << "()";

  stop(m_timer_source);
  m_source = delta(m_timer_source);
  m_summary_source += m_source;
  if (m_dqms)
    m_dqm_source->Fill(m_source * 1000.);                       // convert to ms
}

void FastTimerService::prePathBeginRun(std::string const & path ) {
  //edm::LogImportant("FastTimerService") << __func__ << "(" << path << ")";

  // cache the pointers to the names of the first and last path and endpath
  edm::service::TriggerNamesService & tns = * edm::Service<edm::service::TriggerNamesService>();
  if (not tns.getTrigPaths().empty()) {
    if (path == tns.getTrigPaths().at(0) && !m_skip_first_path)
      m_first_path = & path;
    else if (path == tns.getTrigPaths().at(1) && m_skip_first_path)
      m_first_path = & path;
    if (path == tns.getTrigPaths().back())
      m_last_path = & path;
  }
  if (not tns.getEndPaths().empty()) {
    if (path == tns.getEndPaths().front())
      m_first_endpath = & path;
    if (path == tns.getEndPaths().back())
      m_last_endpath = & path;
  }
}

void FastTimerService::preProcessPath(std::string const & path ) {
  //edm::LogImportant("FastTimerService") << __func__ << "(" << path << ")";

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
  //edm::LogImportant("FastTimerService") << __func__ << "(" << path << ", ...)";

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
#ifdef FASTTIMERSERVICE_DETAILED_OVERHEAD_ACCOUNTING
      double pre      = 0.;                 // time spent before the first active module
      double inter    = 0.;                 // time spent between active modules
      double post     = 0.;                 // time spent after the last active module
#else
      double overhead = 0.;                 // time spent before, between, or after modules
#endif
      double current  = 0.;                 // time spent in modules active in the current path
      double total    = active;             // total per-path time, including modules already run as part of other paths

      size_t last_run = status.index();     // index of the last module run in this path
      for (size_t i = 0; i <= last_run; ++i) {
        ModuleInfo * module = pathinfo.modules[i];
        // fill the counter also for duplicate modules (to properly extract rejection information)
        pathinfo.dqm_module_counter->Fill(i);
        if (module == 0)
          // this is a module occurring more than once in the same path, skip it after the first occurrence
          continue;
        // fill the total time for all non-duplicate modules
        pathinfo.dqm_module_total->Fill(i, module->time_active);
        if (module->has_just_run) {
          // fill the active time only for module actually running in this path
          pathinfo.dqm_module_active->Fill(i, module->time_active);
          current += module->time_active;
        } else {
          total   += module->time_active;
        }
      }
      if (status.accept())
        pathinfo.dqm_module_counter->Fill(pathinfo.modules.size());

      if (m_is_first_module) {
        // no modules were active duruing this path, account all the time as overhead
#ifdef FASTTIMERSERVICE_DETAILED_OVERHEAD_ACCOUNTING
        pre      = 0.;
        inter    = 0.;
        post     = active;
#else
        overhead = active;
#endif
      } else {
        // extract overhead information
#ifdef FASTTIMERSERVICE_DETAILED_OVERHEAD_ACCOUNTING
        pre      = delta(m_timer_path.first, m_timer_first_module);
        post     = delta(m_timer_module.second, m_timer_path.second);
        inter    = active - pre - current - post;
        // take care of numeric precision and rounding errors - the timer is less precise than nanosecond resolution
        if (std::abs(inter) < 1e-9)
          inter = 0.;
#else
        overhead = active - current;
        // take care of numeric precision and rounding errors - the timer is less precise than nanosecond resolution
        if (std::abs(overhead) < 1e-9)
          overhead = 0.;
#endif
      }

#ifdef FASTTIMERSERVICE_DETAILED_OVERHEAD_ACCOUNTING
      pathinfo.time_premodules       = pre;
      pathinfo.time_intermodules     = inter;
      pathinfo.time_postmodules      = post;
#else
      pathinfo.time_overhead         = overhead;
#endif
      pathinfo.time_total            = total;
#ifdef FASTTIMERSERVICE_DETAILED_OVERHEAD_ACCOUNTING
      pathinfo.summary_premodules   += pre;
      pathinfo.summary_intermodules += inter;
      pathinfo.summary_postmodules  += post;
#else
      pathinfo.summary_overhead     += overhead;
#endif
      pathinfo.summary_total        += total;
      if (m_dqms) {
#ifdef FASTTIMERSERVICE_DETAILED_OVERHEAD_ACCOUNTING
        pathinfo.dqm_premodules  ->Fill(pre      * 1000.);      // convert to ms
        pathinfo.dqm_intermodules->Fill(inter    * 1000.);      // convert to ms
        pathinfo.dqm_postmodules ->Fill(post     * 1000.);      // convert to ms
#else
        pathinfo.dqm_overhead    ->Fill(overhead * 1000.);      // convert to ms
#endif
        pathinfo.dqm_total       ->Fill(total    * 1000.);      // convert to ms
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
  //edm::LogImportant("FastTimerService") << __func__ << "(" << & module << ")";

  // this is ever called only if m_enable_timing_modules = true
  assert(m_enable_timing_modules);

#ifdef FASTTIMERSERVICE_DETAILED_OVERHEAD_ACCOUNTING
  // time each module
  start(m_timer_module);

  if (m_is_first_module) {
    m_is_first_module = false;

    // measure the time spent between the beginning of the path and the execution of the first module
    m_timer_first_module = m_timer_module.first;
  } 
#else
  if (m_is_first_module) {
    m_is_first_module = false;

    // track the start of the first module of the path
    start(m_timer_module);

    // measure the time spent between the beginning of the path and the execution of the first module
    m_timer_first_module = m_timer_module.first;
  } else {
    // use the end time of the previous module (assume no inter-module overhead)
    m_timer_module.first = m_timer_module.second;
  }
#endif
}

void FastTimerService::postModule(edm::ModuleDescription const & module) {
  //edm::LogImportant("FastTimerService") << __func__ << "(" << & module << ")";

  // this is ever called only if m_enable_timing_modules = true
  assert(m_enable_timing_modules);

  // time and account each module
  stop(m_timer_module);

  ModuleMap<ModuleInfo>::iterator keyval = m_modules.find(& module);
  if (keyval != m_modules.end()) {
    double time = delta(m_timer_module);
    ModuleInfo & module = keyval->second;
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
  // no descriptions are associated to an empty label
  if (label.empty())
    return 0;

  // fix the name of negated or ignored modules
  std::string const & target = (label[0] == '!' or label[0] == '-') ? label.substr(1) : label;

  BOOST_FOREACH(ModuleMap<ModuleInfo>::value_type const & keyval, m_modules) {
    if (keyval.first == 0) {
      // this should never happen, but it would cause a segmentation fault to insert a null pointer in the path map, se we explicitly check for it and skip it
      edm::LogError("FastTimerService") << "FastTimerService::findModuleDescription: invalid entry detected in ModuleMap<ModuleInfo> m_modules, skipping";
      continue;
    }
    if (keyval.first->moduleLabel() == target) {
      return keyval.first;
    }
  }
  // not found
  return 0;
}

// associate to a path all the modules it contains
void FastTimerService::fillPathMap(std::string const & name, std::vector<std::string> const & modules) {
  std::vector<ModuleInfo *> & pathmap = m_paths[name].modules;
  pathmap.reserve( modules.size() );
  std::tr1::unordered_set<edm::ModuleDescription const *> pool;        // keep track of inserted modules
  BOOST_FOREACH( std::string const & module, modules) {
    edm::ModuleDescription const * md = findModuleDescription(module);
    if (md == 0) {
      // no matching module was found
      pathmap.push_back( 0 );
    } else if (pool.insert(md).second) {
      // new module
      pathmap.push_back( & m_modules[md] );
    } else {
      // duplicate module
      pathmap.push_back( 0 );
    }
  }
}

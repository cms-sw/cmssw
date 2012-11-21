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
#include <limits>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <tr1/unordered_set>
#include <tr1/unordered_map>

// boost headers
#include <boost/format.hpp>

// CMSSW headers
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/HLTPathStatus.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "HLTrigger/Timer/interface/FastTimerService.h"
#include "HLTrigger/Timer/interface/CPUAffinity.h"


// file-static methods to fill a vector of strings with "(dup.) (...)" entries
static
void fill_dups(std::vector<std::string> & dups, unsigned int size) {
  dups.reserve(size);
  for (unsigned int i = dups.size(); i < size; ++i)
    dups.push_back( (boost::format("(dup.) (%d)") % i).str() );
}


FastTimerService::FastTimerService(const edm::ParameterSet & config, edm::ActivityRegistry & registry) :
  // configuration
  m_timer_id(                    config.getUntrackedParameter<bool>(     "useRealTimeClock"         ) ? CLOCK_REALTIME : CLOCK_THREAD_CPUTIME_ID),
  m_is_cpu_bound(                false ),
  m_enable_timing_paths(         config.getUntrackedParameter<bool>(     "enableTimingPaths"        ) ),
  m_enable_timing_modules(       config.getUntrackedParameter<bool>(     "enableTimingModules"      ) ),
  m_enable_timing_exclusive(     config.getUntrackedParameter<bool>(     "enableTimingExclusive"    ) ),
  m_enable_timing_summary(       config.getUntrackedParameter<bool>(     "enableTimingSummary"      ) ),
  m_skip_first_path(             config.getUntrackedParameter<bool>(     "skipFirstPath"            ) ),
  // dqm configuration
  m_enable_dqm(                  config.getUntrackedParameter<bool>(     "enableDQM"                ) ),
  m_enable_dqm_bypath_active(    config.getUntrackedParameter<bool>(     "enableDQMbyPathActive"    ) ),
  m_enable_dqm_bypath_total(     config.getUntrackedParameter<bool>(     "enableDQMbyPathTotal"     ) ),
  m_enable_dqm_bypath_overhead(  config.getUntrackedParameter<bool>(     "enableDQMbyPathOverhead"  ) ),
  m_enable_dqm_bypath_details(   config.getUntrackedParameter<bool>(     "enableDQMbyPathDetails"   ) ),
  m_enable_dqm_bypath_counters(  config.getUntrackedParameter<bool>(     "enableDQMbyPathCounters"  ) ),
  m_enable_dqm_bypath_exclusive( config.getUntrackedParameter<bool>(     "enableDQMbyPathExclusive" ) ),
  m_enable_dqm_bymodule(         config.getUntrackedParameter<bool>(     "enableDQMbyModule"        ) ),
  m_enable_dqm_summary(          config.getUntrackedParameter<bool>(     "enableDQMSummary"         ) ),
  m_enable_dqm_byluminosity(     config.getUntrackedParameter<bool>(     "enableDQMbyLuminosity"    ) ),   
  m_enable_dqm_byls(             config.existsAs<bool>("enableDQMbyLumiSection", false) ? 
                                    config.getUntrackedParameter<bool>("enableDQMbyLumiSection") : 
                                    ( edm::LogWarning("Configuration") << "enableDQMbyLumi is deprecated, please use enableDQMbyLumiSection instead", config.getUntrackedParameter<bool>("enableDQMbyLumi") ) 
                                 ),
  m_enable_dqm_bynproc(          config.getUntrackedParameter<bool>(     "enableDQMbyProcesses"     ) ),
  m_nproc_enabled(               false ),
  m_dqm_eventtime_range(         config.getUntrackedParameter<double>(   "dqmTimeRange"             ) ),            // ms
  m_dqm_eventtime_resolution(    config.getUntrackedParameter<double>(   "dqmTimeResolution"        ) ),            // ms
  m_dqm_pathtime_range(          config.getUntrackedParameter<double>(   "dqmPathTimeRange"         ) ),            // ms
  m_dqm_pathtime_resolution(     config.getUntrackedParameter<double>(   "dqmPathTimeResolution"    ) ),            // ms
  m_dqm_moduletime_range(        config.getUntrackedParameter<double>(   "dqmModuleTimeRange"       ) ),            // ms
  m_dqm_moduletime_resolution(   config.getUntrackedParameter<double>(   "dqmModuleTimeResolution"  ) ),            // ms
  m_dqm_luminosity_range(        config.getUntrackedParameter<double>(   "dqmLuminosityRange"       ) / 1.e30),     // cm-2 s-1
  m_dqm_luminosity_resolution(   config.getUntrackedParameter<double>(   "dqmLuminosityResolution"  ) / 1.e30),     // cm-2 s-1
  m_dqm_ls_range(                config.getUntrackedParameter<uint32_t>( "dqmLumiSectionsRange"     ) ),            // lumisections
  m_dqm_path(                    config.getUntrackedParameter<std::string>("dqmPath" ) ),
  m_luminosity_label(            config.getUntrackedParameter<edm::InputTag>("luminosityProduct") ),                // SCAL unpacker
  m_supported_processes(         config.getUntrackedParameter<std::vector<unsigned int> >("supportedProcesses") ),  // 8, 24, 32
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
  m_interpaths(0.),
  // per-job summary
  m_summary_events(0),
  m_summary_event(0.),
  m_summary_source(0.),
  m_summary_all_paths(0.),
  m_summary_all_endpaths(0.),
  m_summary_interpaths(0.),
  // DQM - these are initialized at postBeginJob(), to make sure the DQM service has been loaded
  m_dqms(0),
  // event summary plots
  m_dqm_event(0),
  m_dqm_source(0),
  m_dqm_all_paths(0),
  m_dqm_all_endpaths(0),
  m_dqm_interpaths(0),
  // event summary plots - summed over nodes with the same number of processes
  m_dqm_nproc_event(0),
  m_dqm_nproc_source(0),
  m_dqm_nproc_all_paths(0),
  m_dqm_nproc_all_endpaths(0),
  m_dqm_nproc_interpaths(0),
  // plots by path
  m_dqm_paths_active_time(0),
  m_dqm_paths_total_time(0),
  m_dqm_paths_exclusive_time(0),
  m_dqm_paths_interpaths(0),
  // plots per lumisection
  m_dqm_byls_event(0),
  m_dqm_byls_source(0),
  m_dqm_byls_all_paths(0),
  m_dqm_byls_all_endpaths(0),
  m_dqm_byls_interpaths(0),
  // plots per lumisection - summed over nodes with the same number of processes
  m_dqm_nproc_byls_event(0),
  m_dqm_nproc_byls_source(0),
  m_dqm_nproc_byls_all_paths(0),
  m_dqm_nproc_byls_all_endpaths(0),
  m_dqm_nproc_byls_interpaths(0),
  // plots vs. instantaneous luminosity
  m_dqm_byluminosity_event(0),
  m_dqm_byluminosity_source(0),
  m_dqm_byluminosity_all_paths(0),
  m_dqm_byluminosity_all_endpaths(0),
  m_dqm_byluminosity_interpaths(0),
  // plots vs. instantaneous luminosity - summed over nodes with the same number of processes
  m_dqm_nproc_byluminosity_event(0),
  m_dqm_nproc_byluminosity_source(0),
  m_dqm_nproc_byluminosity_all_paths(0),
  m_dqm_nproc_byluminosity_all_endpaths(0),
  m_dqm_nproc_byluminosity_interpaths(0),
  // per-path and per-module accounting
  m_current_path(0),
  m_paths(),
  m_modules(),
  m_cache_paths(),
  m_cache_modules()
{
  // enable timers if required by DQM plots
  m_enable_timing_paths     = m_enable_timing_paths     or m_enable_dqm_bypath_active or m_enable_dqm_bypath_total or m_enable_dqm_bypath_overhead or m_enable_dqm_bypath_details or m_enable_dqm_bypath_counters or m_enable_dqm_bypath_exclusive;
  m_enable_timing_modules   = m_enable_timing_modules   or m_enable_dqm_bymodule      or m_enable_dqm_bypath_total or m_enable_dqm_bypath_overhead or m_enable_dqm_bypath_details or m_enable_dqm_bypath_counters or m_enable_dqm_bypath_exclusive;
  m_enable_timing_exclusive = m_enable_timing_exclusive or m_enable_dqm_bypath_exclusive;

  registry.watchPreModuleBeginJob( this, & FastTimerService::preModuleBeginJob );
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
    registry.watchPreModule(         this, & FastTimerService::preModule );
    registry.watchPostModule(        this, & FastTimerService::postModule );
  }

#if defined(__APPLE__) || defined (__MACH__)
  host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &m_clock_port);
#endif // defined(__APPLE__) || defined(__MACH__)
}

FastTimerService::~FastTimerService()
{
#if defined(__APPLE__) || defined (__MACH__)
  mach_port_deallocate(mach_task_self(), m_clock_port);
#endif // defined(__APPLE__) || defined(__MACH__)
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
  uint32_t size_p = tns.getTrigPaths().size();
  uint32_t size_e = tns.getEndPaths().size();
  uint32_t size = size_p + size_e;
  for (uint32_t i = 0; i < size_p; ++i) {
    std::string const & label = tns.getTrigPath(i);
    m_paths[label].index = i;
  }
  for (uint32_t i = 0; i < size_e; ++i) {
    std::string const & label = tns.getEndPath(i);
    m_paths[label].index = size_p + i;
  }

  // cache all pathinfo objects
  m_cache_paths.reserve(m_paths.size());
  for (auto & keyval: m_paths)
    m_cache_paths.push_back(& keyval.second);

  // cache all moduleinfo objects
  m_cache_modules.reserve(m_modules.size());
  for (auto & keyval: m_modules)
    m_cache_modules.push_back(& keyval.second);

  // associate to each path all the modules it contains
  for (uint32_t i = 0; i < tns.getTrigPaths().size(); ++i)
    fillPathMap( tns.getTrigPath(i), tns.getTrigPathModules(i) );
  for (uint32_t i = 0; i < tns.getEndPaths().size(); ++i)
    fillPathMap( tns.getEndPath(i), tns.getEndPathModules(i) );

  if (m_enable_dqm)
    // load the DQM store
    m_dqms = edm::Service<DQMStore>().operator->();

  if (m_dqms) {
    // book MonitorElement's
    int eventbins  = (int) std::ceil(m_dqm_eventtime_range  / m_dqm_eventtime_resolution);
    int pathbins   = (int) std::ceil(m_dqm_pathtime_range   / m_dqm_pathtime_resolution);
    int modulebins = (int) std::ceil(m_dqm_moduletime_range / m_dqm_moduletime_resolution);
    int lumibins   = (int) std::ceil(m_dqm_luminosity_range / m_dqm_luminosity_resolution);

    // event summary plots
    if (m_enable_dqm_summary) {
      m_dqms->setCurrentFolder(m_dqm_path);
      m_dqm_event         = m_dqms->book1D("event",        "Event processing time",    eventbins, 0., m_dqm_eventtime_range)->getTH1F();
      m_dqm_event         ->StatOverflows(true);
      m_dqm_event         ->SetXTitle("processing time [ms]");
      m_dqm_source        = m_dqms->book1D("source",       "Source processing time",   pathbins,  0., m_dqm_pathtime_range)->getTH1F();
      m_dqm_source        ->StatOverflows(true);
      m_dqm_source        ->SetXTitle("processing time [ms]");
      m_dqm_all_paths     = m_dqms->book1D("all_paths",    "Paths processing time",    eventbins, 0., m_dqm_eventtime_range)->getTH1F();
      m_dqm_all_paths     ->StatOverflows(true);
      m_dqm_all_paths     ->SetXTitle("processing time [ms]");
      m_dqm_all_endpaths  = m_dqms->book1D("all_endpaths", "EndPaths processing time", pathbins,  0., m_dqm_pathtime_range)->getTH1F();
      m_dqm_all_endpaths  ->StatOverflows(true);
      m_dqm_all_endpaths  ->SetXTitle("processing time [ms]");
      m_dqm_interpaths    = m_dqms->book1D("interpaths",   "Time spent between paths", pathbins,  0., m_dqm_eventtime_range)->getTH1F();
      m_dqm_interpaths    ->StatOverflows(true);
      m_dqm_interpaths    ->SetXTitle("processing time [ms]");
    }

    // event summary plots - summed over nodes with the same number of processes
    if (m_enable_dqm_summary and m_enable_dqm_bynproc) {
      TH1F * plot;
      for (int n : m_supported_processes) {
        m_dqms->setCurrentFolder( (boost::format("%s/Running %d processes") % m_dqm_path % n).str() );
        plot = m_dqms->book1D("event",        "Event processing time",    eventbins, 0., m_dqm_eventtime_range)->getTH1F();
        plot->StatOverflows(true);
        plot->SetXTitle("processing time [ms]");
        plot = m_dqms->book1D("source",       "Source processing time",   pathbins,  0., m_dqm_pathtime_range)->getTH1F();
        plot->StatOverflows(true);
        plot->SetXTitle("processing time [ms]");
        plot = m_dqms->book1D("all_paths",    "Paths processing time",    eventbins, 0., m_dqm_eventtime_range)->getTH1F();
        plot->StatOverflows(true);
        plot->SetXTitle("processing time [ms]");
        plot = m_dqms->book1D("all_endpaths", "EndPaths processing time", pathbins,  0., m_dqm_pathtime_range)->getTH1F();
        plot->StatOverflows(true);
        plot->SetXTitle("processing time [ms]");
        plot = m_dqms->book1D("interpaths",   "Time spent between paths", pathbins,  0., m_dqm_eventtime_range)->getTH1F();
        plot->StatOverflows(true);
        plot->SetXTitle("processing time [ms]");
      }
    }

    // plots by path
    m_dqms->setCurrentFolder(m_dqm_path);
    m_dqm_paths_active_time     = m_dqms->bookProfile("paths_active_time",    "Additional time spent in each path", size, -0.5, size-0.5, pathbins, 0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
    m_dqm_paths_active_time     ->StatOverflows(true);
    m_dqm_paths_active_time     ->SetYTitle("processing time [ms]");
    m_dqm_paths_total_time      = m_dqms->bookProfile("paths_total_time",     "Total time spent in each path",      size, -0.5, size-0.5, pathbins, 0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
    m_dqm_paths_total_time      ->StatOverflows(true);
    m_dqm_paths_total_time      ->SetYTitle("processing time [ms]");
    m_dqm_paths_exclusive_time  = m_dqms->bookProfile("paths_exclusive_time", "Exclusive time spent in each path",  size, -0.5, size-0.5, pathbins, 0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
    m_dqm_paths_exclusive_time  ->StatOverflows(true);
    m_dqm_paths_exclusive_time  ->SetYTitle("processing time [ms]");
    m_dqm_paths_interpaths      = m_dqms->bookProfile("paths_interpaths",     "Time spent between each path",       size, -0.5, size-0.5, pathbins, 0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
    m_dqm_paths_interpaths      ->StatOverflows(true);
    m_dqm_paths_interpaths      ->SetYTitle("processing time [ms]");

    for (uint32_t i = 0; i < size_p; ++i) {
      std::string const & label = tns.getTrigPath(i);
      m_dqm_paths_active_time    ->GetXaxis()->SetBinLabel(i + 1, label.c_str());
      m_dqm_paths_total_time     ->GetXaxis()->SetBinLabel(i + 1, label.c_str());
      m_dqm_paths_exclusive_time ->GetXaxis()->SetBinLabel(i + 1, label.c_str());
      m_dqm_paths_interpaths     ->GetXaxis()->SetBinLabel(i + 1, label.c_str());
    }
    for (uint32_t i = 0; i < size_e; ++i) {
      std::string const & label = tns.getEndPath(i);
      m_dqm_paths_active_time    ->GetXaxis()->SetBinLabel(i + size_p + 1, label.c_str());
      m_dqm_paths_total_time     ->GetXaxis()->SetBinLabel(i + size_p + 1, label.c_str());
      m_dqm_paths_exclusive_time ->GetXaxis()->SetBinLabel(i + size_p + 1, label.c_str());
      m_dqm_paths_interpaths     ->GetXaxis()->SetBinLabel(i + size_p + 1, label.c_str());
    }

    // per-lumisection plots
    if (m_enable_dqm_byls) {
      m_dqms->setCurrentFolder(m_dqm_path);
      m_dqm_byls_event        = m_dqms->bookProfile("event_byls",        "Event processing time, by LumiSection",    m_dqm_ls_range, 0.5, m_dqm_ls_range+0.5, eventbins, 0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
      m_dqm_byls_event        ->StatOverflows(true);
      m_dqm_byls_event        ->SetXTitle("lumisection");
      m_dqm_byls_event        ->SetYTitle("processing time [ms]");
      m_dqm_byls_source       = m_dqms->bookProfile("source_byls",       "Source processing time, by LumiSection",   m_dqm_ls_range, 0.5, m_dqm_ls_range+0.5, pathbins,  0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
      m_dqm_byls_source       ->StatOverflows(true);
      m_dqm_byls_source       ->SetXTitle("lumisection");
      m_dqm_byls_source       ->SetYTitle("processing time [ms]");
      m_dqm_byls_all_paths    = m_dqms->bookProfile("all_paths_byls",    "Paths processing time, by LumiSection",    m_dqm_ls_range, 0.5, m_dqm_ls_range+0.5, eventbins, 0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
      m_dqm_byls_all_paths    ->StatOverflows(true);
      m_dqm_byls_all_paths    ->SetXTitle("lumisection");
      m_dqm_byls_all_paths    ->SetYTitle("processing time [ms]");
      m_dqm_byls_all_endpaths = m_dqms->bookProfile("all_endpaths_byls", "EndPaths processing time, by LumiSection", m_dqm_ls_range, 0.5, m_dqm_ls_range+0.5, pathbins,  0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
      m_dqm_byls_all_endpaths ->StatOverflows(true);
      m_dqm_byls_all_endpaths ->SetXTitle("lumisection");
      m_dqm_byls_all_endpaths ->SetYTitle("processing time [ms]");
      m_dqm_byls_interpaths   = m_dqms->bookProfile("interpaths_byls",   "Time spent between paths, by LumiSection", m_dqm_ls_range, 0.5, m_dqm_ls_range+0.5, pathbins,  0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
      m_dqm_byls_interpaths   ->StatOverflows(true);
      m_dqm_byls_interpaths   ->SetXTitle("lumisection");
      m_dqm_byls_interpaths   ->SetYTitle("processing time [ms]");
    }

    // per-lumisection plots
    if (m_enable_dqm_byls and m_enable_dqm_bynproc) {
      TProfile * plot;
      for (int n : m_supported_processes) {
        m_dqms->setCurrentFolder( (boost::format("%s/Running %d processes") % m_dqm_path % n).str() );
        plot = m_dqms->bookProfile("event_byls",        "Event processing time, by LumiSection",    m_dqm_ls_range, 0.5, m_dqm_ls_range+0.5, eventbins, 0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
        plot->StatOverflows(true);
        plot->SetXTitle("lumisection");
        plot->SetYTitle("processing time [ms]");
        plot = m_dqms->bookProfile("source_byls",       "Source processing time, by LumiSection",   m_dqm_ls_range, 0.5, m_dqm_ls_range+0.5, pathbins,  0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
        plot->StatOverflows(true);
        plot->SetXTitle("lumisection");
        plot->SetYTitle("processing time [ms]");
        plot = m_dqms->bookProfile("all_paths_byls",    "Paths processing time, by LumiSection",    m_dqm_ls_range, 0.5, m_dqm_ls_range+0.5, eventbins, 0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
        plot->StatOverflows(true);
        plot->SetXTitle("lumisection");
        plot->SetYTitle("processing time [ms]");
        plot = m_dqms->bookProfile("all_endpaths_byls", "EndPaths processing time, by LumiSection", m_dqm_ls_range, 0.5, m_dqm_ls_range+0.5, pathbins,  0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
        plot->StatOverflows(true);
        plot->SetXTitle("lumisection");
        plot->SetYTitle("processing time [ms]");
        plot = m_dqms->bookProfile("interpaths_byls",   "Time spent between paths, by LumiSection", m_dqm_ls_range, 0.5, m_dqm_ls_range+0.5, pathbins,  0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
        plot->StatOverflows(true);
        plot->SetXTitle("lumisection");
        plot->SetYTitle("processing time [ms]");
      }
    }

    // plots vs. instantaneous luminosity
    if (m_enable_dqm_byluminosity) {
      m_dqms->setCurrentFolder(m_dqm_path);
      m_dqm_byluminosity_event        = m_dqms->bookProfile("event_byluminosity",        "Event processing time vs. instantaneous luminosity",    lumibins, 0., m_dqm_luminosity_range, eventbins, 0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
      m_dqm_byluminosity_event        ->StatOverflows(true);
      m_dqm_byluminosity_event        ->SetXTitle("instantaneous luminosity [10^{30} cm^{-2}s^{-1}]");
      m_dqm_byluminosity_event        ->SetYTitle("processing time [ms]");
      m_dqm_byluminosity_source       = m_dqms->bookProfile("source_byluminosity",       "Source processing time vs. instantaneous luminosity",   lumibins, 0., m_dqm_luminosity_range, pathbins,  0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
      m_dqm_byluminosity_source       ->StatOverflows(true);
      m_dqm_byluminosity_source       ->SetXTitle("instantaneous luminosity [10^{30} cm^{-2}s^{-1}]");
      m_dqm_byluminosity_source       ->SetYTitle("processing time [ms]");
      m_dqm_byluminosity_all_paths    = m_dqms->bookProfile("all_paths_byluminosity",    "Paths processing time vs. instantaneous luminosity",    lumibins, 0., m_dqm_luminosity_range, eventbins, 0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
      m_dqm_byluminosity_all_paths    ->StatOverflows(true);
      m_dqm_byluminosity_all_paths    ->SetXTitle("instantaneous luminosity [10^{30} cm^{-2}s^{-1}]");
      m_dqm_byluminosity_all_paths    ->SetYTitle("processing time [ms]");
      m_dqm_byluminosity_all_endpaths = m_dqms->bookProfile("all_endpaths_byluminosity", "EndPaths processing time vs. instantaneous luminosity", lumibins, 0., m_dqm_luminosity_range, pathbins,  0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
      m_dqm_byluminosity_all_endpaths ->StatOverflows(true);
      m_dqm_byluminosity_all_endpaths ->SetXTitle("instantaneous luminosity [10^{30} cm^{-2}s^{-1}]");
      m_dqm_byluminosity_all_endpaths ->SetYTitle("processing time [ms]");
      m_dqm_byluminosity_interpaths   = m_dqms->bookProfile("interpaths_byluminosity",   "Time spent between paths vs. instantaneous luminosity", lumibins, 0., m_dqm_luminosity_range, pathbins,  0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
      m_dqm_byluminosity_interpaths   ->StatOverflows(true);
      m_dqm_byluminosity_interpaths   ->SetXTitle("instantaneous luminosity [10^{30} cm^{-2}s^{-1}]");
      m_dqm_byluminosity_interpaths   ->SetYTitle("processing time [ms]");
    }

    // plots vs. instantaneous luminosity
    if (m_enable_dqm_byluminosity and m_enable_dqm_bynproc) {
      TProfile * plot;
      for (int n : m_supported_processes) {
        m_dqms->setCurrentFolder( (boost::format("%s/Running %d processes") % m_dqm_path % n).str() );
        plot = m_dqms->bookProfile("event_byluminosity",        "Event processing time vs. instantaneous luminosity",    lumibins, 0., m_dqm_luminosity_range, eventbins, 0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
        plot->StatOverflows(true);
        plot->SetYTitle("processing time [ms]");
        plot->SetXTitle("instantaneous luminosity [10^{30} cm^{-2}s^{-1}]");
        plot = m_dqms->bookProfile("source_byluminosity",       "Source processing time vs. instantaneous luminosity",   lumibins, 0., m_dqm_luminosity_range, pathbins,  0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
        plot->StatOverflows(true);
        plot->SetYTitle("processing time [ms]");
        plot->SetXTitle("instantaneous luminosity [10^{30} cm^{-2}s^{-1}]");
        plot = m_dqms->bookProfile("all_paths_byluminosity",    "Paths processing time vs. instantaneous luminosity",    lumibins, 0., m_dqm_luminosity_range, eventbins, 0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
        plot->StatOverflows(true);
        plot->SetYTitle("processing time [ms]");
        plot->SetXTitle("instantaneous luminosity [10^{30} cm^{-2}s^{-1}]");
        plot = m_dqms->bookProfile("all_endpaths_byluminosity", "EndPaths processing time vs. instantaneous luminosity", lumibins, 0., m_dqm_luminosity_range, pathbins,  0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
        plot->StatOverflows(true);
        plot->SetYTitle("processing time [ms]");
        plot->SetXTitle("instantaneous luminosity [10^{30} cm^{-2}s^{-1}]");
        plot = m_dqms->bookProfile("interpaths_byluminosity",   "Time spent between paths vs. instantaneous luminosity", lumibins, 0., m_dqm_luminosity_range, pathbins,  0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
        plot->StatOverflows(true);
        plot->SetYTitle("processing time [ms]");
        plot->SetXTitle("instantaneous luminosity [10^{30} cm^{-2}s^{-1}]");
      }
    }

    // per-path and per-module accounting
    if (m_enable_timing_paths) {
      m_dqms->setCurrentFolder((m_dqm_path + "/Paths"));
      for (auto & keyval: m_paths) {
        std::string const & pathname = keyval.first;
        PathInfo          & pathinfo = keyval.second;

        if (m_enable_dqm_bypath_active) {
          pathinfo.dqm_active       = m_dqms->book1D(pathname + "_active",       pathname + " active time",            pathbins, 0., m_dqm_pathtime_range)->getTH1F();
          pathinfo.dqm_active       ->StatOverflows(true);
          pathinfo.dqm_active       ->SetXTitle("processing time [ms]");
        }

        if (m_enable_dqm_bypath_total) {
          pathinfo.dqm_total        = m_dqms->book1D(pathname + "_total",        pathname + " total time",             pathbins, 0., m_dqm_pathtime_range)->getTH1F();
          pathinfo.dqm_total        ->StatOverflows(true);
          pathinfo.dqm_total        ->SetXTitle("processing time [ms]");
        }

        if (m_enable_dqm_bypath_overhead) {
          pathinfo.dqm_premodules   = m_dqms->book1D(pathname + "_premodules",   pathname + " pre-modules overhead",   modulebins, 0., m_dqm_moduletime_range)->getTH1F();
          pathinfo.dqm_premodules   ->StatOverflows(true);
          pathinfo.dqm_premodules   ->SetXTitle("processing time [ms]");
          pathinfo.dqm_intermodules = m_dqms->book1D(pathname + "_intermodules", pathname + " inter-modules overhead", modulebins, 0., m_dqm_moduletime_range)->getTH1F();
          pathinfo.dqm_intermodules ->StatOverflows(true);
          pathinfo.dqm_intermodules ->SetXTitle("processing time [ms]");
          pathinfo.dqm_postmodules  = m_dqms->book1D(pathname + "_postmodules",  pathname + " post-modules overhead",  modulebins, 0., m_dqm_moduletime_range)->getTH1F();
          pathinfo.dqm_postmodules  ->StatOverflows(true);
          pathinfo.dqm_postmodules  ->SetXTitle("processing time [ms]");
          pathinfo.dqm_overhead     = m_dqms->book1D(pathname + "_overhead",     pathname + " overhead time",          modulebins, 0., m_dqm_moduletime_range)->getTH1F();
          pathinfo.dqm_overhead     ->StatOverflows(true);
          pathinfo.dqm_overhead     ->SetXTitle("processing time [ms]");
        }

        if (m_enable_dqm_bypath_details or m_enable_dqm_bypath_counters) {
          // book histograms for modules-in-paths statistics

          // find histograms X-axis labels
          uint32_t id;
          std::vector<std::string> const & modules = ((id = tns.findTrigPath(pathname)) != tns.getTrigPaths().size()) ? tns.getTrigPathModules(id) :
                                                     ((id = tns.findEndPath(pathname))  != tns.getEndPaths().size())  ? tns.getEndPathModules(id)  :
                                                     std::vector<std::string>();

          static std::vector<std::string> dup;
          if (modules.size() > dup.size())
            fill_dups(dup, modules.size());

          std::vector<const char *> labels(modules.size(), nullptr);
          for (uint32_t i = 0; i < modules.size(); ++i)
            labels[i] = (pathinfo.modules[i]) ? modules[i].c_str() : dup[i].c_str();
          
          // book counter histograms
          if (m_enable_dqm_bypath_counters) {
            pathinfo.dqm_module_counter = m_dqms->book1D(pathname + "_module_counter", pathname + " module counter", modules.size(), -0.5, modules.size() - 0.5)->getTH1F();
            // find module labels
            for (uint32_t i = 0; i < modules.size(); ++i) {
              pathinfo.dqm_module_counter->GetXaxis()->SetBinLabel( i+1, labels[i] );
            }
          }
          // book detailed timing histograms
          if (m_enable_dqm_bypath_details) {
            pathinfo.dqm_module_active  = m_dqms->book1D(pathname + "_module_active",  pathname + " module active",  modules.size(), -0.5, modules.size() - 0.5)->getTH1F();
            pathinfo.dqm_module_active  ->SetYTitle("cumulative processing time [ms]");
            pathinfo.dqm_module_total   = m_dqms->book1D(pathname + "_module_total",   pathname + " module total",   modules.size(), -0.5, modules.size() - 0.5)->getTH1F();
            pathinfo.dqm_module_total   ->SetYTitle("cumulative processing time [ms]");
            // find module labels
            for (uint32_t i = 0; i < modules.size(); ++i) {
              pathinfo.dqm_module_active ->GetXaxis()->SetBinLabel( i+1, labels[i] );
              pathinfo.dqm_module_total  ->GetXaxis()->SetBinLabel( i+1, labels[i] );
            }
          }
        }

        // book exclusive path time histograms
        if (m_enable_dqm_bypath_exclusive) {
          pathinfo.dqm_exclusive = m_dqms->book1D(pathname + "_exclusive", pathname + " exclusive time", pathbins, 0., m_dqm_pathtime_range)->getTH1F();
          pathinfo.dqm_exclusive ->StatOverflows(true);
          pathinfo.dqm_exclusive ->SetYTitle("processing time [ms]");
        }

      }
    }
   
    if (m_enable_dqm_bymodule) {
      m_dqms->setCurrentFolder((m_dqm_path + "/Modules"));
      for (auto & keyval: m_modules) {
        std::string const & label  = keyval.first->moduleLabel();
        ModuleInfo        & module = keyval.second;
        module.dqm_active = m_dqms->book1D(label, label, modulebins, 0., m_dqm_moduletime_range)->getTH1F();
        module.dqm_active->StatOverflows(true);
        module.dqm_active->SetYTitle("processing time [ms]");
      }
    }

  }
}

void FastTimerService::setNumberOfProcesses(unsigned int procs) {
  // the service is not configured to support plots by number of processes
  if (not m_enable_dqm or not m_enable_dqm_bynproc)
    return;

  // the DQM store has not been configured yet
  if (not m_dqms)
    return;

  // check if the service is configured to support this number of processes
  if (std::find(m_supported_processes.begin(), m_supported_processes.end(), procs) == m_supported_processes.end()) {
    m_nproc_enabled = false;
    // event summary plots - summed over nodes with the same number of processes
    m_dqm_nproc_event                     = 0;
    m_dqm_nproc_source                    = 0;
    m_dqm_nproc_all_paths                 = 0;
    m_dqm_nproc_all_endpaths              = 0;
    m_dqm_nproc_interpaths                = 0;
    // plots per lumisection - summed over nodes with the same number of processes
    m_dqm_nproc_byls_event                = 0;
    m_dqm_nproc_byls_source               = 0;
    m_dqm_nproc_byls_all_paths            = 0;
    m_dqm_nproc_byls_all_endpaths         = 0;
    m_dqm_nproc_byls_interpaths           = 0;
    // plots vs. instantaneous luminosity - summed over nodes with the same number of processes
    m_dqm_nproc_byluminosity_event        = 0;
    m_dqm_nproc_byluminosity_source       = 0;
    m_dqm_nproc_byluminosity_all_paths    = 0;
    m_dqm_nproc_byluminosity_all_endpaths = 0;
    m_dqm_nproc_byluminosity_interpaths   = 0;
  } else {
    m_nproc_enabled = true;
    // event summary plots - summed over nodes with the same number of processes
    m_dqm_nproc_event                     = m_dqms->get( (boost::format("%s/Running %d processes/%s") % m_dqm_path % procs % "event"                    ).str() )->getTH1F();
    m_dqm_nproc_source                    = m_dqms->get( (boost::format("%s/Running %d processes/%s") % m_dqm_path % procs % "source"                   ).str() )->getTH1F();
    m_dqm_nproc_all_paths                 = m_dqms->get( (boost::format("%s/Running %d processes/%s") % m_dqm_path % procs % "all_paths"                ).str() )->getTH1F();
    m_dqm_nproc_all_endpaths              = m_dqms->get( (boost::format("%s/Running %d processes/%s") % m_dqm_path % procs % "all_endpaths"             ).str() )->getTH1F();
    m_dqm_nproc_interpaths                = m_dqms->get( (boost::format("%s/Running %d processes/%s") % m_dqm_path % procs % "interpaths"               ).str() )->getTH1F();
    // plots per lumisection - summed over nodes with the same number of processes
    m_dqm_nproc_byls_event                = m_dqms->get( (boost::format("%s/Running %d processes/%s") % m_dqm_path % procs % "event_byls"               ).str() )->getTProfile();
    m_dqm_nproc_byls_source               = m_dqms->get( (boost::format("%s/Running %d processes/%s") % m_dqm_path % procs % "source_byls"              ).str() )->getTProfile();
    m_dqm_nproc_byls_all_paths            = m_dqms->get( (boost::format("%s/Running %d processes/%s") % m_dqm_path % procs % "all_paths_byls"           ).str() )->getTProfile();
    m_dqm_nproc_byls_all_endpaths         = m_dqms->get( (boost::format("%s/Running %d processes/%s") % m_dqm_path % procs % "all_endpaths_byls"        ).str() )->getTProfile();
    m_dqm_nproc_byls_interpaths           = m_dqms->get( (boost::format("%s/Running %d processes/%s") % m_dqm_path % procs % "interpaths_byls"          ).str() )->getTProfile();
    // plots vs. instantaneous luminosity - summed over nodes with the same number of processes
    m_dqm_nproc_byluminosity_event        = m_dqms->get( (boost::format("%s/Running %d processes/%s") % m_dqm_path % procs % "event_byluminosity"       ).str() )->getTProfile();
    m_dqm_nproc_byluminosity_source       = m_dqms->get( (boost::format("%s/Running %d processes/%s") % m_dqm_path % procs % "source_byluminosity"      ).str() )->getTProfile();
    m_dqm_nproc_byluminosity_all_paths    = m_dqms->get( (boost::format("%s/Running %d processes/%s") % m_dqm_path % procs % "all_paths_byluminosity"   ).str() )->getTProfile();
    m_dqm_nproc_byluminosity_all_endpaths = m_dqms->get( (boost::format("%s/Running %d processes/%s") % m_dqm_path % procs % "all_endpaths_byluminosity").str() )->getTProfile();
    m_dqm_nproc_byluminosity_interpaths   = m_dqms->get( (boost::format("%s/Running %d processes/%s") % m_dqm_path % procs % "interpaths_byluminosity"  ).str() )->getTProfile();
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
    out << "FastReport              " << std::right << std::setw(10) << m_summary_interpaths   / (double) m_summary_events << "  between paths" << '\n';
    if (m_enable_timing_modules) {
      double modules_total = 0.;
      for (auto & keyval: m_modules)
        modules_total += keyval.second.summary_active;
      out << "FastReport              " << std::right << std::setw(10) << modules_total          / (double) m_summary_events << "  all Modules"   << '\n';
    }
    out << '\n';
    if (m_enable_timing_paths and not m_enable_timing_modules) {
      out << "FastReport " << (m_timer_id == CLOCK_REALTIME ? "(real time) " : "(CPU time)  ")    << "     Active Path" << '\n';
      for (auto const & name: tns.getTrigPaths())
        out << "FastReport              "
            << std::right << std::setw(10) << m_paths[name].summary_active  / (double) m_summary_events << "  "
            << name << '\n';
      out << '\n';
      out << "FastReport " << (m_timer_id == CLOCK_REALTIME ? "(real time) " : "(CPU time)  ")    << "     Active EndPath" << '\n';
      for (auto const & name: tns.getEndPaths())
        out << "FastReport              "
            << std::right << std::setw(10) << m_paths[name].summary_active  / (double) m_summary_events << "  "
            << name << '\n';
    } else if (m_enable_timing_paths and m_enable_timing_modules) {
      out << "FastReport " << (m_timer_id == CLOCK_REALTIME ? "(real time) " : "(CPU time)  ")    << "     Active       Pre-     Inter-  Post-mods   Overhead      Total  Path" << '\n';
      for (auto const & name: tns.getTrigPaths()) {
        out << "FastReport              "
            << std::right << std::setw(10) << m_paths[name].summary_active        / (double) m_summary_events << " "
            << std::right << std::setw(10) << m_paths[name].summary_premodules    / (double) m_summary_events << " "
            << std::right << std::setw(10) << m_paths[name].summary_intermodules  / (double) m_summary_events << " "
            << std::right << std::setw(10) << m_paths[name].summary_postmodules   / (double) m_summary_events << " "
            << std::right << std::setw(10) << m_paths[name].summary_overhead      / (double) m_summary_events << " "
            << std::right << std::setw(10) << m_paths[name].summary_total         / (double) m_summary_events << "  "
            << name << '\n';
      }
      out << '\n';
      out << "FastReport " << (m_timer_id == CLOCK_REALTIME ? "(real time) " : "(CPU time)  ")    << "     Active       Pre-     Inter-  Post-mods   Overhead      Total  EndPath" << '\n';
      for (auto const & name: tns.getEndPaths()) {
        out << "FastReport              "
            << std::right << std::setw(10) << m_paths[name].summary_active        / (double) m_summary_events << " "
            << std::right << std::setw(10) << m_paths[name].summary_premodules    / (double) m_summary_events << " "
            << std::right << std::setw(10) << m_paths[name].summary_intermodules  / (double) m_summary_events << " "
            << std::right << std::setw(10) << m_paths[name].summary_postmodules   / (double) m_summary_events << " "
            << std::right << std::setw(10) << m_paths[name].summary_overhead      / (double) m_summary_events << " "
            << std::right << std::setw(10) << m_paths[name].summary_total         / (double) m_summary_events << "  "
            << name << '\n';
      }
    }
    out << '\n';
    if (m_enable_timing_modules) {
      out << "FastReport " << (m_timer_id == CLOCK_REALTIME ? "(real time) " : "(CPU time)  ")    << "     Active  Module" << '\n';
      for (auto & keyval: m_modules) {
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
  // transient dqm configuration
  m_nproc_enabled = false;
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
  m_interpaths = 0.;
  // per-job summary
  m_summary_events = 0;
  m_summary_event = 0.;
  m_summary_source = 0.;
  m_summary_all_paths = 0.;
  m_summary_all_endpaths = 0.;
  m_summary_interpaths = 0.;
  // DQM - the DAQ destroys and re-creates the DQM and DQMStore services at each reconfigure, so we don't need to clean them up
  m_dqms = 0;
  // event summary plots
  m_dqm_event = 0;
  m_dqm_source = 0;
  m_dqm_all_paths = 0;
  m_dqm_all_endpaths = 0;
  m_dqm_interpaths = 0;
  // event summary plots - summed over nodes with the same number of processes
  m_dqm_nproc_event = 0;
  m_dqm_nproc_source = 0;
  m_dqm_nproc_all_paths = 0;
  m_dqm_nproc_all_endpaths = 0;
  m_dqm_nproc_interpaths = 0;
  // plots by path
  m_dqm_paths_active_time = 0;
  m_dqm_paths_total_time = 0;
  m_dqm_paths_exclusive_time = 0;
  m_dqm_paths_interpaths = 0;
  // plots per lumisection
  m_dqm_byls_event = 0;
  m_dqm_byls_source = 0;
  m_dqm_byls_all_paths = 0;
  m_dqm_byls_all_endpaths = 0;
  m_dqm_byls_interpaths = 0;
  // plots per lumisection - summed over nodes with the same number of processes
  m_dqm_nproc_byls_event = 0;
  m_dqm_nproc_byls_source = 0;
  m_dqm_nproc_byls_all_paths = 0;
  m_dqm_nproc_byls_all_endpaths = 0;
  m_dqm_nproc_byls_interpaths = 0;
  // plots vs. instantaneous luminosity
  m_dqm_byluminosity_event = 0;
  m_dqm_byluminosity_source = 0;
  m_dqm_byluminosity_all_paths = 0;
  m_dqm_byluminosity_all_endpaths = 0;
  m_dqm_byluminosity_interpaths = 0;
  // plots vs. instantaneous luminosity - summed over nodes with the same number of processes
  m_dqm_nproc_byluminosity_event = 0;
  m_dqm_nproc_byluminosity_source = 0;
  m_dqm_nproc_byluminosity_all_paths = 0;
  m_dqm_nproc_byluminosity_all_endpaths = 0;
  m_dqm_nproc_byluminosity_interpaths = 0;
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
  m_interpaths   = 0;
  for (PathInfo * path: m_cache_paths) {
    path->time_active       = 0.;
    path->time_premodules   = 0.;
    path->time_intermodules = 0.;
    path->time_postmodules  = 0.;
    path->time_total        = 0.;
  }
  for (ModuleInfo * module: m_cache_modules) {
    module->time_active     = 0.;
    module->has_just_run    = false;
    module->is_exclusive    = false;
  }

  // copy the start event timestamp as the end of the previous path
  // used by the inter-path overhead measurement
  m_timer_path.second = m_timer_event.first;
}

void FastTimerService::postProcessEvent(edm::Event const & event, edm::EventSetup const & setup) {
  //edm::LogImportant("FastTimerService") << __func__ << "(...)";

  if (m_enable_timing_exclusive) {
    for (auto & keyval: m_paths) {
      PathInfo & pathinfo = keyval.second;
      float exclusive = pathinfo.time_overhead;

      for (uint32_t i = 0; i <= pathinfo.last_run; ++i) {
        ModuleInfo * module = pathinfo.modules[i];
        if (module == 0)
          // this is a module occurring more than once in the same path, skip it after the first occurrence
          continue;
        if (module->is_exclusive)
          exclusive += module->time_active;
      }
      m_dqm_paths_exclusive_time->Fill(pathinfo.index, exclusive * 1000.);
      if (m_enable_dqm_bypath_exclusive) {
        pathinfo.dqm_exclusive->Fill(exclusive * 1000.);
      }
    }
  }

  // stop the per-event timer, and account event time
  stop(m_timer_event);
  m_event = delta(m_timer_event);
  m_summary_event += m_event;
  // the last part of inter-path overhead is the time between the end of the last (end)path and the end of the event processing
  double interpaths = delta(m_timer_path.second, m_timer_event.second);
  m_interpaths += interpaths;
  m_summary_interpaths += m_interpaths;
  if (m_dqms) {
    m_dqm_paths_interpaths->Fill(m_paths.size(), interpaths * 1000.);
    
    if (m_enable_dqm_summary) {
      m_dqm_event         ->Fill(m_event          * 1000.);
      m_dqm_source        ->Fill(m_source         * 1000.);
      m_dqm_all_paths     ->Fill(m_all_paths      * 1000.);
      m_dqm_all_endpaths  ->Fill(m_all_endpaths   * 1000.);
      m_dqm_interpaths    ->Fill(m_interpaths     * 1000.);

      if (m_nproc_enabled) {
        m_dqm_nproc_event         ->Fill(m_event          * 1000.);
        m_dqm_nproc_source        ->Fill(m_source         * 1000.);
        m_dqm_nproc_all_paths     ->Fill(m_all_paths      * 1000.);
        m_dqm_nproc_all_endpaths  ->Fill(m_all_endpaths   * 1000.);
        m_dqm_nproc_interpaths    ->Fill(m_interpaths     * 1000.);
      }
    }

    if (m_enable_dqm_byls) {
      unsigned int ls = event.getLuminosityBlock().luminosityBlock();
      m_dqm_byls_event        ->Fill(ls, m_event        * 1000.);
      m_dqm_byls_source       ->Fill(ls, m_source       * 1000.);
      m_dqm_byls_all_paths    ->Fill(ls, m_all_paths    * 1000.);
      m_dqm_byls_all_endpaths ->Fill(ls, m_all_endpaths * 1000.);
      m_dqm_byls_interpaths   ->Fill(ls, m_interpaths   * 1000.);
      
      if (m_nproc_enabled) {
        m_dqm_nproc_byls_event        ->Fill(ls, m_event        * 1000.);
        m_dqm_nproc_byls_source       ->Fill(ls, m_source       * 1000.);
        m_dqm_nproc_byls_all_paths    ->Fill(ls, m_all_paths    * 1000.);
        m_dqm_nproc_byls_all_endpaths ->Fill(ls, m_all_endpaths * 1000.);
        m_dqm_nproc_byls_interpaths   ->Fill(ls, m_interpaths   * 1000.);
      }
    }

    if (m_enable_dqm_byluminosity) {
      float luminosity = 0.;
      edm::Handle<LumiScalersCollection> h_luminosity;
      if (event.getByLabel(m_luminosity_label, h_luminosity) and not h_luminosity->empty())
        luminosity = h_luminosity->front().instantLumi();   // in units of 1e30 cm-2s-1

      m_dqm_byluminosity_event        ->Fill(luminosity, m_event        * 1000.);
      m_dqm_byluminosity_source       ->Fill(luminosity, m_source       * 1000.);
      m_dqm_byluminosity_all_paths    ->Fill(luminosity, m_all_paths    * 1000.);
      m_dqm_byluminosity_all_endpaths ->Fill(luminosity, m_all_endpaths * 1000.);
      m_dqm_byluminosity_interpaths   ->Fill(luminosity, m_interpaths   * 1000.);
      
      if (m_nproc_enabled) {
        m_dqm_nproc_byluminosity_event        ->Fill(luminosity, m_event        * 1000.);
        m_dqm_nproc_byluminosity_source       ->Fill(luminosity, m_source       * 1000.);
        m_dqm_nproc_byluminosity_all_paths    ->Fill(luminosity, m_all_paths    * 1000.);
        m_dqm_nproc_byluminosity_all_endpaths ->Fill(luminosity, m_all_endpaths * 1000.);
        m_dqm_nproc_byluminosity_interpaths   ->Fill(luminosity, m_interpaths   * 1000.);
      }
    }
  }
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
}

void FastTimerService::prePathBeginRun(std::string const & path ) {
  //edm::LogImportant("FastTimerService") << __func__ << "(" << path << ")";

  // cache the pointers to the names of the first and last path and endpath
  edm::service::TriggerNamesService & tns = * edm::Service<edm::service::TriggerNamesService>();
  if (not m_skip_first_path and not tns.getTrigPaths().empty()) {
    if (path == tns.getTrigPaths().front())
      m_first_path = & path;
    if (path == tns.getTrigPaths().back())
      m_last_path = & path;
  }
  else if (m_skip_first_path and tns.getTrigPaths().size() > 1) {
    if (path == tns.getTrigPaths().at(1))
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
      for (ModuleInfo * module: m_current_path->modules)
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

  // measure the inter-path overhead as the time elapsed since the end of preiovus path
  // (or the beginning of the event, if this is the first path - see preProcessEvent)
  double interpaths = delta(m_timer_path.second,  m_timer_path.first);
  m_interpaths += interpaths;
  if (m_dqms) {
    m_dqm_paths_interpaths->Fill(m_current_path->index, interpaths * 1000.);
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

    if (m_dqms) {
      m_dqm_paths_active_time->Fill(pathinfo.index, active * 1000.);
      if (m_enable_dqm_bypath_active) {
        pathinfo.dqm_active->Fill(active * 1000.);
      }
    }

    // measure the time spent between the execution of the last module and the end of the path
    if (m_enable_timing_modules) {
      double pre      = 0.;                 // time spent before the first active module
      double inter    = 0.;                 // time spent between active modules
      double post     = 0.;                 // time spent after the last active module
      double overhead = 0.;                 // time spent before, between, or after modules
      double current  = 0.;                 // time spent in modules active in the current path
      double total    = active;             // total per-path time, including modules already run as part of other paths

      // implementation note:
      // "active"   has already measured all the time spent in this path
      // "current"  will be the sum of the time spent inside each module while running this path, so that
      // "overhead" will be active - current
      // "total"    will be active + the sum of the time spent in non-active modules

      uint32_t last_run = status.index();     // index of the last module run in this path
      for (uint32_t i = 0; i <= last_run; ++i) {
        ModuleInfo * module = pathinfo.modules[i];

        // fill counter histograms - also for duplicate modules, to properly extract rejection information
        if (m_enable_dqm_bypath_counters) {
          pathinfo.dqm_module_counter->Fill(i);
        }
        
        if (module == 0)
          // this is a module occurring more than once in the same path, skip it after the first occurrence
          continue;

        if (module->has_just_run) {
          current += module->time_active;
          module->is_exclusive = true;
        } else {
          total   += module->time_active;
          module->is_exclusive = false;
        }

        // fill detailed timing histograms
        if (m_enable_dqm_bypath_details) {
          // fill the total time for all non-duplicate modules
          pathinfo.dqm_module_total->Fill(i, module->time_active * 1000.);
          if (module->has_just_run) {
            // fill the active time only for module actually running in this path
            pathinfo.dqm_module_active->Fill(i, module->time_active * 1000.);
          }
        }

      }

      if (status.accept())
        if (m_enable_dqm_bypath_counters) {
          pathinfo.dqm_module_counter->Fill(pathinfo.modules.size());
        }

      if (m_is_first_module) {
        // no modules were active during this path, account all the time as overhead
        pre      = 0.;
        inter    = 0.;
        post     = active;
        overhead = active;
      } else {
        // extract overhead information
        pre      = delta(m_timer_path.first,    m_timer_first_module);
        post     = delta(m_timer_module.second, m_timer_path.second);
        inter    = active - pre - current - post;
        // take care of numeric precision and rounding errors - the timer is less precise than nanosecond resolution
        if (std::abs(inter) < 1e-9)
          inter = 0.;
        overhead = active - current;
        // take care of numeric precision and rounding errors - the timer is less precise than nanosecond resolution
        if (std::abs(overhead) < 1e-9)
          overhead = 0.;
      }

      pathinfo.time_premodules       = pre;
      pathinfo.time_intermodules     = inter;
      pathinfo.time_postmodules      = post;
      pathinfo.time_overhead         = overhead;
      pathinfo.time_total            = total;
      pathinfo.summary_premodules   += pre;
      pathinfo.summary_intermodules += inter;
      pathinfo.summary_postmodules  += post;
      pathinfo.summary_overhead     += overhead;
      pathinfo.summary_total        += total;
      pathinfo.last_run              = last_run;
      if (m_dqms) {
        if (m_enable_dqm_bypath_overhead) {
          pathinfo.dqm_premodules  ->Fill(pre      * 1000.);
          pathinfo.dqm_intermodules->Fill(inter    * 1000.);
          pathinfo.dqm_postmodules ->Fill(post     * 1000.);
          pathinfo.dqm_overhead    ->Fill(overhead * 1000.);
        }
        m_dqm_paths_total_time->Fill(pathinfo.index, total * 1000.);
        if (m_enable_dqm_bypath_total) {
          pathinfo.dqm_total       ->Fill(total    * 1000.);
        }
      }
    }
  }

  if (& path == m_last_path) {
    // this is the last path, stop and account the "all paths" counter
    m_timer_paths.second = m_timer_path.second;
    m_all_paths = delta(m_timer_paths);
    m_summary_all_paths += m_all_paths;
  } else if (& path == m_last_endpath) {
    // this is the last endpath, stop and account the "all endpaths" counter
    m_timer_endpaths.second = m_timer_path.second;
    m_all_endpaths = delta(m_timer_endpaths);
    m_summary_all_endpaths += m_all_endpaths;
  }

}

void FastTimerService::preModule(edm::ModuleDescription const & module) {
  //edm::LogImportant("FastTimerService") << __func__ << "(" << & module << ")";

  // this is ever called only if m_enable_timing_modules = true
  assert(m_enable_timing_modules);

  // time each module
  start(m_timer_module);

  if (m_is_first_module) {
    m_is_first_module = false;

    // measure the time spent between the beginning of the path and the execution of the first module
    m_timer_first_module = m_timer_module.first;
  } 
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

    if (m_dqms and m_enable_dqm_bymodule) {
      module.dqm_active->Fill(time * 1000.);
    }
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

  for (auto const & keyval: m_modules) {
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
  for (auto const & module: modules) {
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


// query the current module/path/event
// Note: these functions incur in a "per-call timer overhead" (see above), currently of the order of 340ns

// return the time spent since the last preModule() event
double FastTimerService::currentModuleTime() const {
  struct timespec now;
  gettime(now);
  return delta(m_timer_module.first, now);
}

// return the time spent since the last preProcessPath() event
double FastTimerService::currentPathTime() const {
  struct timespec now;
  gettime(now);
  return delta(m_timer_path.first, now);
}

// return the time spent since the last preProcessEvent() event
double FastTimerService::currentEventTime() const {
  struct timespec now;
  gettime(now);
  return delta(m_timer_event.first, now);
}

// query the time spent in a module (available after the module has run)
double FastTimerService::queryModuleTime(const edm::ModuleDescription & module) const {
  ModuleMap<ModuleInfo>::const_iterator keyval = m_modules.find(& module);
  if (keyval != m_modules.end()) {
    return keyval->second.time_active;
  } else {
    edm::LogError("FastTimerService") << "FastTimerService::postModule: unexpected module " << module.moduleLabel();
    return 0.;
  }
}

// query the time spent in a path (available after the path has run)
double FastTimerService::queryPathActiveTime(const std::string & path) const {
  PathMap<PathInfo>::const_iterator keyval = m_paths.find(path);
  if (keyval != m_paths.end()) {
    return keyval->second.time_active;
  } else {
    edm::LogError("FastTimerService") << "FastTimerService::postModule: unexpected path " << path;
    return 0.;
  }
}

// query the total time spent in a path (available after the path has run)
double FastTimerService::queryPathTotalTime(const std::string & path) const {
  PathMap<PathInfo>::const_iterator keyval = m_paths.find(path);
  if (keyval != m_paths.end()) {
    return keyval->second.time_total;
  } else {
    edm::LogError("FastTimerService") << "FastTimerService::postModule: unexpected path " << path;
    return 0.;
  }
}

// query the time spent in the current event's source (available during event processing)
double FastTimerService::querySourceTime() const {
  return m_source;
}

// query the time spent in the current event's paths (available during endpaths)
double FastTimerService::queryPathsTime() const {
  return m_all_paths;
}

// query the time spent in the current event's endpaths (available after all endpaths have run)
double FastTimerService::queryEndPathsTime() const {
  return m_all_endpaths;
}

// query the time spent processing the current event (available after the event has been processed)
double FastTimerService::queryEventTime() const {
  return m_event;
}

// describe the module's configuration
void FastTimerService::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>(   "useRealTimeClock",         true);
  desc.addUntracked<bool>(   "enableTimingPaths",        true);
  desc.addUntracked<bool>(   "enableTimingModules",      true);
  desc.addUntracked<bool>(   "enableTimingExclusive",    false);
  desc.addUntracked<bool>(   "enableTimingSummary",      false);
  desc.addUntracked<bool>(   "skipFirstPath",            false), 
  desc.addUntracked<bool>(   "enableDQM",                true);
  desc.addUntracked<bool>(   "enableDQMbyPathActive",    false);
  desc.addUntracked<bool>(   "enableDQMbyPathTotal",     true);
  desc.addUntracked<bool>(   "enableDQMbyPathOverhead",  false);
  desc.addUntracked<bool>(   "enableDQMbyPathDetails",   false);
  desc.addUntracked<bool>(   "enableDQMbyPathCounters",  true);
  desc.addUntracked<bool>(   "enableDQMbyPathExclusive", false);
  desc.addUntracked<bool>(   "enableDQMbyModule",        false);
  desc.addUntracked<bool>(   "enableDQMSummary",         false);
  desc.addUntracked<bool>(   "enableDQMbyLuminosity",    false);
  desc.addNode(
    edm::ParameterDescription<bool>(   "enableDQMbyLumiSection", false, false) or
    edm::ParameterDescription<bool>(   "enableDQMbyLumi",        false, false)
  );
  desc.addUntracked<bool>(   "enableDQMbyProcesses",     false);
  desc.addUntracked<double>( "dqmTimeRange",             1000. );   // ms
  desc.addUntracked<double>( "dqmTimeResolution",           5. );   // ms
  desc.addUntracked<double>( "dqmPathTimeRange",          100. );   // ms
  desc.addUntracked<double>( "dqmPathTimeResolution",       0.5);   // ms
  desc.addUntracked<double>( "dqmModuleTimeRange",         40. );   // ms
  desc.addUntracked<double>( "dqmModuleTimeResolution",     0.2);   // ms
  desc.addUntracked<double>( "dqmLuminosityRange",      1.e34  );   // cm-2 s-1
  desc.addUntracked<double>( "dqmLuminosityResolution", 1.e31  );   // cm-2 s-1
  desc.addUntracked<uint32_t>( "dqmLumiSectionsRange",   2500  );   // ~ 16 hours
  desc.addUntracked<std::string>(   "dqmPath",           "HLT/TimerService");
  desc.addUntracked<edm::InputTag>( "luminosityProduct", edm::InputTag("hltScalersRawToDigi"));
  desc.addUntracked<std::vector<unsigned int> >("supportedProcesses", { 8, 24, 32 });
  descriptions.add("FastTimerService", desc);
}

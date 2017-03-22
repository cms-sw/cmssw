// FIXME
// we are by-passing the ME's when filling the plots, so we might need to call the ME's update() by hand


// C++ headers
#include <cmath>
#include <limits>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <string>
#include <sstream>
#include <unordered_set>
#include <unordered_map>

// boost headers
#include <boost/format.hpp>

// tbb headers
#include <tbb/concurrent_vector.h>

// CMSSW headers
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "DataFormats/Common/interface/HLTPathStatus.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "HLTrigger/Timer/interface/FastTimerService.h"


FastTimerService::FastTimerService(const edm::ParameterSet & config, edm::ActivityRegistry & registry) :
  // configuration
  m_callgraph(),
  // FIXME - reimplement support for cpu time vs. real time
  m_use_realtime(                config.getUntrackedParameter<bool>(     "useRealTimeClock"         ) ),
  m_enable_timing_paths(         config.getUntrackedParameter<bool>(     "enableTimingPaths"        ) ),
  m_enable_timing_modules(       config.getUntrackedParameter<bool>(     "enableTimingModules"      ) ),
  m_enable_timing_exclusive(     config.getUntrackedParameter<bool>(     "enableTimingExclusive"    ) ),
  m_enable_timing_summary(       config.getUntrackedParameter<bool>(     "enableTimingSummary"      ) ),
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
  m_enable_dqm_byls(             config.getUntrackedParameter<bool>(     "enableDQMbyLumiSection"   ) ),
  m_enable_dqm_bynproc(          config.getUntrackedParameter<bool>(     "enableDQMbyProcesses"     ) ),
  // job configuration
  m_concurrent_runs(             0 ),
  m_concurrent_streams(          0 ),
  m_concurrent_threads(          0 ),
  module_id_(                    edm::ModuleDescription::invalidID() ),
  m_dqm_eventtime_range(         config.getUntrackedParameter<double>(   "dqmTimeRange"             ) ),            // ms
  m_dqm_eventtime_resolution(    config.getUntrackedParameter<double>(   "dqmTimeResolution"        ) ),            // ms
  m_dqm_pathtime_range(          config.getUntrackedParameter<double>(   "dqmPathTimeRange"         ) ),            // ms
  m_dqm_pathtime_resolution(     config.getUntrackedParameter<double>(   "dqmPathTimeResolution"    ) ),            // ms
  m_dqm_moduletime_range(        config.getUntrackedParameter<double>(   "dqmModuleTimeRange"       ) ),            // ms
  m_dqm_moduletime_resolution(   config.getUntrackedParameter<double>(   "dqmModuleTimeResolution"  ) ),            // ms
  m_dqm_lumisections_range(      config.getUntrackedParameter<uint32_t>( "dqmLumiSectionsRange"     ) ),
  m_dqm_path(                    config.getUntrackedParameter<std::string>("dqmPath" ) )
  /*
  // description of the process(es)
  m_process(),
  // DQM - these are initialized at preStreamBeginRun(), to make sure the DQM service has been loaded
  m_stream(),
  // summary data
  m_run_summary(),
  m_job_summary(),
  m_run_summary_perprocess(),
  m_job_summary_perprocess()
  */
{
  // enable timers if required by DQM plots
  m_enable_timing_paths     = m_enable_timing_paths         or
                              m_enable_dqm_bypath_active    or
                              m_enable_dqm_bypath_total     or
                              m_enable_dqm_bypath_overhead  or
                              m_enable_dqm_bypath_details   or
                              m_enable_dqm_bypath_counters  or
                              m_enable_dqm_bypath_exclusive;

  m_enable_timing_modules   = m_enable_timing_modules       or
                              m_enable_dqm_bymodule         or
                              m_enable_dqm_bypath_total     or
                              m_enable_dqm_bypath_overhead  or
                              m_enable_dqm_bypath_details   or
                              m_enable_dqm_bypath_counters  or
                              m_enable_dqm_bypath_exclusive;

  m_enable_timing_exclusive = m_enable_timing_exclusive     or
                              m_enable_dqm_bypath_exclusive;


  registry.watchPreallocate(                this, & FastTimerService::preallocate );
  registry.watchPreBeginJob(                this, & FastTimerService::preBeginJob );
  registry.watchPostBeginJob(               this, & FastTimerService::postBeginJob );
  registry.watchPostEndJob(                 this, & FastTimerService::postEndJob );
  registry.watchPreGlobalBeginRun(          this, & FastTimerService::preGlobalBeginRun );
  registry.watchPostGlobalBeginRun(         this, & FastTimerService::postGlobalBeginRun );
  registry.watchPreGlobalEndRun(            this, & FastTimerService::preGlobalEndRun );
  registry.watchPostGlobalEndRun(           this, & FastTimerService::postGlobalEndRun );
  registry.watchPreStreamBeginRun(          this, & FastTimerService::preStreamBeginRun );
  registry.watchPostStreamBeginRun(         this, & FastTimerService::postStreamBeginRun );
  registry.watchPreStreamEndRun(            this, & FastTimerService::preStreamEndRun );
  registry.watchPostStreamEndRun(           this, & FastTimerService::postStreamEndRun );
  registry.watchPreGlobalBeginLumi(         this, & FastTimerService::preGlobalBeginLumi );
  registry.watchPostGlobalBeginLumi(        this, & FastTimerService::postGlobalBeginLumi );
  registry.watchPreGlobalEndLumi(           this, & FastTimerService::preGlobalEndLumi );
  registry.watchPostGlobalEndLumi(          this, & FastTimerService::postGlobalEndLumi );
  registry.watchPreStreamBeginLumi(         this, & FastTimerService::preStreamBeginLumi );
  registry.watchPostStreamBeginLumi(        this, & FastTimerService::postStreamBeginLumi );
  registry.watchPreStreamEndLumi(           this, & FastTimerService::preStreamEndLumi );
  registry.watchPostStreamEndLumi(          this, & FastTimerService::postStreamEndLumi );
  registry.watchPreEvent(                   this, & FastTimerService::preEvent );
  registry.watchPostEvent(                  this, & FastTimerService::postEvent );
  registry.watchPrePathEvent(               this, & FastTimerService::prePathEvent );
  registry.watchPostPathEvent(              this, & FastTimerService::postPathEvent );
  registry.watchPreSourceConstruction(      this, & FastTimerService::preSourceConstruction);
  registry.watchPreSourceRun(               this, & FastTimerService::preSourceRun );
  registry.watchPostSourceRun(              this, & FastTimerService::postSourceRun );
  registry.watchPreSourceLumi(              this, & FastTimerService::preSourceLumi );
  registry.watchPostSourceLumi(             this, & FastTimerService::postSourceLumi );
  registry.watchPreSourceEvent(             this, & FastTimerService::preSourceEvent );
  registry.watchPostSourceEvent(            this, & FastTimerService::postSourceEvent );
  registry.watchPreModuleBeginJob(          this, & FastTimerService::preModuleBeginJob );
  registry.watchPreEventReadFromSource(     this, & FastTimerService::preEventReadFromSource );
  registry.watchPostEventReadFromSource(    this, & FastTimerService::postEventReadFromSource );
//registry.watchPreModuleBeginStream(       this, & FastTimerService::preModuleBeginStream );
//registry.watchPostModuleBeginStream(      this, & FastTimerService::postModuleBeginStream );
//registry.watchPreModuleEndStream(         this, & FastTimerService::preModuleEndStream );
//registry.watchPostModuleEndStream(        this, & FastTimerService::postModuleEndStream );
  registry.watchPreModuleGlobalBeginRun(    this, & FastTimerService::preModuleGlobalBeginRun );
  registry.watchPostModuleGlobalBeginRun(   this, & FastTimerService::postModuleGlobalBeginRun );
  registry.watchPreModuleGlobalEndRun(      this, & FastTimerService::preModuleGlobalEndRun );
  registry.watchPostModuleGlobalEndRun(     this, & FastTimerService::postModuleGlobalEndRun );
  registry.watchPreModuleGlobalBeginLumi(   this, & FastTimerService::preModuleGlobalBeginLumi );
  registry.watchPostModuleGlobalBeginLumi(  this, & FastTimerService::postModuleGlobalBeginLumi );
  registry.watchPreModuleGlobalEndLumi(     this, & FastTimerService::preModuleGlobalEndLumi );
  registry.watchPostModuleGlobalEndLumi(    this, & FastTimerService::postModuleGlobalEndLumi );
  registry.watchPreModuleStreamBeginRun(    this, & FastTimerService::preModuleStreamBeginRun );
  registry.watchPostModuleStreamBeginRun(   this, & FastTimerService::postModuleStreamBeginRun );
  registry.watchPreModuleStreamEndRun(      this, & FastTimerService::preModuleStreamEndRun );
  registry.watchPostModuleStreamEndRun(     this, & FastTimerService::postModuleStreamEndRun );
  registry.watchPreModuleStreamBeginLumi(   this, & FastTimerService::preModuleStreamBeginLumi );
  registry.watchPostModuleStreamBeginLumi(  this, & FastTimerService::postModuleStreamBeginLumi );
  registry.watchPreModuleStreamEndLumi(     this, & FastTimerService::preModuleStreamEndLumi );
  registry.watchPostModuleStreamEndLumi(    this, & FastTimerService::postModuleStreamEndLumi );
  registry.watchPreModuleEventPrefetching(  this, & FastTimerService::preModuleEventPrefetching );
  registry.watchPostModuleEventPrefetching( this, & FastTimerService::postModuleEventPrefetching );
  registry.watchPreModuleEvent(             this, & FastTimerService::preModuleEvent );
  registry.watchPostModuleEvent(            this, & FastTimerService::postModuleEvent );
  registry.watchPreModuleEventDelayedGet(   this, & FastTimerService::preModuleEventDelayedGet );
  registry.watchPostModuleEventDelayedGet(  this, & FastTimerService::postModuleEventDelayedGet );
}

FastTimerService::~FastTimerService()
{
}

void FastTimerService::preGlobalBeginRun(edm::GlobalContext const & gc)
{
  /*
  unsigned int pid = m_callgraph.processId(* gc.processContext());
  unsigned int rid = gc.runIndex();

  edm::service::TriggerNamesService & tns = * edm::Service<edm::service::TriggerNamesService>();

  uint32_t size_p = tns.getTrigPaths().size();
  uint32_t size_e = tns.getEndPaths().size();
  // resize the path maps
  for (auto & stream: m_stream)
    if (stream.paths.size() <= pid)
      stream.paths.resize(pid+1);
  for (uint32_t i = 0; i < size_p; ++i) {
    std::string const & label = tns.getTrigPath(i);
    for (auto & stream: m_stream)
      stream.paths[pid][label].index = i;
  }
  for (uint32_t i = 0; i < size_e; ++i) {
    std::string const & label = tns.getEndPath(i);
    for (auto & stream: m_stream)
      stream.paths[pid][label].index = size_p + i;
  }
  for (auto & stream: m_stream) {
    // resize the stream buffers to account the number of subprocesses
    if (stream.timing_perprocess.size() <= pid)
      stream.timing_perprocess.resize(pid+1);
    // resize the stream plots to account the number of subprocesses
    if (stream.dqm_perprocess.size() <= pid)
      stream.dqm_perprocess.resize(pid+1);
    if (stream.dqm_perprocess_byls.size() <= pid)
      stream.dqm_perprocess_byls.resize(pid+1);
    if (stream.dqm_paths.size() <= pid)
      stream.dqm_paths.resize(pid+1);
  }
  for (auto & summary: m_run_summary_perprocess) {
    if (summary.size() <= pid)
      summary.resize(pid+1);
  }
  if (m_job_summary_perprocess.size() <= pid)
    m_job_summary_perprocess.resize(pid+1);

  // reset the run summaries
  if (pid == 0)
    m_run_summary[rid].reset();
  m_run_summary_perprocess[rid][pid].reset();

  // associate to each path all the modules it contains
  for (uint32_t i = 0; i < tns.getTrigPaths().size(); ++i)
    fillPathMap( pid, tns.getTrigPath(i), tns.getTrigPathModules(i) );
  for (uint32_t i = 0; i < tns.getEndPaths().size(); ++i)
    fillPathMap( pid, tns.getEndPath(i), tns.getEndPathModules(i) );

  // cache the names of the process, and of first and last non-empty path and endpath
  if (m_process.size() <= pid)
    m_process.resize(pid+1);
  m_process[pid].name = gc.processContext()->processName();
  std::tie(m_process[pid].first_path, m_process[pid].last_path) = findFirstLast(pid, tns.getTrigPaths());
  std::tie(m_process[pid].first_endpath, m_process[pid].last_endpath) = findFirstLast(pid, tns.getEndPaths());
  */
}

void FastTimerService::postGlobalBeginRun(edm::GlobalContext const&)
{
  unsupportedSignal(__func__);
}

void FastTimerService::preStreamBeginRun(edm::StreamContext const & sc)
{
  /*
  std::string const & process_name = sc.processContext()->processName();
  unsigned int pid = m_callgraph.processId(* sc.processContext());
  unsigned int sid = sc.streamID().value();
  auto & stream = m_stream[sid];


  if (not m_enable_dqm)
    return;

  if (not edm::Service<DQMStore>().isAvailable()) {
    // the DQMStore is not available, disable all DQM plots
    m_enable_dqm = false;
    return;
  }

  edm::service::TriggerNamesService & tns = * edm::Service<edm::service::TriggerNamesService>();
  uint32_t size_p = tns.getTrigPaths().size();
  uint32_t size_e = tns.getEndPaths().size();
  uint32_t size = size_p + size_e;

  int eventbins  = (int) std::ceil(m_dqm_eventtime_range  / m_dqm_eventtime_resolution);
  int pathbins   = (int) std::ceil(m_dqm_pathtime_range   / m_dqm_pathtime_resolution);
  int modulebins = (int) std::ceil(m_dqm_moduletime_range / m_dqm_moduletime_resolution);

  // define a callback that can book the histograms
  auto bookTransactionCallback = [&, this] (DQMStore::IBooker & booker) {

    // event summary plots
    if (m_enable_dqm_summary) {
      // whole event
      if (pid == 0) {
        booker.setCurrentFolder(m_dqm_path);
        stream.dqm.presource     = booker.book1D("presource",    "Pre-Source processing time",    modulebins, 0., m_dqm_moduletime_range)->getTH1F();
        stream.dqm.presource     ->StatOverflows(true);
        stream.dqm.presource     ->SetXTitle("processing time [ms]");
        stream.dqm.source        = booker.book1D("source",       "Source processing time",        modulebins, 0., m_dqm_moduletime_range)->getTH1F();
        stream.dqm.source        ->StatOverflows(true);
        stream.dqm.source        ->SetXTitle("processing time [ms]");
        stream.dqm.preevent      = booker.book1D("preevent",     "Pre-Event processing time",     modulebins, 0., m_dqm_moduletime_range)->getTH1F();
        stream.dqm.preevent      ->StatOverflows(true);
        stream.dqm.preevent      ->SetXTitle("processing time [ms]");
        stream.dqm.event         = booker.book1D("event",        "Event processing time",         eventbins,  0., m_dqm_eventtime_range)->getTH1F();
        stream.dqm.event         ->StatOverflows(true);
        stream.dqm.event         ->SetXTitle("processing time [ms]");
      }

      // per subprocess
      booker.setCurrentFolder(m_dqm_path + "/process " + process_name);
      stream.dqm_perprocess[pid].preevent      = booker.book1D("preevent",     "Pre-Event processing time",     modulebins, 0., m_dqm_moduletime_range)->getTH1F();
      stream.dqm_perprocess[pid].preevent      ->StatOverflows(true);
      stream.dqm_perprocess[pid].preevent      ->SetXTitle("processing time [ms]");
      stream.dqm_perprocess[pid].event         = booker.book1D("event",        "Event processing time",         eventbins,  0., m_dqm_eventtime_range)->getTH1F();
      stream.dqm_perprocess[pid].event         ->StatOverflows(true);
      stream.dqm_perprocess[pid].event         ->SetXTitle("processing time [ms]");
      stream.dqm_perprocess[pid].all_paths     = booker.book1D("all_paths",    "Paths processing time",         eventbins,  0., m_dqm_eventtime_range)->getTH1F();
      stream.dqm_perprocess[pid].all_paths     ->StatOverflows(true);
      stream.dqm_perprocess[pid].all_paths     ->SetXTitle("processing time [ms]");
      stream.dqm_perprocess[pid].all_endpaths  = booker.book1D("all_endpaths", "EndPaths processing time",      pathbins,   0., m_dqm_pathtime_range)->getTH1F();
      stream.dqm_perprocess[pid].all_endpaths  ->StatOverflows(true);
      stream.dqm_perprocess[pid].all_endpaths  ->SetXTitle("processing time [ms]");
      stream.dqm_perprocess[pid].interpaths    = booker.book1D("interpaths",   "Time spent between paths",      pathbins,   0., m_dqm_eventtime_range)->getTH1F();
      stream.dqm_perprocess[pid].interpaths    ->StatOverflows(true);
      stream.dqm_perprocess[pid].interpaths    ->SetXTitle("processing time [ms]");
    }

    // plots by path
    if (m_enable_timing_paths) {
      booker.setCurrentFolder(m_dqm_path + "/process " + process_name);
      stream.dqm_paths[pid].active_time     = booker.bookProfile("paths_active_time",    "Additional time spent in each path", size, -0.5, size-0.5, pathbins, 0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
      stream.dqm_paths[pid].active_time     ->StatOverflows(true);
      stream.dqm_paths[pid].active_time     ->SetYTitle("processing time [ms]");
      stream.dqm_paths[pid].total_time      = booker.bookProfile("paths_total_time",     "Total time spent in each path",      size, -0.5, size-0.5, pathbins, 0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
      stream.dqm_paths[pid].total_time      ->StatOverflows(true);
      stream.dqm_paths[pid].total_time      ->SetYTitle("processing time [ms]");
      stream.dqm_paths[pid].exclusive_time  = booker.bookProfile("paths_exclusive_time", "Exclusive time spent in each path",  size, -0.5, size-0.5, pathbins, 0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
      stream.dqm_paths[pid].exclusive_time  ->StatOverflows(true);
      stream.dqm_paths[pid].exclusive_time  ->SetYTitle("processing time [ms]");

      for (uint32_t i = 0; i < size_p; ++i) {
        std::string const & label = tns.getTrigPath(i);
        stream.dqm_paths[pid].active_time    ->GetXaxis()->SetBinLabel(i + 1, label.c_str());
        stream.dqm_paths[pid].total_time     ->GetXaxis()->SetBinLabel(i + 1, label.c_str());
        stream.dqm_paths[pid].exclusive_time ->GetXaxis()->SetBinLabel(i + 1, label.c_str());
      }
      for (uint32_t i = 0; i < size_e; ++i) {
        std::string const & label = tns.getEndPath(i);
        stream.dqm_paths[pid].active_time    ->GetXaxis()->SetBinLabel(i + size_p + 1, label.c_str());
        stream.dqm_paths[pid].total_time     ->GetXaxis()->SetBinLabel(i + size_p + 1, label.c_str());
        stream.dqm_paths[pid].exclusive_time ->GetXaxis()->SetBinLabel(i + size_p + 1, label.c_str());
      }
    }

    // plots vs. instantaneous luminosity
    if (m_enable_dqm_byls) {
      if (pid == 0) {
        // whole event
        booker.setCurrentFolder(m_dqm_path);
        auto & plots = stream.dqm_byls;
        plots.presource = booker.bookProfile("presource_byls",    "Pre-Source processing time vs. lumisection",   m_dqm_lumisections_range, 0.5, m_dqm_lumisections_range + 0.5, pathbins,  0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
        plots.presource ->StatOverflows(true);
        plots.presource ->SetXTitle("lumisection");
        plots.presource ->SetYTitle("processing time [ms]");
        plots.source    = booker.bookProfile("source_byls",       "Source processing time vs. lumisection",       m_dqm_lumisections_range, 0.5, m_dqm_lumisections_range + 0.5, pathbins,  0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
        plots.source    ->StatOverflows(true);
        plots.source    ->SetXTitle("lumisection");
        plots.source    ->SetYTitle("processing time [ms]");
        plots.preevent  = booker.bookProfile("preevent_byls",     "Pre-Event processing time vs. lumisection",    m_dqm_lumisections_range, 0.5, m_dqm_lumisections_range + 0.5, pathbins,  0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
        plots.preevent  ->StatOverflows(true);
        plots.preevent  ->SetXTitle("lumisection");
        plots.preevent  ->SetYTitle("processing time [ms]");
        plots.event     = booker.bookProfile("event_byls",        "Event processing time vs. lumisection",        m_dqm_lumisections_range, 0.5, m_dqm_lumisections_range + 0.5, eventbins, 0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
        plots.event     ->StatOverflows(true);
        plots.event     ->SetXTitle("lumisection");
        plots.event     ->SetYTitle("processing time [ms]");
      }

      // per subprocess
      booker.setCurrentFolder(m_dqm_path + "/process " + process_name);
      auto & plots = stream.dqm_perprocess_byls[pid];
      plots.preevent     = booker.bookProfile("preevent_byls",     "Pre-Event processing time vs. lumisection",    m_dqm_lumisections_range, 0.5, m_dqm_lumisections_range + 0.5, pathbins,  0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
      plots.preevent     ->StatOverflows(true);
      plots.preevent     ->SetXTitle("lumisection");
      plots.preevent     ->SetYTitle("processing time [ms]");
      plots.event        = booker.bookProfile("event_byls",        "Event processing time vs. lumisection",        m_dqm_lumisections_range, 0.5, m_dqm_lumisections_range + 0.5, eventbins, 0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
      plots.event        ->StatOverflows(true);
      plots.event        ->SetXTitle("lumisection");
      plots.event        ->SetYTitle("processing time [ms]");
      plots.all_paths    = booker.bookProfile("all_paths_byls",    "Paths processing time vs. lumisection",        m_dqm_lumisections_range, 0.5, m_dqm_lumisections_range + 0.5, eventbins, 0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
      plots.all_paths    ->StatOverflows(true);
      plots.all_paths    ->SetXTitle("lumisection");
      plots.all_paths    ->SetYTitle("processing time [ms]");
      plots.all_endpaths = booker.bookProfile("all_endpaths_byls", "EndPaths processing time vs. lumisection",     m_dqm_lumisections_range, 0.5, m_dqm_lumisections_range + 0.5, pathbins,  0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
      plots.all_endpaths ->StatOverflows(true);
      plots.all_endpaths ->SetXTitle("lumisection");
      plots.all_endpaths ->SetYTitle("processing time [ms]");
      plots.interpaths   = booker.bookProfile("interpaths_byls",   "Time spent between paths vs. lumisection",     m_dqm_lumisections_range, 0.5, m_dqm_lumisections_range + 0.5, pathbins,  0., std::numeric_limits<double>::infinity(), " ")->getTProfile();
      plots.interpaths   ->StatOverflows(true);
      plots.interpaths   ->SetXTitle("lumisection");
      plots.interpaths   ->SetYTitle("processing time [ms]");
    }

    // per-path and per-module accounting
    if (m_enable_timing_paths) {
      booker.setCurrentFolder(m_dqm_path + "/process " + process_name + "/Paths");
      for (auto & keyval: stream.paths[pid]) {
        std::string const & pathname = keyval.first;
        PathInfo          & pathinfo = keyval.second;

        if (m_enable_dqm_bypath_active) {
          pathinfo.dqm_active       = booker.book1D(pathname + "_active",       pathname + " active time",            pathbins, 0., m_dqm_pathtime_range)->getTH1F();
          pathinfo.dqm_active       ->StatOverflows(true);
          pathinfo.dqm_active       ->SetXTitle("processing time [ms]");
        }

        if (m_enable_dqm_bypath_total) {
          pathinfo.dqm_total        = booker.book1D(pathname + "_total",        pathname + " total time",             pathbins, 0., m_dqm_pathtime_range)->getTH1F();
          pathinfo.dqm_total        ->StatOverflows(true);
          pathinfo.dqm_total        ->SetXTitle("processing time [ms]");
        }

        if (m_enable_dqm_bypath_overhead) {
          pathinfo.dqm_overhead     = booker.book1D(pathname + "_overhead",     pathname + " overhead time",          modulebins, 0., m_dqm_moduletime_range)->getTH1F();
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

          // use a mutex to prevent two threads from assigning to the same element at the same time
          static std::mutex                          dup_mutex;
          // use a tbb::concurrent_vector because growing does not invalidate existing iterators and pointers
          static tbb::concurrent_vector<std::string> dup;
          // lock, and fill the first 32 elements
          if (dup.empty()) {
            std::lock_guard<std::mutex> lock(dup_mutex);
            if (dup.empty()) {
              dup.resize(32);
              for (unsigned int i = 0; i < 32; ++i)
                dup[i] = (boost::format("(dup.) (%d)") % i).str();
            }
          }
          // lock, and fill as many elements as needed
          if (modules.size() > dup.size()) {
            std::lock_guard<std::mutex> lock(dup_mutex);
            unsigned int old_size = dup.size();
            unsigned int new_size = modules.size();
            if (new_size > old_size) {
              dup.resize(new_size);
              for (unsigned int i = old_size; i < new_size; ++i)
                dup[i] = (boost::format("(dup.) (%d)") % i).str();
            }
          }

          std::vector<const char *> labels(modules.size(), nullptr);
          for (uint32_t i = 0; i < modules.size(); ++i)
            labels[i] = (pathinfo.modules[i]) ? modules[i].c_str() : dup[i].c_str();

          // book counter histograms
          if (m_enable_dqm_bypath_counters) {
            pathinfo.dqm_module_counter = booker.book1D(pathname + "_module_counter", pathname + " module counter", modules.size() + 1, -0.5, modules.size() + 0.5)->getTH1F();
            // find module labels
            for (uint32_t i = 0; i < modules.size(); ++i)
              pathinfo.dqm_module_counter->GetXaxis()->SetBinLabel( i+1, labels[i] );
            pathinfo.dqm_module_counter->GetXaxis()->SetBinLabel( modules.size() + 1, "" );
          }
          // book detailed timing histograms
          if (m_enable_dqm_bypath_details) {
            pathinfo.dqm_module_active  = booker.book1D(pathname + "_module_active",  pathname + " module active",  modules.size(), -0.5, modules.size() - 0.5)->getTH1F();
            pathinfo.dqm_module_active  ->SetYTitle("cumulative processing time [ms]");
            pathinfo.dqm_module_total   = booker.book1D(pathname + "_module_total",   pathname + " module total",   modules.size(), -0.5, modules.size() - 0.5)->getTH1F();
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
          pathinfo.dqm_exclusive = booker.book1D(pathname + "_exclusive", pathname + " exclusive time", pathbins, 0., m_dqm_pathtime_range)->getTH1F();
          pathinfo.dqm_exclusive ->StatOverflows(true);
          pathinfo.dqm_exclusive ->SetXTitle("processing time [ms]");
        }

      }
    }

    if (m_enable_dqm_bymodule) {
      booker.setCurrentFolder(m_dqm_path + "/Modules");
      for (auto & keyval: stream.modules) {
        std::string const & label  = keyval.first;
        ModuleInfo        & module = keyval.second;
        module.dqm_active = booker.book1D(label, label, modulebins, 0., m_dqm_moduletime_range)->getTH1F();
        module.dqm_active->StatOverflows(true);
        module.dqm_active->SetXTitle("processing time [ms]");
      }
    }

  };

  // book MonitorElement's for this stream
  edm::Service<DQMStore>()->bookTransaction(bookTransactionCallback, sc.eventID().run(), sid, module_id_);
  */
}


void
FastTimerService::unsupportedSignal(std::string signal) const
{
  /*
  // warn about each signal only once per job
  if (m_unsupported_signals.insert(signal).second)
    edm::LogWarning("FastTimerService") << "The FastTimerService received the unsupported signal \"" << signal << "\".\n"
      << "Please report how to reproduce the issue to cms-hlt@cern.ch .";
  */
}

void
FastTimerService::preallocate(edm::service::SystemBounds const & bounds)
{
  m_concurrent_runs    = bounds.maxNumberOfConcurrentRuns();
  m_concurrent_streams = bounds.maxNumberOfStreams();
  m_concurrent_threads = bounds.maxNumberOfThreads();

  if (m_enable_dqm_bynproc)
    m_dqm_path += (boost::format("/Running %d processes") % m_concurrent_threads).str();

  /*
  m_run_summary.resize(m_concurrent_runs);
  m_run_summary_perprocess.resize(m_concurrent_runs);
  m_stream.resize(m_concurrent_streams);
  */

  // assign a pseudo module id to the FastTimerService
  module_id_ = edm::ModuleDescription::getUniqueID();
  /*
  for (auto & stream: m_stream) {
    stream.fast_modules.resize(module_id_, nullptr);
  }
  */
}

void
FastTimerService::preSourceConstruction(edm::ModuleDescription const & module) {
  m_callgraph.preSourceConstruction(module);
}

void
FastTimerService::preBeginJob(edm::PathsAndConsumesOfModulesBase const & pathsAndConsumes, edm::ProcessContext const & context) {
  m_callgraph.preBeginJob(pathsAndConsumes, context);
}

void
FastTimerService::postBeginJob() {
  unsigned int modules   = m_callgraph.size();
  unsigned int processes = m_callgraph.processes().size();

  // allocate the resource counters for each stream, process, path and module
  streams_.resize(m_concurrent_streams);
  for (auto & stream: streams_) {
    stream.modules.resize(modules);
    stream.processes.resize(processes);
    for (unsigned int i = 0; i < processes; ++i) {
      auto const & process = m_callgraph.processDescription(i);
      stream.processes[i] = {
        Resources(),
        std::vector<ResourcesPerPath>(process.paths_.size()),
        std::vector<ResourcesPerPath>(process.endPaths_.size())
      };
    }
  }

  // allocate the resource measurements per thread
  threads_.resize(m_concurrent_threads);
}

void
FastTimerService::postStreamBeginRun(edm::StreamContext const & sc) {
  /*
  unsigned int sid = sc.streamID().value();
  auto & stream = m_stream[sid];
  stream.timer_last_transition = FastTimer::Clock::now();
  */
}

void
FastTimerService::preStreamEndRun(edm::StreamContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::postStreamEndRun(edm::StreamContext const & sc)
{
  /*
  unsigned int sid = sc.streamID().value();
  auto & stream = m_stream[sid];

  if (m_enable_dqm) {
    DQMStore * store = edm::Service<DQMStore>().operator->();
    assert(store);
    store->mergeAndResetMEsRunSummaryCache(sc.eventID().run(), sid, module_id_);
  }

  stream.reset();
  stream.timer_last_transition = FastTimer::Clock::now();
  */
}

void
FastTimerService::preGlobalBeginLumi(edm::GlobalContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::postGlobalBeginLumi(edm::GlobalContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::preGlobalEndLumi(edm::GlobalContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::postGlobalEndLumi(edm::GlobalContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::preStreamBeginLumi(edm::StreamContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::postStreamBeginLumi(edm::StreamContext const & sc) {
  /*
  unsigned int sid = sc.streamID().value();
  auto & stream = m_stream[sid];
  stream.timer_last_transition = FastTimer::Clock::now();
  */
}

void
FastTimerService::preStreamEndLumi(edm::StreamContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::postStreamEndLumi(edm::StreamContext const & sc) {
  /*
  unsigned int sid = sc.streamID().value();
  auto & stream = m_stream[sid];

  if (m_enable_dqm) {
    DQMStore * store = edm::Service<DQMStore>().operator->();
    assert(store);
    store->mergeAndResetMEsLuminositySummaryCache(sc.eventID().run(),sc.eventID().luminosityBlock(),sid, module_id_);
  }

  stream.timer_last_transition = FastTimer::Clock::now();
  */
}

void
FastTimerService::preGlobalEndRun(edm::GlobalContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::postGlobalEndRun(edm::GlobalContext const & gc)
{
  /*
  if (m_enable_timing_summary) {
    unsigned int pid = m_callgraph.processId(* gc.processContext());
    unsigned int rid = gc.runIndex();
    unsigned int run = gc.luminosityBlockID().run();
    const std::string label = (boost::format("run %d") % run).str();

    printProcessSummary(m_run_summary[rid], m_run_summary_perprocess[rid][pid], label, m_process[pid].name);

    if (pid+1 == m_process.size())
      printSummary(m_run_summary[rid], label);
  }
  */
}

void
FastTimerService::preSourceRun()
{
  unsupportedSignal(__func__);
}

void
FastTimerService::postSourceRun()
{
  unsupportedSignal(__func__);
}

void
FastTimerService::preSourceLumi()
{
  unsupportedSignal(__func__);
}

void
FastTimerService::postSourceLumi()
{
  unsupportedSignal(__func__);
}

void
FastTimerService::postEndJob()
{
  /*
  if (m_enable_timing_summary) {
    const std::string label = "the whole job";
    for (unsigned int pid = 0; pid < m_process.size(); ++pid)
      printProcessSummary(m_job_summary, m_job_summary_perprocess[pid], label, m_process[pid].name);

    printSummary(m_job_summary, label);
  }
  */
}

/*
void
FastTimerService::printProcessSummary(Timing const & total, TimingPerProcess const & summary, std::string const & label, std::string const & process) const
{
  // print a timing summary for the run or job, for each subprocess
  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  out << "FastReport for " << label << ", process " << process << '\n';
  //out << "FastReport " << (m_use_realtime ? "(real time) " : "(CPU time)  ") << '\n';
  out << "FastReport              " << std::right << std::setw(10) << summary.preevent     / (double) total.count   << "  Pre-Event"     << '\n';
  out << "FastReport              " << std::right << std::setw(10) << summary.event        / (double) total.count   << "  Event"         << '\n';
  out << "FastReport              " << std::right << std::setw(10) << summary.all_paths    / (double) total.count   << "  all Paths"     << '\n';
  out << "FastReport              " << std::right << std::setw(10) << summary.all_endpaths / (double) total.count   << "  all EndPaths"  << '\n';
  out << "FastReport              " << std::right << std::setw(10) << summary.interpaths   / (double) total.count   << "  between paths" << '\n';
  edm::LogVerbatim("FastReport") << out.str();
}

void
FastTimerService::printSummary(Timing const & summary, std::string const & label) const
{
  // print a timing summary for the run or job
  //edm::service::TriggerNamesService & tns = * edm::Service<edm::service::TriggerNamesService>();

  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  out << "FastReport for " << label << ", over all subprocesses" << '\n';
  //out << "FastReport " << (m_use_realtime ? "(real time) " : "(CPU time)  ") << '\n';
  out << "FastReport              " << std::right << std::setw(10) << summary.presource    / (double) summary.count << "  Pre-Source"    << '\n';
  out << "FastReport              " << std::right << std::setw(10) << summary.source       / (double) summary.count << "  Source"        << '\n';
  out << "FastReport              " << std::right << std::setw(10) << summary.preevent     / (double) summary.count << "  Pre-Event"     << '\n';
  out << "FastReport              " << std::right << std::setw(10) << summary.event        / (double) summary.count << "  Event"         << '\n';
  edm::LogVerbatim("FastReport") << out.str();
}
*/

/*
  if (m_enable_timing_modules) {
    double modules_total = 0.;
    for (auto & keyval: m_stream.modules)
      modules_total += keyval.second.summary_active;
    out << "FastReport              " << std::right << std::setw(10) << modules_total / (double) summary.count << "  all Modules"   << '\n';
  }
  out << '\n';
  if (m_enable_timing_paths and not m_enable_timing_modules) {
    //out << "FastReport " << (m_use_realtime ? "(real time) " : "(CPU time)  ")    << "     Active Path" << '\n';
    for (auto const & name: tns.getTrigPaths())
      out << "FastReport              "
          << std::right << std::setw(10) << m_stream.paths[pid][name].summary_active / (double) summary.count << "  "
          << name << '\n';
    out << '\n';
    //out << "FastReport " << (m_use_realtime ? "(real time) " : "(CPU time)  ")    << "     Active EndPath" << '\n';
    for (auto const & name: tns.getEndPaths())
      out << "FastReport              "
          << std::right << std::setw(10) << m_stream.paths[pid][name].summary_active / (double) summary.count << "  "
          << name << '\n';
  } else if (m_enable_timing_paths and m_enable_timing_modules) {
    //out << "FastReport " << (m_use_realtime ? "(real time) " : "(CPU time)  ")    << "     Active       Pre-     Inter-  Post-mods   Overhead      Total  Path" << '\n';
    for (auto const & name: tns.getTrigPaths()) {
      out << "FastReport              "
          << std::right << std::setw(10) << m_stream.paths[pid][name].summary_active        / (double) summary.count << " "
          << std::right << std::setw(10) << m_stream.paths[pid][name].summary_overhead      / (double) summary.count << " "
          << std::right << std::setw(10) << m_stream.paths[pid][name].summary_total         / (double) summary.count << "  "
          << name << '\n';
    }
    out << '\n';
    //out << "FastReport " << (m_use_realtime ? "(real time) " : "(CPU time)  ")    << "     Active       Pre-     Inter-  Post-mods   Overhead      Total  EndPath" << '\n';
    for (auto const & name: tns.getEndPaths()) {
      out << "FastReport              "
          << std::right << std::setw(10) << m_stream.paths[pid][name].summary_active        / (double) summary.count << " "
          << std::right << std::setw(10) << m_stream.paths[pid][name].summary_overhead      / (double) summary.count << " "
          << std::right << std::setw(10) << m_stream.paths[pid][name].summary_total         / (double) summary.count << "  "
          << name << '\n';
    }
  }
  out << '\n';
  if (m_enable_timing_modules) {
    //out << "FastReport " << (m_use_realtime ? "(real time) " : "(CPU time)  ")    << "     Active  Module" << '\n';
    for (auto & keyval: m_stream.modules) {
      std::string const & label  = keyval.first;
      ModuleInfo  const & module = keyval.second;
      out << "FastReport              " << std::right << std::setw(10) << module.summary_active  / (double) summary.count << "  " << label << '\n';
    }
    //out << "FastReport " << (m_use_realtime ? "(real time) " : "(CPU time)  ")    << "     Active  Module" << '\n';
    out << '\n';
    //out << "FastReport " << (m_use_realtime ? "(real time) " : "(CPU time)  ")    << "     Active  Module" << '\n';
  }
*/

void FastTimerService::preModuleBeginJob(edm::ModuleDescription const & module) {
  /*
  // allocate a counter for each module and module type
  for (auto & stream: m_stream) {
    if (module.id() >= stream.fast_modules.size())
      stream.fast_modules.resize(module.id() + 1, nullptr);
    stream.fast_modules[module.id()] = & stream.modules[module.moduleLabel()];;
  }
  */
}

void FastTimerService::preEvent(edm::StreamContext const & sc) {
  /*
  unsigned int pid = m_callgraph.processId(* sc.processContext());
  unsigned int sid = sc.streamID().value();
  auto & stream = m_stream[sid];

  // new event, reset the per-event counter
  stream.timer_event.start();

  // account the time spent between the last transition and the beginning of the event
  stream.timing_perprocess[pid].preevent = delta(stream.timer_last_transition, stream.timer_event.getStartTime());

  // clear the event counters
  stream.timing_perprocess[pid].event        = 0;
  stream.timing_perprocess[pid].all_paths    = 0;
  stream.timing_perprocess[pid].all_endpaths = 0;
  stream.timing_perprocess[pid].interpaths   = 0;
  for (auto & keyval : stream.paths[pid]) {
    keyval.second.timer.reset();
    keyval.second.time_active       = 0.;
    keyval.second.time_exclusive    = 0.;
    keyval.second.time_total        = 0.;
  }

  // copy the start event timestamp as the end of the previous path
  // used by the inter-path overhead measurement
  stream.timer_last_path = stream.timer_event.getStartTime();
  */
}

template <typename T>
double ms(T duration) {
  return boost::chrono::duration_cast<boost::chrono::duration<double, boost::milli>>(duration).count();;
}

void FastTimerService::postEvent(edm::StreamContext const & sc)
{
  unsigned int pid = m_callgraph.processId(* sc.processContext());
  unsigned int sid = sc.streamID();
  auto & stream  = streams_[sid];
  auto & process = m_callgraph.processDescription(pid);

  // compute the event timing as the sum of all modules' timing
  auto & data = stream.processes[pid].total;
  data.reset();
  for (unsigned int i: process.modules_)
    data += stream.modules[i];
  stream.total += data;

  // write the summary only after the last subprocess has run
  if (pid != m_callgraph.processes().size() - 1)
    return;

  std::ostringstream out;
  out << "Modules:\n";
  auto const & source_d = m_callgraph.source();
  auto const & source   = stream.modules[source_d.id()];
  out << boost::format("  %10.3f ms    %10.3f ms    source %s\n") % ms(source.time_thread) % ms(source.time_real) % source_d.moduleLabel();
  for (unsigned int i = 0; i < m_callgraph.processes().size(); ++i) {
    auto const & proc_d = m_callgraph.processDescription(i);
    auto const & proc   = stream.processes[i];
    out << boost::format("  %10.3f ms    %10.3f ms    process %s\n") % ms(proc.total.time_thread) % ms(proc.total.time_real) % proc_d.name_;
    for (unsigned int m: proc_d.modules_) {
      auto const & module_d = m_callgraph.module(m);
      auto const & module   = stream.modules[m];
      out << boost::format("  %10.3f ms    %10.3f ms      %s\n") % ms(module.time_thread) % ms(module.time_real) % module_d.moduleLabel();
    }
  }
  out << boost::format("  %10.3f ms    %10.3f ms    total\n") % ms(stream.total.time_thread) % ms(stream.total.time_real);
  out << std::endl;

  out << "Process:\n";
  out << boost::format("  %10.3f ms    %10.3f ms    source %s\n") % ms(source.time_thread) % ms(source.time_real) % source_d.moduleLabel();
  for (unsigned int i = 0; i < m_callgraph.processes().size(); ++i) {
    auto const & proc_d = m_callgraph.processDescription(i);
    auto const & proc   = stream.processes[i];
    out << boost::format("  %10.3f ms    %10.3f ms    process %s\n") % ms(proc.total.time_thread) % ms(proc.total.time_real) % proc_d.name_;
    for (unsigned int p = 0; p < proc.paths.size(); ++p) {
      auto const & name = proc_d.paths_[p].name_;
      auto const & path = proc.paths[p];
      out << boost::format("  %10.3f ms    %10.3f ms      %s (active)\n") % ms(path.active.time_thread) % ms(path.active.time_real) % name;
      out << boost::format("  %10.3f ms    %10.3f ms      %s (total)\n")  % ms(path.total.time_thread)  % ms(path.total.time_real)  % name;
    }
    for (unsigned int p = 0; p < proc.endpaths.size(); ++p) {
      auto const & name = proc_d.endPaths_[p].name_;
      auto const & path = proc.endpaths[p];
      out << boost::format("  %10.3f ms    %10.3f ms      %s (active)\n") % ms(path.active.time_thread) % ms(path.active.time_real) % name;
      out << boost::format("  %10.3f ms    %10.3f ms      %s (total)\n")  % ms(path.total.time_thread)  % ms(path.total.time_real)  % name;
    }
  }
  out << boost::format("  %10.3f ms    %10.3f ms    total\n") % ms(stream.total.time_thread) % ms(stream.total.time_real);
  edm::LogVerbatim("FastReport") << out.str();

  /*
  unsigned int pid = m_callgraph.processId(* sc.processContext());
  unsigned int sid = sc.streamID();
  unsigned int rid = sc.runIndex();
  auto & stream = m_stream[sid];

  // stop the per-event timer, and account event time
  stream.timer_event.stop();
  stream.timer_last_transition = stream.timer_event.getStopTime();
  stream.timing_perprocess[pid].event = stream.timer_event.seconds();

  // the last part of inter-path overhead is the time between the end of the last (end)path and the end of the event processing
  double interpaths = delta(stream.timer_last_path, stream.timer_event.getStopTime());
  stream.timing_perprocess[pid].interpaths += interpaths;

  {
    // prevent different threads from updating the summary information at the same time
    std::lock_guard<std::mutex> lock_summary(m_summary_mutex);

    // keep track of the total number of events and add this event's time to the per-run and per-job summary
    m_run_summary_perprocess[rid][pid] += stream.timing_perprocess[pid];
    m_job_summary_perprocess[pid]      += stream.timing_perprocess[pid];

    // account the whole event timing details
    if (pid+1 == m_process.size()) {
      stream.timing.count    = 1;
      stream.timing.preevent = stream.timing_perprocess[0].preevent;
      stream.timing.event    = stream.timing_perprocess[0].event;
      for (unsigned int i = 1; i < m_process.size(); ++i) {
        stream.timing.event += stream.timing_perprocess[i].preevent;
        stream.timing.event += stream.timing_perprocess[i].event;
      }
      m_run_summary[rid] += stream.timing;
      m_job_summary      += stream.timing;
    }
  }

  // elaborate "exclusive" modules
  if (m_enable_timing_exclusive) {
    for (auto & keyval: stream.paths[pid]) {
      PathInfo & pathinfo = keyval.second;
      pathinfo.time_exclusive = pathinfo.time_overhead;

      for (uint32_t i = 0; i < pathinfo.last_run; ++i) {
        ModuleInfo * module = pathinfo.modules[i];
        if (module == 0)
          // this is a module occurring more than once in the same path, skip it after the first occurrence
          continue;
        if ((module->run_in_path == & pathinfo) and (module->counter == 1))
          pathinfo.time_exclusive += module->time_active;
      }
    }
  }

  // fill the DQM plots from the internal buffers
  if (not m_enable_dqm)
    return;

  // fill plots for per-event time by path
  if (m_enable_timing_paths) {

    for (auto & keyval: stream.paths[pid]) {
      PathInfo & pathinfo = keyval.second;

      stream.dqm_paths[pid].active_time->Fill(pathinfo.index, pathinfo.time_active * 1000.);
      if (m_enable_dqm_bypath_active)
        pathinfo.dqm_active->Fill(pathinfo.time_active * 1000.);

      stream.dqm_paths[pid].exclusive_time->Fill(pathinfo.index, pathinfo.time_exclusive * 1000.);
      if (m_enable_dqm_bypath_exclusive)
        pathinfo.dqm_exclusive->Fill(pathinfo.time_exclusive * 1000.);

      stream.dqm_paths[pid].total_time->Fill(pathinfo.index, pathinfo.time_total * 1000.);
      if (m_enable_dqm_bypath_total)
        pathinfo.dqm_total->Fill(pathinfo.time_total * 1000.);

      // fill path overhead histograms
      if (m_enable_dqm_bypath_overhead) {
        pathinfo.dqm_overhead->Fill(pathinfo.time_overhead * 1000.);
      }

      // fill detailed timing histograms
      if (m_enable_dqm_bypath_details) {
        for (uint32_t i = 0; i < pathinfo.last_run; ++i) {
          ModuleInfo * module = pathinfo.modules[i];
          // skip duplicate modules
          if (module == nullptr)
            continue;
          // fill the total time for all non-duplicate modules
          pathinfo.dqm_module_total->Fill(i, module->time_active * 1000.);
          // fill the active time only for module that have actually run in this path
          if (module->run_in_path == & pathinfo)
            pathinfo.dqm_module_active->Fill(i, module->time_active * 1000.);
        }
      }

      // fill path counter histograms
      //   - also for duplicate modules, to properly extract rejection information
      //   - fill the N+1th bin for paths accepting the event, so the FastTimerServiceClient can properly measure the last filter efficiency
      if (m_enable_dqm_bypath_counters) {
        for (uint32_t i = 0; i < pathinfo.last_run; ++i)
          pathinfo.dqm_module_counter->Fill(i);
        if (pathinfo.accept)
          pathinfo.dqm_module_counter->Fill(pathinfo.modules.size());
      }

    }
  }

  // fill plots for per-event time by module
  // note: this is done only for the last subprocess, to avoid filling the same plots multiple times
  if ((m_enable_dqm_bymodule) and (pid+1 == m_process.size())) {
    for (auto & keyval : stream.modules) {
      ModuleInfo & module = keyval.second;
      module.dqm_active->Fill(module.time_active * 1000.);
    }
  }

  if (m_enable_dqm_summary) {
    if (pid+1 == m_process.size())
      stream.dqm.fill(stream.timing);
    stream.dqm_perprocess[pid].fill(stream.timing_perprocess[pid]);
  }

  if (m_enable_dqm_byls) {
    if (pid+1 == m_process.size())
      stream.dqm_byls.fill(sc.eventID().luminosityBlock(), stream.timing);
    stream.dqm_perprocess_byls[pid].fill(sc.eventID().luminosityBlock(), stream.timing_perprocess[pid]);
  }
  */
}

void FastTimerService::preSourceEvent(edm::StreamID sid)
{
  // clear the event counters
  auto & stream = streams_[sid];
  stream.reset();

  thread().measure();
}


void FastTimerService::postSourceEvent(edm::StreamID sid) {
  edm::ModuleDescription const & md = m_callgraph.source();
  unsigned int id  = md.id();
  auto & stream = streams_[sid];

  thread().measure(stream.modules[id]);
}


void FastTimerService::prePathEvent(edm::StreamContext const & sc, edm::PathContext const & pc) {
}


void FastTimerService::postPathEvent(edm::StreamContext const & sc, edm::PathContext const & pc, edm::HLTPathStatus const & status) {
  unsigned int pid = m_callgraph.processId(* sc.processContext());
  unsigned int id  = pc.pathID();
  auto const & path = pc.isEndPath() ? m_callgraph.processDescription(pid).endPaths_[id] : m_callgraph.processDescription(pid).paths_[id];

  // is the Path does not contain any modules, there is nothing to do
  if (path.modules_on_path_.empty())
    return;

  unsigned int sid = sc.streamID().value();
  auto & stream = streams_[sid];
  auto & data = pc.isEndPath() ? stream.processes[pid].endpaths[id] : stream.processes[pid].paths[id];

  unsigned int index = status.index();
  for (unsigned int i = 0; i < index; ++i) {
    auto const & module = stream.modules[path.modules_on_path_[i]];
    data.active += module;
  }
  unsigned int lastDep = path.last_dependency_of_module_[index];
  for (unsigned int i = 0; i < lastDep; ++i) {
    auto const & module = stream.modules[path.modules_and_dependencies_[i]];
    data.total += module;
  }

  /*
  std::string const & path = pc.pathName();
  unsigned int pid = m_callgraph.processId(* sc.processContext());
  unsigned int sid = sc.streamID().value();
  auto & stream = m_stream[sid];

  if (stream.current_path == nullptr) {
    edm::LogError("FastTimerService") << "FastTimerService::postPathEvent: unexpected path " << path;
    return;
  }

  // time each (end)path
  stream.current_path->timer.stop();
  stream.current_path->time_active = stream.current_path->timer.seconds();
  stream.timer_last_path = stream.current_path->timer.getStopTime();

  double active = stream.current_path->time_active;

  // if enabled, account each (end)path
  if (m_enable_timing_paths) {

    PathInfo & pathinfo = * stream.current_path;
    pathinfo.summary_active += active;

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

      uint32_t last_run = 0;                // index of the last module run in this path, plus one
      if (status.wasrun() and not pathinfo.modules.empty())
        last_run = status.index() + 1;      // index of the last module run in this path, plus one
      for (uint32_t i = 0; i < last_run; ++i) {
        ModuleInfo * module = pathinfo.modules[i];

        if (module == 0)
          // this is a module occurring more than once in the same path, skip it after the first occurrence
          continue;

        ++module->counter;
        if (module->run_in_path == & pathinfo) {
          current += module->time_active;
        } else {
          total   += module->time_active;
        }

      }

      if (stream.current_path->first_module == nullptr) {
        // no modules were active during this path, account all the time as overhead
        pre      = 0.;
        inter    = 0.;
        post     = active;
        overhead = active;
      } else {
        // extract overhead information
        pre      = delta(stream.current_path->timer.getStartTime(),  stream.current_path->first_module->timer.getStartTime());
        post     = delta(stream.current_module->timer.getStopTime(), stream.current_path->timer.getStopTime());
        inter    = active - pre - current - post;
        // take care of numeric precision and rounding errors - the timer is less precise than nanosecond resolution
        if (std::abs(inter) < 1e-9)
          inter = 0.;
        overhead = active - current;
        // take care of numeric precision and rounding errors - the timer is less precise than nanosecond resolution
        if (std::abs(overhead) < 1e-9)
          overhead = 0.;
      }

      pathinfo.time_overhead         = overhead;
      pathinfo.time_total            = total;
      pathinfo.summary_overhead     += overhead;
      pathinfo.summary_total        += total;
      pathinfo.last_run              = last_run;
      pathinfo.accept                = status.accept();
    }
  }

  if (path == m_process[pid].last_path) {
    // this is the last path, stop and account the "all paths" counter
    stream.timer_paths.setStopTime(stream.current_path->timer.getStopTime());
    stream.timing_perprocess[pid].all_paths = stream.timer_paths.seconds();
  } else if (path == m_process[pid].last_endpath) {
    // this is the last endpath, stop and account the "all endpaths" counter
    stream.timer_endpaths.setStopTime(stream.current_path->timer.getStopTime());
    stream.timing_perprocess[pid].all_endpaths = stream.timer_endpaths.seconds();
  }
  */
}

void FastTimerService::preModuleEvent(edm::StreamContext const & sc, edm::ModuleCallingContext const & mcc) {
  thread().measure();

  /*
  // this is ever called only if m_enable_timing_modules = true
  assert(m_enable_timing_modules);

  if (mcc.moduleDescription() == nullptr) {
    edm::LogError("FastTimerService") << "FastTimerService::preModuleEvent: invalid module";
    return;
  }

  edm::ModuleDescription const & md = * mcc.moduleDescription();
  unsigned int sid = sc.streamID().value();
  auto & stream = m_stream[sid];

  // time each module
  if (md.id() < stream.fast_modules.size()) {
    ModuleInfo & module = * stream.fast_modules[md.id()];
    module.run_in_path = stream.current_path;
    module.timer.start();
    stream.current_module = & module;
    // used to measure the time spent between the beginning of the path and the execution of the first module
    if (stream.current_path->first_module == nullptr)
      stream.current_path->first_module = & module;
  } else {
    // should never get here
    edm::LogError("FastTimerService") << "FastTimerService::preModuleEvent: unexpected module " << md.moduleLabel();
  }
  */
}

void FastTimerService::postModuleEvent(edm::StreamContext const & sc, edm::ModuleCallingContext const & mcc) {
  edm::ModuleDescription const & md = * mcc.moduleDescription();
  unsigned int id  = md.id();
  unsigned int sid = sc.streamID().value();
  auto & stream = streams_[sid];

  thread().measure(stream.modules[id]);

  /*
  // this is ever called only if m_enable_timing_modules = true
  assert(m_enable_timing_modules);

  if (mcc.moduleDescription() == nullptr) {
    edm::LogError("FastTimerService") << "FastTimerService::postModuleEvent: invalid module";
    return;
  }

  edm::ModuleDescription const & md = * mcc.moduleDescription();
  unsigned int sid = sc.streamID().value();
  auto & stream = m_stream[sid];

  double active = 0.;

  // time and account each module
  if (md.id() < stream.fast_modules.size()) {
    ModuleInfo & module = * stream.fast_modules[md.id()];
    module.timer.stop();
    active = module.timer.seconds();
    module.time_active     = active;
    module.summary_active += active;
    // plots are filled post event processing
  } else {
    // should never get here
    edm::LogError("FastTimerService") << "FastTimerService::postModuleEvent: unexpected module " << md.moduleLabel();
  }
  */
}

void FastTimerService::preModuleEventDelayedGet(edm::StreamContext const & sc, edm::ModuleCallingContext const & mcc) {
  /*
  // this is ever called only if m_enable_timing_modules = true
  assert(m_enable_timing_modules);

  if (mcc.moduleDescription() == nullptr) {
    edm::LogError("FastTimerService") << "FastTimerService::postModuleEventDelayedGet: invalid module";
    return;
  }

  edm::ModuleDescription const & md = * mcc.moduleDescription();
  unsigned int sid = sc.streamID().value();
  auto & stream = m_stream[sid];

  // if the ModuleCallingContext state is "Pefetching", the module is not running,
  // and is asking for its dependencies due to a "consumes" declaration.
  // we can ignore this signal.

  // if the ModuleCallingContext state is "Running", the module was running:
  // it declared its dependencies as "mayConsume", and is now calling getByToken/getByLabel.
  // we pause the timer for this module, and resume it later in the postModuleEventDelayedGet signal.

  // if the ModuleCallingContext state is "Invalid", we ignore the signal.

  if (mcc.state() == edm::ModuleCallingContext::State::kRunning) {
    if (md.id() < stream.fast_modules.size()) {
      ModuleInfo & module = * stream.fast_modules[md.id()];
      module.timer.pause();
    } else {
      // should never get here
      edm::LogError("FastTimerService") << "FastTimerService::preModuleEventDelayedGet: unexpected module " << md.moduleLabel();
    }
  }
  */
}

void FastTimerService::postModuleEventDelayedGet(edm::StreamContext const & sc, edm::ModuleCallingContext const & mcc) {
  /*
  // this is ever called only if m_enable_timing_modules = true
  assert(m_enable_timing_modules);

  if (mcc.moduleDescription() == nullptr) {
    edm::LogError("FastTimerService") << "FastTimerService::postModuleEventDelayedGet: invalid module";
    return;
  }

  edm::ModuleDescription const & md = * mcc.moduleDescription();
  unsigned int sid = sc.streamID().value();
  auto & stream = m_stream[sid];

  // see the description of the possible ModuleCallingContext states in preModuleEventDelayedGet, above.
  if (mcc.state() == edm::ModuleCallingContext::State::kRunning) {
    if (md.id() < stream.fast_modules.size()) {
      ModuleInfo & module = * stream.fast_modules[md.id()];
      module.timer.resume();
    } else {
      // should never get here
      edm::LogError("FastTimerService") << "FastTimerService::postModuleEventDelayedGet: unexpected module " << md.moduleLabel();
    }
  }
  */
}

void
FastTimerService::preModuleEventPrefetching(edm::StreamContext const&, edm::ModuleCallingContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::postModuleEventPrefetching(edm::StreamContext const&, edm::ModuleCallingContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::preEventReadFromSource(edm::StreamContext const&, edm::ModuleCallingContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::postEventReadFromSource(edm::StreamContext const&, edm::ModuleCallingContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::preModuleGlobalBeginRun(edm::GlobalContext const&, edm::ModuleCallingContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::postModuleGlobalBeginRun(edm::GlobalContext const&, edm::ModuleCallingContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::preModuleGlobalEndRun(edm::GlobalContext const&, edm::ModuleCallingContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::postModuleGlobalEndRun(edm::GlobalContext const&, edm::ModuleCallingContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::preModuleGlobalBeginLumi(edm::GlobalContext const&, edm::ModuleCallingContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::postModuleGlobalBeginLumi(edm::GlobalContext const&, edm::ModuleCallingContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::preModuleGlobalEndLumi(edm::GlobalContext const&, edm::ModuleCallingContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::postModuleGlobalEndLumi(edm::GlobalContext const&, edm::ModuleCallingContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::preModuleStreamBeginRun(edm::StreamContext const&, edm::ModuleCallingContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::postModuleStreamBeginRun(edm::StreamContext const&, edm::ModuleCallingContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::preModuleStreamEndRun(edm::StreamContext const&, edm::ModuleCallingContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::postModuleStreamEndRun(edm::StreamContext const&, edm::ModuleCallingContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::preModuleStreamBeginLumi(edm::StreamContext const&, edm::ModuleCallingContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::postModuleStreamBeginLumi(edm::StreamContext const&, edm::ModuleCallingContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::preModuleStreamEndLumi(edm::StreamContext const&, edm::ModuleCallingContext const&)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::postModuleStreamEndLumi(edm::StreamContext const&, edm::ModuleCallingContext const&)
{
  unsupportedSignal(__func__);
}

/*
// associate to a path all the modules it contains
void FastTimerService::fillPathMap(unsigned int pid, std::string const & name, std::vector<std::string> const & modules) {
  for (auto & stream: m_stream) {

    std::vector<ModuleInfo *> & pathmap = stream.paths[pid][name].modules;
    pathmap.clear();
    pathmap.reserve( modules.size() );
    std::unordered_set<ModuleInfo const *> pool;        // keep track of inserted modules
    for (auto const & module: modules) {
      // fix the name of negated or ignored modules
      std::string const & label = (module[0] == '!' or module[0] == '-') ? module.substr(1) : module;

      auto const & it = stream.modules.find(label);
      if (it == stream.modules.end()) {
        // no matching module was found
        pathmap.push_back( 0 );
      } else if (pool.insert(& it->second).second) {
        // new module
        pathmap.push_back(& it->second);
      } else {
        // duplicate module
        pathmap.push_back( 0 );
      }
    }

  }
}

// find the first and last non-empty paths, optionally skipping the first one
std::pair<std::string,std::string> FastTimerService::findFirstLast(unsigned int pid, std::vector<std::string> const & paths) {
  std::vector<std::string const *> p(paths.size(), nullptr);

  // mark the empty paths
  auto address_if_non_empty = [&](std::string const & name){
    return m_stream.front().paths[pid][name].modules.empty() ? nullptr : & name;
  };
  std::transform(paths.begin(), paths.end(), p.begin(), address_if_non_empty);

  // remove the empty paths
  p.erase(std::remove(p.begin(), p.end(), nullptr), p.end());

  // return the first and last non-empty paths, if there are any
  if (not p.empty())
    return std::make_pair(* p.front(), * p.back());
  else
    return std::make_pair(std::string(), std::string());
}
*/


unsigned int
FastTimerService::threadId()
{
  static unsigned int unique_thread_id = 0;
  static thread_local unsigned int thread_id = unique_thread_id++;
  return thread_id;
}

FastTimerService::Measurement &
FastTimerService::thread()
{
  return threads_.at(threadId());
}


// describe the module's configuration
void FastTimerService::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>(   "useRealTimeClock",         true);
  desc.addUntracked<bool>(   "enableTimingPaths",        true);
  desc.addUntracked<bool>(   "enableTimingModules",      true);
  desc.addUntracked<bool>(   "enableTimingExclusive",    false);
  desc.addUntracked<bool>(   "enableTimingSummary",      false);
  desc.addUntracked<bool>(   "enableDQM",                true);
  desc.addUntracked<bool>(   "enableDQMbyPathActive",    false);
  desc.addUntracked<bool>(   "enableDQMbyPathTotal",     true);
  desc.addUntracked<bool>(   "enableDQMbyPathOverhead",  false);
  desc.addUntracked<bool>(   "enableDQMbyPathDetails",   false);
  desc.addUntracked<bool>(   "enableDQMbyPathCounters",  true);
  desc.addUntracked<bool>(   "enableDQMbyPathExclusive", false);
  desc.addUntracked<bool>(   "enableDQMbyModule",        false);
  desc.addUntracked<bool>(   "enableDQMbyModuleType",    false);
  desc.addUntracked<bool>(   "enableDQMSummary",         false);
  desc.addUntracked<bool>(   "enableDQMbyLumiSection",   false);
  desc.addUntracked<bool>(   "enableDQMbyProcesses",     false);
  desc.addUntracked<double>( "dqmTimeRange",             1000. );   // ms
  desc.addUntracked<double>( "dqmTimeResolution",           5. );   // ms
  desc.addUntracked<double>( "dqmPathTimeRange",          100. );   // ms
  desc.addUntracked<double>( "dqmPathTimeResolution",       0.5);   // ms
  desc.addUntracked<double>( "dqmModuleTimeRange",         40. );   // ms
  desc.addUntracked<double>( "dqmModuleTimeResolution",     0.2);   // ms
  desc.addUntracked<uint32_t>( "dqmLumiSectionsRange",   2500  );   // ~ 16 hours
  desc.addUntracked<std::string>(   "dqmPath",           "HLT/TimerService");
  descriptions.add("FastTimerService", desc);
}

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
#include <boost/range/irange.hpp>

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


///////////////////////////////////////////////////////////////////////////////

template <class Rep, class Period>
double ms(std::chrono::duration<Rep, Period> duration)
{
  return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(duration).count();;
}

template <class Rep, class Period>
double ms(boost::chrono::duration<Rep, Period> duration)
{
  return boost::chrono::duration_cast<boost::chrono::duration<double, boost::milli>>(duration).count();;
}

///////////////////////////////////////////////////////////////////////////////

FastTimerService::PlotsPerElement::PlotsPerElement() :
  time_thread_(nullptr),
  time_thread_byls_(nullptr),
  time_real_(nullptr),
  time_real_byls_(nullptr)
{
}

void
FastTimerService::PlotsPerElement::reset()
{
  // the plots are owned by the DQMStore
  time_thread_      = nullptr;
  time_thread_byls_ = nullptr;
  time_real_        = nullptr;
  time_real_byls_   = nullptr;
}

void
FastTimerService::PlotsPerElement::book(
    DQMStore::IBooker & booker,
    std::string const& name,
    std::string const& title,
    double range,
    double resolution,
    unsigned int lumisections)
{
  int bins = (int) std::ceil(range / resolution);
  std::string y_title = (boost::format("events / %f ms") % resolution).str();

  time_thread_ = booker.book1D(
      name + "_time_thread",
      title + " processing time (cpu)",
      bins, 0., range
      )->getTH1F();
  time_thread_->StatOverflows(true);
  time_thread_->SetXTitle("processing time [ms]");
  time_thread_->SetYTitle(y_title.c_str());

  time_real_ = booker.book1D(
      name + "_time_real",
      title + " processing time (real)",
      bins, 0., range
      )->getTH1F();
  time_real_->StatOverflows(true);
  time_real_->SetXTitle("processing time [ms]");
  time_real_->SetYTitle(y_title.c_str());

  time_thread_byls_ = booker.bookProfile(
      name + "_time_thread_byls",
      title + " processing time (cpu) vs. lumisection",
      lumisections, 0.5, lumisections + 0.5,
      bins, 0., std::numeric_limits<double>::infinity(),
      " ")->getTProfile();
  time_thread_byls_->StatOverflows(true);
  time_thread_byls_->SetXTitle("lumisection");
  time_thread_byls_->SetYTitle("processing time [ms]");

  time_real_byls_ = booker.bookProfile(
      name + "_time_real_byls",
      title + " processing time (real) vs. lumisection",
      lumisections, 0.5, lumisections + 0.5,
      bins, 0., std::numeric_limits<double>::infinity(),
      " ")->getTProfile();
  time_real_byls_->StatOverflows(true);
  time_real_byls_->SetXTitle("lumisection");
  time_real_byls_->SetYTitle("processing time [ms]");
}

void
FastTimerService::PlotsPerElement::fill(Resources const& data, unsigned int lumisection)
{
  if (time_thread_)
    time_thread_->Fill(ms(data.time_thread));

  if (time_thread_byls_)
    time_thread_byls_->Fill(lumisection, ms(data.time_thread));

  if (time_real_)
    time_real_->Fill(ms(data.time_real));

  if (time_real_byls_)
    time_real_byls_->Fill(lumisection, ms(data.time_real));
}


FastTimerService::PlotsPerPath::PlotsPerPath() :
  total_(),
  module_counter_(nullptr),
  module_time_thread_total_(nullptr),
  module_time_real_total_(nullptr)
{
}

void
FastTimerService::PlotsPerPath::reset()
{
  // the plots are owned by the DQMStore
  total_.reset();
  module_counter_            = nullptr;
  module_time_thread_total_  = nullptr;
  module_time_real_total_    = nullptr;
}

void
FastTimerService::PlotsPerPath::book(
    DQMStore::IBooker & booker,
    ProcessCallGraph const& job,
    ProcessCallGraph::PathType const& path,
    double range,
    double resolution,
    unsigned int lumisections)
{
  total_.book(booker, path.name_, path.name_, range, resolution, lumisections);

  unsigned int bins = path.modules_and_dependencies_.size();
  module_counter_ = booker.book1D(
      path.name_ + "_module_counter",
      path.name_ + " module counter",
      bins + 1, -0.5, bins + 0.5
      )->getTH1F();
  module_counter_->SetYTitle("processing time [ms]");
  module_time_thread_total_ = booker.book1D(
      path.name_ + "_module_time_thread_total",
      path.name_ + " total module time (cpu)",
      bins, -0.5, bins - 0.5
      )->getTH1F();
  module_time_thread_total_->SetYTitle("processing time [ms]");
  module_time_real_total_ = booker.book1D(
      path.name_ + "_module_time_real_total",
      path.name_ + " total module time (real)",
      bins, -0.5, bins - 0.5
      )->getTH1F();
  module_time_real_total_->SetYTitle("processing time [ms]");
  for (unsigned int bin: boost::irange(0u, bins)) {
    auto const& module = job[path.modules_and_dependencies_[bin]];
    std::string const& label = module.scheduled_ ? module.module_.moduleLabel() : module.module_.moduleLabel() + " (unscheduled)";
    module_counter_          ->GetXaxis()->SetBinLabel(bin + 1, label.c_str());
    module_time_thread_total_->GetXaxis()->SetBinLabel(bin + 1, label.c_str());
    module_time_real_total_  ->GetXaxis()->SetBinLabel(bin + 1, label.c_str());
  }
  module_counter_->GetXaxis()->SetBinLabel(bins + 1, "");
}

void
FastTimerService::PlotsPerPath::fill(ProcessCallGraph::PathType const& description, ResourcesPerJob const& data, ResourcesPerPath const& path, unsigned int ls)
{
  // fill the total path time
  total_.fill(path.total, ls);

  // fill the modules that actually ran and the total time spent in each od them
  for (unsigned int i = 0; i < path.last; ++i) {
    auto const& module = data.modules[description.modules_and_dependencies_[i]];
    module_counter_->Fill(i);
    module_time_thread_total_->Fill(i, ms(module.time_thread));
    module_time_real_total_->Fill(i, ms(module.time_real));
  }
  if (path.status)
    module_counter_->Fill(path.last);
}


FastTimerService::PlotsPerProcess::PlotsPerProcess(ProcessCallGraph::ProcessType const& process) :
  event_(),
  paths_(process.paths_.size()),
  endpaths_(process.endPaths_.size())
{
}

void
FastTimerService::PlotsPerProcess::reset()
{
  event_.reset();
  for (auto & path: paths_)
    path.reset();
  for (auto & path: endpaths_)
    path.reset();
}

void
FastTimerService::PlotsPerProcess::book(
    DQMStore::IBooker & booker,
    ProcessCallGraph const& job,
    ProcessCallGraph::ProcessType const& process,
    double event_range,
    double event_resolution,
    double path_range,
    double path_resolution,
    unsigned int lumisections)
{
  const std::string basedir = booker.pwd();
  event_.book(booker,
      process.name_, "process " + process.name_,
      event_range,
      event_resolution,
      lumisections);
  booker.setCurrentFolder(basedir + "/process " + process.name_ + " paths_");
  for (unsigned int id: boost::irange(0ul, paths_.size()))
  {
    paths_[id].book(booker,
        job, process.paths_[id],
        path_range,
        path_resolution,
        lumisections);
  }
  for (unsigned int id: boost::irange(0ul, endpaths_.size()))
  {
    endpaths_[id].book(booker,
        job, process.endPaths_[id],
        path_range,
        path_resolution,
        lumisections);
  }
  booker.setCurrentFolder(basedir);
}

void
FastTimerService::PlotsPerProcess::fill(ProcessCallGraph::ProcessType const& description, ResourcesPerJob const& data, ResourcesPerProcess const& process, unsigned int ls)
{
  // fill process event plots
  event_.fill(process.total, ls);

  // fill all paths plots
  for (unsigned int id: boost::irange(0ul, paths_.size()))
    paths_[id].fill(description.paths_[id], data, process.paths[id], ls);

  // fill all endpaths plots
  for (unsigned int id: boost::irange(0ul, endpaths_.size()))
    endpaths_[id].fill(description.endPaths_[id], data, process.endpaths[id], ls);
}


FastTimerService::PlotsPerJob::PlotsPerJob() :
  event_(),
  modules_(),
  processes_()
{
}

FastTimerService::PlotsPerJob::PlotsPerJob(ProcessCallGraph const& job) :
  event_(),
  modules_(job.size()),
  processes_()
{
  processes_.reserve(job.processes().size());
  for (auto const& process: job.processes())
    processes_.emplace_back(process);
}

void
FastTimerService::PlotsPerJob::reset()
{
  event_.reset();
  for (auto & module: modules_)
    module.reset();
  for (auto & process: processes_)
    process.reset();
}

void
FastTimerService::PlotsPerJob::fill(ProcessCallGraph const& job, ResourcesPerJob const& data, unsigned int ls)
{
  // fill total event plots
  event_.fill(data.total, ls);

  // fill modules plots
  for (unsigned int id: boost::irange(0ul, modules_.size()))
    modules_[id].fill(data.modules[id], ls);

  for (unsigned int pid: boost::irange(0ul, processes_.size()))
    processes_[pid].fill(job.processDescription(pid), data, data.processes[pid], ls);
}


void
FastTimerService::PlotsPerJob::book(
    DQMStore::IBooker & booker,
    ProcessCallGraph const& job,
    double event_range,
    double event_resolution,
    double path_range,
    double path_resolution,
    double module_range,
    double module_resolution,
    unsigned int lumisections)
{
  const std::string basedir = booker.pwd();

  // event summary plots
  event_.book(booker,
      "event", "Event",
      event_range,
      event_resolution,
      lumisections);

  modules_[job.source().id()].book(booker,
      "source", "Source",
      module_range,
      module_resolution,
      lumisections);

  // plots per subprocess (event, modules, paths and endpaths)
  for (unsigned int pid: boost::irange(0ul, job.processes().size())) {
    auto const& process = job.processDescription(pid);
    processes_[pid].book(booker,
        job, process,
        event_range,
        event_resolution,
        path_range,
        path_resolution,
        lumisections);

    booker.setCurrentFolder(basedir + "/process " + process.name_ + " modules");
    for (unsigned int id: process.modules_)
    {
      auto const& module_name = job.module(id).moduleLabel();
      modules_[id].book(booker,
          module_name, module_name,
          module_range,
          module_resolution,
          lumisections);
    }
    booker.setCurrentFolder(basedir);
  }
}


///////////////////////////////////////////////////////////////////////////////

FastTimerService::FastTimerService(const edm::ParameterSet & config, edm::ActivityRegistry & registry) :
  // configuration
  callgraph_(),
  // job configuration
  concurrent_runs_(             0 ),
  concurrent_streams_(          0 ),
  concurrent_threads_(          0 ),
  // dqm configuration
  module_id_(                   edm::ModuleDescription::invalidID() ),
  enable_dqm_(                  config.getUntrackedParameter<bool>(     "enableDQM"                ) ),
//enable_dqm_bypath_active_(    config.getUntrackedParameter<bool>(     "enableDQMbyPathActive"    ) ),
//enable_dqm_bypath_total_(     config.getUntrackedParameter<bool>(     "enableDQMbyPathTotal"     ) ),
//enable_dqm_bypath_overhead_(  config.getUntrackedParameter<bool>(     "enableDQMbyPathOverhead"  ) ),
//enable_dqm_bypath_details_(   config.getUntrackedParameter<bool>(     "enableDQMbyPathDetails"   ) ),
//enable_dqm_bypath_counters_(  config.getUntrackedParameter<bool>(     "enableDQMbyPathCounters"  ) ),
//enable_dqm_bypath_exclusive_( config.getUntrackedParameter<bool>(     "enableDQMbyPathExclusive" ) ),
  enable_dqm_bymodule_(         config.getUntrackedParameter<bool>(     "enableDQMbyModule"        ) ),
//enable_dqm_summary_(          config.getUntrackedParameter<bool>(     "enableDQMSummary"         ) ),
  enable_dqm_byls_(             config.getUntrackedParameter<bool>(     "enableDQMbyLumiSection"   ) ),
  enable_dqm_bynproc_(          config.getUntrackedParameter<bool>(     "enableDQMbyProcesses"     ) ),
  dqm_eventtime_range_(         config.getUntrackedParameter<double>(   "dqmTimeRange"             ) ),            // ms
  dqm_eventtime_resolution_(    config.getUntrackedParameter<double>(   "dqmTimeResolution"        ) ),            // ms
  dqm_pathtime_range_(          config.getUntrackedParameter<double>(   "dqmPathTimeRange"         ) ),            // ms
  dqm_pathtime_resolution_(     config.getUntrackedParameter<double>(   "dqmPathTimeResolution"    ) ),            // ms
  dqm_moduletime_range_(        config.getUntrackedParameter<double>(   "dqmModuleTimeRange"       ) ),            // ms
  dqm_moduletime_resolution_(   config.getUntrackedParameter<double>(   "dqmModuleTimeResolution"  ) ),            // ms
  dqm_lumisections_range_(      config.getUntrackedParameter<uint32_t>( "dqmLumiSectionsRange"     ) ),
  dqm_path_(                    config.getUntrackedParameter<std::string>("dqmPath" ) )
  /*
  // DQM - these are initialized at preStreamBeginRun(), to make sure the DQM service has been loaded
  stream_(),
  // summary data
  run_summary_(),
  job_summary_(),
  run_summary_perprocess_(),
  job_summary_perprocess_()
  */
{
  /*
  // enable timers if required by DQM plots
  enable_timing_paths_      = enable_timing_paths_         or
                              enable_dqm_bypath_active_    or
                              enable_dqm_bypath_total_     or
                              enable_dqm_bypath_overhead_  or
                              enable_dqm_bypath_details_   or
                              enable_dqm_bypath_counters_  or
                              enable_dqm_bypath_exclusive_;

  enable_timing_modules_    = enable_timing_modules_       or
                              enable_dqm_bymodule_         or
                              enable_dqm_bypath_total_     or
                              enable_dqm_bypath_overhead_  or
                              enable_dqm_bypath_details_   or
                              enable_dqm_bypath_counters_  or
                              enable_dqm_bypath_exclusive_;

  enable_timing_exclusive_  = enable_timing_exclusive_     or
                              enable_dqm_bypath_exclusive_;
  */

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
//registry.watchPostSourceConstruction(     this, & FastTimerService::postSourceConstruction);
  registry.watchPreSourceRun(               this, & FastTimerService::preSourceRun );
  registry.watchPostSourceRun(              this, & FastTimerService::postSourceRun );
  registry.watchPreSourceLumi(              this, & FastTimerService::preSourceLumi );
  registry.watchPostSourceLumi(             this, & FastTimerService::postSourceLumi );
  registry.watchPreSourceEvent(             this, & FastTimerService::preSourceEvent );
  registry.watchPostSourceEvent(            this, & FastTimerService::postSourceEvent );
//registry.watchPreModuleBeginJob(          this, & FastTimerService::preModuleBeginJob );
//registry.watchPostModuleBeginJob(         this, & FastTimerService::postModuleBeginJob );
//registry.watchPreModuleEndJob(            this, & FastTimerService::preModuleEndJob );
//registry.watchPostModuleEndJob(           this, & FastTimerService::postModuleEndJob );
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
  registry.watchPreEventReadFromSource(     this, & FastTimerService::preEventReadFromSource );
  registry.watchPostEventReadFromSource(    this, & FastTimerService::postEventReadFromSource );
}

FastTimerService::~FastTimerService()
{
}

double
FastTimerService::querySourceTime(edm::StreamID sid) const
{
  auto const& stream = streams_[sid];
  auto const& source = stream.modules[callgraph_.source().id()];
  return ms(source.time_real);
}

double
FastTimerService::queryEventTime(edm::StreamID sid) const
{
  auto const& stream = streams_[sid];
  return ms(stream.total.time_real);
}

double
FastTimerService::queryEventTime(edm::StreamID sid, std::string const& process) const
{
  unsigned int pid = callgraph_.processId(process);
  auto const& stream = streams_[sid];
  return ms(stream.processes[pid].total.time_real);
}

double
FastTimerService::queryModuleTime(edm::StreamID sid, const edm::ModuleDescription & md) const
{
  auto const& stream = streams_[sid];
  auto const& module = stream.modules[md.id()];
  return ms(module.time_real);
}

double
FastTimerService::queryModuleTime(edm::StreamID sid, unsigned int id) const
{
  auto const& stream = streams_[sid];
  auto const& module = stream.modules[id];
  //FIXME add a check that "id" is valid
  return ms(module.time_real);
}

double
FastTimerService::queryModuleTimeByLabel(edm::StreamID sid, std::string const& label) const
{
  for (unsigned int id: boost::irange(0u, callgraph_.size()))
    if (callgraph_.module(id).moduleLabel() == label)
      return queryModuleTime(sid, id);

  //FIXME issue a LogWarning or raise an exception
  return 0.;
}

double
FastTimerService::queryModuleTimeByLabel(edm::StreamID sid, std::string const& process, const std::string & label) const
{
  for (unsigned int id: callgraph_.processDescription(process).modules_)
    if (callgraph_.module(id).moduleLabel() == label)
      return queryModuleTime(sid, id);

  //FIXME issue a LogWarning or raise an exception
  return 0.;
}

double
FastTimerService::queryPathTime(edm::StreamID sid, std::string const& path) const
{
  auto const& stream = streams_[sid];
  for (unsigned int pid: boost::irange(0ul, callgraph_.processes().size()))
  {
    auto const& desc = callgraph_.processDescription(pid);
    for (unsigned int id: boost::irange(0ul, desc.paths_.size()))
      if (desc.paths_[id].name_ == path)
        return ms(stream.processes[pid].paths[id].total.time_real);
    for (unsigned int id: boost::irange(0ul, desc.endPaths_.size()))
      if (desc.paths_[id].name_ == path)
        return ms(stream.processes[pid].endpaths[id].total.time_real);
  }

  //FIXME issue a LogWarning or raise an exception
  return 0.;
}

double
FastTimerService::queryPathTime(edm::StreamID sid, std::string const& process, std::string const& path) const
{
  auto const& stream = streams_[sid];
  unsigned int pid = callgraph_.processId(process);
  auto const& desc = callgraph_.processDescription(pid);
  for (unsigned int id: boost::irange(0ul, desc.paths_.size()))
    if (desc.paths_[id].name_ == path)
      return ms(stream.processes[pid].paths[id].total.time_real);
  for (unsigned int id: boost::irange(0ul, desc.endPaths_.size()))
    if (desc.paths_[id].name_ == path)
      return ms(stream.processes[pid].endpaths[id].total.time_real);

  //FIXME issue a LogWarning or raise an exception
  return 0.;
}


void
FastTimerService::ignoredSignal(std::string signal) const
{
  LogDebug("FastTimerService") << "The FastTimerService received is currently not monitoring the signal \"" << signal << "\".\n";
}

void
FastTimerService::unsupportedSignal(std::string signal) const
{
  // warn about each signal only once per job
  if (unsupported_signals_.insert(signal).second)
    edm::LogWarning("FastTimerService") << "The FastTimerService received the unsupported signal \"" << signal << "\".\n"
      << "Please report how to reproduce the issue to cms-hlt@cern.ch .";
}

void
FastTimerService::preGlobalBeginRun(edm::GlobalContext const& gc)
{
  ignoredSignal(__func__);
}

void
FastTimerService::postGlobalBeginRun(edm::GlobalContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::preStreamBeginRun(edm::StreamContext const& sc)
{
  unsigned int sid = sc.streamID().value();

  // book the DQM plots for each stream during the main process being run
  if (enable_dqm_ and not sc.processContext()->isSubProcess()) {

    // define a callback to book the MonitorElements
    auto bookTransactionCallback = [&, this] (DQMStore::IBooker & booker)
    {
      booker.setCurrentFolder(dqm_path_);
      stream_plots_[sid].book(booker, callgraph_,
          dqm_eventtime_range_,
          dqm_eventtime_resolution_,
          dqm_pathtime_range_,
          dqm_pathtime_resolution_,
          dqm_moduletime_range_,
          dqm_moduletime_resolution_,
          dqm_lumisections_range_);
    };

    // book MonitorElements for this stream
    edm::Service<DQMStore>()->bookTransaction(bookTransactionCallback, sc.eventID().run(), sid, module_id_);
  }

  ignoredSignal(__func__);
}


void
FastTimerService::preallocate(edm::service::SystemBounds const& bounds)
{
  concurrent_runs_    = bounds.maxNumberOfConcurrentRuns();
  concurrent_streams_ = bounds.maxNumberOfStreams();
  concurrent_threads_ = bounds.maxNumberOfThreads();

  if (enable_dqm_bynproc_)
    dqm_path_ += (boost::format("/Running %d processes") % concurrent_threads_).str();

  /*
  run_summary_.resize(concurrent_runs_);
  run_summary_perprocess_.resize(concurrent_runs_);
  stream_.resize(concurrent_streams_);
  */

  // assign a pseudo module id to the FastTimerService
  module_id_ = edm::ModuleDescription::getUniqueID();
}

void
FastTimerService::preSourceConstruction(edm::ModuleDescription const& module) {
  callgraph_.preSourceConstruction(module);
}

void
FastTimerService::preBeginJob(edm::PathsAndConsumesOfModulesBase const& pathsAndConsumes, edm::ProcessContext const & context) {
  callgraph_.preBeginJob(pathsAndConsumes, context);
}

void
FastTimerService::postBeginJob() {
  unsigned int modules   = callgraph_.size();
  unsigned int processes = callgraph_.processes().size();

  // allocate the resource measurements per thread
  threads_.resize(concurrent_threads_);

  // allocate the resource counters for each stream, process, path and module
  streams_.resize(concurrent_streams_);
  for (auto & stream: streams_) {
    // FIXME move this into the constructor for ResourcesPerJob ?
    stream.modules.resize(modules);
    stream.processes.resize(processes);
    for (unsigned int i = 0; i < processes; ++i) {
      auto const& process = callgraph_.processDescription(i);
      stream.processes[i] = {
        Resources(),
        std::vector<ResourcesPerPath>(process.paths_.size()),
        std::vector<ResourcesPerPath>(process.endPaths_.size())
      };
    }
  }

  // check that the DQMStore service is available
  if (enable_dqm_ and not edm::Service<DQMStore>().isAvailable()) {
    // the DQMStore is not available, disable all DQM plots
    enable_dqm_ = false;
    // FIXME LogWarning ?
  }

  // allocate the structures to hold pointers to the DQM plots
  if (enable_dqm_)
    stream_plots_.resize(concurrent_threads_, PlotsPerJob(callgraph_));

}

void
FastTimerService::postStreamBeginRun(edm::StreamContext const& sc)
{
  ignoredSignal(__func__);
}

void
FastTimerService::preStreamEndRun(edm::StreamContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::postStreamEndRun(edm::StreamContext const& sc)
{
  unsigned int sid = sc.streamID().value();

  if (enable_dqm_) {
    DQMStore & store = * edm::Service<DQMStore>();
    store.mergeAndResetMEsRunSummaryCache(sc.eventID().run(), sid, module_id_);
  }

  ignoredSignal(__func__);
}

void
FastTimerService::preGlobalBeginLumi(edm::GlobalContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::postGlobalBeginLumi(edm::GlobalContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::preGlobalEndLumi(edm::GlobalContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::postGlobalEndLumi(edm::GlobalContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::preStreamBeginLumi(edm::StreamContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::postStreamBeginLumi(edm::StreamContext const& sc) {
  ignoredSignal(__func__);
}

void
FastTimerService::preStreamEndLumi(edm::StreamContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::postStreamEndLumi(edm::StreamContext const& sc) {
  unsigned int sid = sc.streamID().value();

  if (enable_dqm_) {
    DQMStore & store = * edm::Service<DQMStore>();
    store.mergeAndResetMEsLuminositySummaryCache(sc.eventID().run(),sc.luminosityBlockIndex(),sid, module_id_);
  }

  ignoredSignal(__func__);
}

void
FastTimerService::preGlobalEndRun(edm::GlobalContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::postGlobalEndRun(edm::GlobalContext const& gc)
{
  ignoredSignal(__func__);

  // TODO
  // print run summary
}

void
FastTimerService::preSourceRun()
{
  ignoredSignal(__func__);
}

void
FastTimerService::postSourceRun()
{
  ignoredSignal(__func__);
}

void
FastTimerService::preSourceLumi()
{
  ignoredSignal(__func__);
}

void
FastTimerService::postSourceLumi()
{
  ignoredSignal(__func__);
}

void
FastTimerService::postEndJob()
{
  // TODO
  // print job summary
}

/*
void
FastTimerService::printProcessSummary(Timing const& total, TimingPerProcess const & summary, std::string const & label, std::string const & process) const
{
  // print a timing summary for the run or job, for each subprocess
  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  out << "FastReport for " << label << ", process " << process << '\n';
  //out << "FastReport " << (use_realtime_ ? "(real time) " : "(CPU time)  ") << '\n';
  out << "FastReport              " << std::right << std::setw(10) << summary.event        / (double) total.count   << "  Event"         << '\n';
  out << "FastReport              " << std::right << std::setw(10) << summary.all_paths    / (double) total.count   << "  all Paths"     << '\n';
  out << "FastReport              " << std::right << std::setw(10) << summary.all_endpaths / (double) total.count   << "  all EndPaths"  << '\n';
  out << "FastReport              " << std::right << std::setw(10) << summary.interpaths   / (double) total.count   << "  between paths" << '\n';
  edm::LogVerbatim("FastReport") << out.str();
}

void
FastTimerService::printSummary(Timing const& summary, std::string const & label) const
{
  // print a timing summary for the run or job
  //edm::service::TriggerNamesService & tns = * edm::Service<edm::service::TriggerNamesService>();

  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  out << "FastReport for " << label << ", over all subprocesses" << '\n';
  //out << "FastReport " << (use_realtime_ ? "(real time) " : "(CPU time)  ") << '\n';
  out << "FastReport              " << std::right << std::setw(10) << summary.source       / (double) summary.count << "  Source"        << '\n';
  out << "FastReport              " << std::right << std::setw(10) << summary.event        / (double) summary.count << "  Event"         << '\n';
  edm::LogVerbatim("FastReport") << out.str();
}
*/

/*
  if (enable_timing_modules_) {
    double modules_total = 0.;
    for (auto & keyval: stream_.modules)
      modules_total += keyval.second.summary_active;
    out << "FastReport              " << std::right << std::setw(10) << modules_total / (double) summary.count << "  all Modules"   << '\n';
  }
  out << '\n';
  if (enable_timing_paths_ and not enable_timing_modules_) {
    //out << "FastReport " << (use_realtime_ ? "(real time) " : "(CPU time)  ")    << "     Active Path" << '\n';
    for (auto const& name: tns.getTrigPaths())
      out << "FastReport              "
          << std::right << std::setw(10) << stream_.paths[pid][name].summary_active / (double) summary.count << "  "
          << name << '\n';
    out << '\n';
    //out << "FastReport " << (use_realtime_ ? "(real time) " : "(CPU time)  ")    << "     Active EndPath" << '\n';
    for (auto const& name: tns.getEndPaths())
      out << "FastReport              "
          << std::right << std::setw(10) << stream_.paths[pid][name].summary_active / (double) summary.count << "  "
          << name << '\n';
  } else if (enable_timing_paths_ and enable_timing_modules_) {
    //out << "FastReport " << (use_realtime_ ? "(real time) " : "(CPU time)  ")    << "     Active       Pre-     Inter-  Post-mods   Overhead      Total  Path" << '\n';
    for (auto const& name: tns.getTrigPaths()) {
      out << "FastReport              "
          << std::right << std::setw(10) << stream_.paths[pid][name].summary_active        / (double) summary.count << " "
          << std::right << std::setw(10) << stream_.paths[pid][name].summary_overhead      / (double) summary.count << " "
          << std::right << std::setw(10) << stream_.paths[pid][name].summary_total         / (double) summary.count << "  "
          << name << '\n';
    }
    out << '\n';
    //out << "FastReport " << (use_realtime_ ? "(real time) " : "(CPU time)  ")    << "     Active       Pre-     Inter-  Post-mods   Overhead      Total  EndPath" << '\n';
    for (auto const& name: tns.getEndPaths()) {
      out << "FastReport              "
          << std::right << std::setw(10) << stream_.paths[pid][name].summary_active        / (double) summary.count << " "
          << std::right << std::setw(10) << stream_.paths[pid][name].summary_overhead      / (double) summary.count << " "
          << std::right << std::setw(10) << stream_.paths[pid][name].summary_total         / (double) summary.count << "  "
          << name << '\n';
    }
  }
  out << '\n';
  if (enable_timing_modules_) {
    //out << "FastReport " << (use_realtime_ ? "(real time) " : "(CPU time)  ")    << "     Active  Module" << '\n';
    for (auto & keyval: stream_.modules) {
      std::string const& label  = keyval.first;
      ModuleInfo  const& module = keyval.second;
      out << "FastReport              " << std::right << std::setw(10) << module.summary_active  / (double) summary.count << "  " << label << '\n';
    }
    //out << "FastReport " << (use_realtime_ ? "(real time) " : "(CPU time)  ")    << "     Active  Module" << '\n';
    out << '\n';
    //out << "FastReport " << (use_realtime_ ? "(real time) " : "(CPU time)  ")    << "     Active  Module" << '\n';
  }
*/

void
FastTimerService::preEvent(edm::StreamContext const& sc)
{
  ignoredSignal(__func__);
}

void
FastTimerService::postEvent(edm::StreamContext const& sc)
{
  ignoredSignal(__func__);

  unsigned int pid = callgraph_.processId(* sc.processContext());
  unsigned int sid = sc.streamID();
  auto & stream  = streams_[sid];
  auto & process = callgraph_.processDescription(pid);

  // compute the event timing as the sum of all modules' timing
  auto & data = stream.processes[pid].total;
  for (unsigned int i: process.modules_)
    data += stream.modules[i];
  stream.total += data;

  // write the summary and fill the plots only after the last subprocess has run
  if (pid != callgraph_.processes().size() - 1)
    return;

  std::ostringstream out;
  out << "Modules:\n";
  auto const& source_d = callgraph_.source();
  auto const& source   = stream.modules[source_d.id()];
  out << boost::format("  %10.3f ms    %10.3f ms    source %s\n") % ms(source.time_thread) % ms(source.time_real) % source_d.moduleLabel();
  for (unsigned int i = 0; i < callgraph_.processes().size(); ++i) {
    auto const& proc_d = callgraph_.processDescription(i);
    auto const& proc   = stream.processes[i];
    out << boost::format("  %10.3f ms    %10.3f ms    process %s\n") % ms(proc.total.time_thread) % ms(proc.total.time_real) % proc_d.name_;
    for (unsigned int m: proc_d.modules_) {
      auto const& module_d = callgraph_.module(m);
      auto const& module   = stream.modules[m];
      out << boost::format("  %10.3f ms    %10.3f ms      %s\n") % ms(module.time_thread) % ms(module.time_real) % module_d.moduleLabel();
    }
  }
  out << boost::format("  %10.3f ms    %10.3f ms    total\n") % ms(stream.total.time_thread) % ms(stream.total.time_real);
  out << std::endl;

  out << "Process:\n";
  out << boost::format("  %10.3f ms    %10.3f ms    source %s\n") % ms(source.time_thread) % ms(source.time_real) % source_d.moduleLabel();
  for (unsigned int i = 0; i < callgraph_.processes().size(); ++i) {
    auto const& proc_d = callgraph_.processDescription(i);
    auto const& proc   = stream.processes[i];
    out << boost::format("  %10.3f ms    %10.3f ms    process %s\n") % ms(proc.total.time_thread) % ms(proc.total.time_real) % proc_d.name_;
    for (unsigned int p = 0; p < proc.paths.size(); ++p) {
      auto const& name = proc_d.paths_[p].name_;
      auto const& path = proc.paths[p];
      out << boost::format("  %10.3f ms    %10.3f ms      %s (active)\n") % ms(path.active.time_thread) % ms(path.active.time_real) % name;
      out << boost::format("  %10.3f ms    %10.3f ms      %s (total)\n")  % ms(path.total.time_thread)  % ms(path.total.time_real)  % name;
    }
    for (unsigned int p = 0; p < proc.endpaths.size(); ++p) {
      auto const& name = proc_d.endPaths_[p].name_;
      auto const& path = proc.endpaths[p];
      out << boost::format("  %10.3f ms    %10.3f ms      %s (active)\n") % ms(path.active.time_thread) % ms(path.active.time_real) % name;
      out << boost::format("  %10.3f ms    %10.3f ms      %s (total)\n")  % ms(path.total.time_thread)  % ms(path.total.time_real)  % name;
    }
  }
  out << boost::format("  %10.3f ms    %10.3f ms    total\n") % ms(stream.total.time_thread) % ms(stream.total.time_real);
  edm::LogVerbatim("FastReport") << out.str();

  if (enable_dqm_)
    stream_plots_[sid].fill(callgraph_, stream, sc.luminosityBlockIndex());
}

void
FastTimerService::preSourceEvent(edm::StreamID sid)
{
  // clear the event counters
  auto & stream = streams_[sid];
  stream.reset();

  thread().measure();
}


void
FastTimerService::postSourceEvent(edm::StreamID sid)
{
  edm::ModuleDescription const& md = callgraph_.source();
  unsigned int id  = md.id();
  auto & stream = streams_[sid];

  thread().measure_and_store(stream.modules[id]);
}


void
FastTimerService::prePathEvent(edm::StreamContext const& sc, edm::PathContext const & pc)
{
  unsigned int sid = sc.streamID().value();
  unsigned int pid = callgraph_.processId(* sc.processContext());
  unsigned int id  = pc.pathID();
  auto & stream = streams_[sid];
  auto & data = pc.isEndPath() ? stream.processes[pid].endpaths[id] : stream.processes[pid].paths[id];
  data.status = false;
  data.last   = 0;
}


void
FastTimerService::postPathEvent(edm::StreamContext const& sc, edm::PathContext const & pc, edm::HLTPathStatus const & status)
{
  unsigned int sid = sc.streamID().value();
  unsigned int pid = callgraph_.processId(* sc.processContext());
  unsigned int id  = pc.pathID();
  auto & stream = streams_[sid];
  auto & data = pc.isEndPath() ? stream.processes[pid].endpaths[id] : stream.processes[pid].paths[id];

  auto const& path = pc.isEndPath() ? callgraph_.processDescription(pid).endPaths_[id] : callgraph_.processDescription(pid).paths_[id];
  unsigned int index = path.modules_on_path_.empty() ? 0 : status.index() + 1;
  data.last          = path.modules_on_path_.empty() ? 0 : path.last_dependency_of_module_[status.index()];

  for (unsigned int i = 0; i < index; ++i) {
    auto const& module = stream.modules[path.modules_on_path_[i]];
    data.active += module;
  }
  for (unsigned int i = 0; i < data.last; ++i) {
    auto const& module = stream.modules[path.modules_and_dependencies_[i]];
    data.total += module;
  }
}

void
FastTimerService::preModuleEvent(edm::StreamContext const& sc, edm::ModuleCallingContext const & mcc)
{
  thread().measure();
}

void
FastTimerService::postModuleEvent(edm::StreamContext const& sc, edm::ModuleCallingContext const & mcc)
{
  edm::ModuleDescription const& md = * mcc.moduleDescription();
  unsigned int id  = md.id();
  unsigned int sid = sc.streamID().value();
  auto & stream = streams_[sid];

  thread().measure_and_store(stream.modules[id]);
}

void
FastTimerService::preModuleEventDelayedGet(edm::StreamContext const& sc, edm::ModuleCallingContext const & mcc)
{
  unsupportedSignal(__func__);
}

void
FastTimerService::postModuleEventDelayedGet(edm::StreamContext const& sc, edm::ModuleCallingContext const & mcc)
{
  unsupportedSignal(__func__);
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
  ignoredSignal(__func__);
}

void
FastTimerService::postModuleGlobalBeginRun(edm::GlobalContext const&, edm::ModuleCallingContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::preModuleGlobalEndRun(edm::GlobalContext const&, edm::ModuleCallingContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::postModuleGlobalEndRun(edm::GlobalContext const&, edm::ModuleCallingContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::preModuleGlobalBeginLumi(edm::GlobalContext const&, edm::ModuleCallingContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::postModuleGlobalBeginLumi(edm::GlobalContext const&, edm::ModuleCallingContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::preModuleGlobalEndLumi(edm::GlobalContext const&, edm::ModuleCallingContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::postModuleGlobalEndLumi(edm::GlobalContext const&, edm::ModuleCallingContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::preModuleStreamBeginRun(edm::StreamContext const&, edm::ModuleCallingContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::postModuleStreamBeginRun(edm::StreamContext const&, edm::ModuleCallingContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::preModuleStreamEndRun(edm::StreamContext const&, edm::ModuleCallingContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::postModuleStreamEndRun(edm::StreamContext const&, edm::ModuleCallingContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::preModuleStreamBeginLumi(edm::StreamContext const&, edm::ModuleCallingContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::postModuleStreamBeginLumi(edm::StreamContext const&, edm::ModuleCallingContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::preModuleStreamEndLumi(edm::StreamContext const&, edm::ModuleCallingContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::postModuleStreamEndLumi(edm::StreamContext const&, edm::ModuleCallingContext const&)
{
  ignoredSignal(__func__);
}

/*
// associate to a path all the modules it contains
void
FastTimerService::fillPathMap(unsigned int pid, std::string const& name, std::vector<std::string> const & modules)
{
  for (auto & stream: stream_) {

    std::vector<ModuleInfo *> & pathmap = stream.paths[pid][name].modules;
    pathmap.clear();
    pathmap.reserve( modules.size() );
    std::unordered_set<ModuleInfo const *> pool;        // keep track of inserted modules
    for (auto const& module: modules) {
      // fix the name of negated or ignored modules
      std::string const& label = (module[0] == '!' or module[0] == '-') ? module.substr(1) : module;

      auto const& it = stream.modules.find(label);
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
std::pair<std::string,std::string> FastTimerService::findFirstLast(unsigned int pid, std::vector<std::string> const& paths) {
  std::vector<std::string const *> p(paths.size(), nullptr);

  // mark the empty paths
  auto address_if_non_empty = [&](std::string const& name){
    return stream_.front().paths[pid][name].modules.empty() ? nullptr : & name;
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
void
FastTimerService::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
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

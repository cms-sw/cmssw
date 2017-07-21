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

// local headers
#include "memory_usage.h"

///////////////////////////////////////////////////////////////////////////////

namespace {

  // convert any std::chrono::duration to milliseconds
  template <class Rep, class Period>
  double ms(std::chrono::duration<Rep, Period> duration)
  {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(duration).count();;
  }

  // convert any boost::chrono::duration to milliseconds
  template <class Rep, class Period>
  double ms(boost::chrono::duration<Rep, Period> duration)
  {
    return boost::chrono::duration_cast<boost::chrono::duration<double, boost::milli>>(duration).count();;
  }

  // convert from bytes to kilobytes, rounding down
  uint64_t kB(uint64_t bytes)
  {
    return bytes / 1024;
  }
} // namespace

///////////////////////////////////////////////////////////////////////////////

// resources being monitored by the service

// Resources

FastTimerService::Resources::Resources() :
  time_thread(boost::chrono::nanoseconds::zero()),
  time_real(boost::chrono::nanoseconds::zero()),
  allocated(0UL),
  deallocated(0UL)
{ }

void
FastTimerService::Resources::reset() {
  time_thread = boost::chrono::nanoseconds::zero();
  time_real   = boost::chrono::nanoseconds::zero();
  allocated   = 0UL;
  deallocated = 0UL;
}

FastTimerService::Resources &
FastTimerService::Resources::operator+=(Resources const& other) {
  time_thread += other.time_thread;
  time_real   += other.time_real;
  allocated   += other.allocated;
  deallocated += other.deallocated;
  return *this;
}

FastTimerService::Resources
FastTimerService::Resources::operator+(Resources const& other) const {
  Resources result(*this);
  result += other;
  return result;
}

// ResourcesPerModule

FastTimerService::ResourcesPerModule::ResourcesPerModule()
  = default;

void
FastTimerService::ResourcesPerModule::reset() {
  total.reset();
  events = 0;
}

FastTimerService::ResourcesPerModule &
FastTimerService::ResourcesPerModule::operator+=(ResourcesPerModule const& other) {
  total  += other.total;
  events += other.events;
  return *this;
}

FastTimerService::ResourcesPerModule
FastTimerService::ResourcesPerModule::operator+(ResourcesPerModule const& other) const {
  ResourcesPerModule result(*this);
  result += other;
  return result;
}


// ResourcesPerPath

FastTimerService::ResourcesPerPath::ResourcesPerPath()
  = default;

void
FastTimerService::ResourcesPerPath::reset() {
  active.reset();
  total.reset();
  last = 0;
  status = false;
}

FastTimerService::ResourcesPerPath &
FastTimerService::ResourcesPerPath::operator+=(ResourcesPerPath const& other) {
  active += other.active;
  total  += other.total;
  last   = 0;           // summing these makes no sense, reset them instead
  status = false;
  return *this;
}

FastTimerService::ResourcesPerPath
FastTimerService::ResourcesPerPath::operator+(ResourcesPerPath const& other) const {
  ResourcesPerPath result(*this);
  result += other;
  return result;
}


// ResourcesPerProcess

FastTimerService::ResourcesPerProcess::ResourcesPerProcess(ProcessCallGraph::ProcessType const& process) :
  total(),
  paths(process.paths_.size()),
  endpaths(process.endPaths_.size())
{
}

void
FastTimerService::ResourcesPerProcess::reset() {
  total.reset();
  for (auto & path: paths)
    path.reset();
  for (auto & path: endpaths)
    path.reset();
}

FastTimerService::ResourcesPerProcess &
FastTimerService::ResourcesPerProcess::operator+=(ResourcesPerProcess const& other) {
  total += other.total;
  assert(paths.size() == other.paths.size());
  for (unsigned int i: boost::irange(0ul, paths.size()))
    paths[i]    += other.paths[i];
  assert(endpaths.size() == other.endpaths.size());
  for (unsigned int i: boost::irange(0ul, endpaths.size()))
    endpaths[i] += other.endpaths[i];
  return *this;
}

FastTimerService::ResourcesPerProcess
FastTimerService::ResourcesPerProcess::operator+(ResourcesPerProcess const& other) const {
  ResourcesPerProcess result(*this);
  result += other;
  return result;
}


// ResourcesPerJob

FastTimerService::ResourcesPerJob::ResourcesPerJob()
  = default;

FastTimerService::ResourcesPerJob::ResourcesPerJob(ProcessCallGraph const& job, std::vector<GroupOfModules> const& groups) :
  total(),
  highlight( groups.size() ),
  modules( job.size() ),
  processes(),
  events(0)
{
  processes.reserve(job.processes().size());
  for (auto const& process: job.processes())
    processes.emplace_back(process);
}

void
FastTimerService::ResourcesPerJob::reset() {
  total.reset();
  for (auto & module: highlight)
    module.reset();
  for (auto & module: modules)
    module.reset();
  for (auto & process: processes)
    process.reset();
  events = 0;
}

FastTimerService::ResourcesPerJob &
FastTimerService::ResourcesPerJob::operator+=(ResourcesPerJob const& other) {
  total     += other.total;
  assert(highlight.size() == other.highlight.size());
  for (unsigned int i: boost::irange(0ul, highlight.size()))
    highlight[i] += other.highlight[i];
  assert(modules.size() == other.modules.size());
  for (unsigned int i: boost::irange(0ul, modules.size()))
    modules[i] += other.modules[i];
  assert(processes.size() == other.processes.size());
  for (unsigned int i: boost::irange(0ul, processes.size()))
    processes[i] += other.processes[i];
  events += other.events;
  return *this;
}

FastTimerService::ResourcesPerJob
FastTimerService::ResourcesPerJob::operator+(ResourcesPerJob const& other) const {
  ResourcesPerJob result(*this);
  result += other;
  return result;
}


// per-thread measurements

// Measurement

FastTimerService::Measurement::Measurement()
  = default;

void
FastTimerService::Measurement::measure() {
  #ifdef DEBUG_THREAD_CONCURRENCY
  id = std::this_thread::get_id();
  #endif // DEBUG_THREAD_CONCURRENCY
  time_thread = boost::chrono::thread_clock::now();
  time_real   = boost::chrono::high_resolution_clock::now();
  allocated   = memory_usage::allocated();
  deallocated = memory_usage::deallocated();
}

void
FastTimerService::Measurement::measure_and_store(Resources & store) {
  #ifdef DEBUG_THREAD_CONCURRENCY
  assert(std::this_thread::get_id() == id);
  #endif // DEBUG_THREAD_CONCURRENCY
  auto new_time_thread = boost::chrono::thread_clock::now();
  auto new_time_real   = boost::chrono::high_resolution_clock::now();
  auto new_allocated   = memory_usage::allocated();
  auto new_deallocated = memory_usage::deallocated();
  store.time_thread = new_time_thread - time_thread;
  store.time_real   = new_time_real   - time_real;
  store.allocated   = new_allocated   - allocated;
  store.deallocated = new_deallocated - deallocated;
  time_thread = new_time_thread;
  time_real   = new_time_real;
  allocated   = new_allocated;
  deallocated = new_deallocated;
}

///////////////////////////////////////////////////////////////////////////////

FastTimerService::PlotsPerElement::PlotsPerElement() :
  time_thread_(nullptr),
  time_thread_byls_(nullptr),
  time_real_(nullptr),
  time_real_byls_(nullptr),
  allocated_(nullptr),
  allocated_byls_(nullptr),
  deallocated_(nullptr),
  deallocated_byls_(nullptr)
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
  allocated_        = nullptr;
  allocated_byls_   = nullptr;
  deallocated_      = nullptr;
  deallocated_byls_ = nullptr;
}

void
FastTimerService::PlotsPerElement::book(
    DQMStore::IBooker & booker,
    std::string const& name,
    std::string const& title,
    PlotRanges const& ranges,
    unsigned int lumisections,
    bool byls)
{
  int time_bins = (int) std::ceil(ranges.time_range   / ranges.time_resolution);
  int mem_bins  = (int) std::ceil(ranges.memory_range / ranges.memory_resolution);
  std::string y_title_ms = (boost::format("events / %.1f ms") % ranges.time_resolution).str();
  std::string y_title_kB = (boost::format("events / %.1f kB") % ranges.memory_resolution).str();

  time_thread_ = booker.book1D(
      name + " time_thread",
      title + " processing time (cpu)",
      time_bins, 0., ranges.time_range
      )->getTH1F();
  time_thread_->StatOverflows(true);
  time_thread_->SetXTitle("processing time [ms]");
  time_thread_->SetYTitle(y_title_ms.c_str());

  time_real_ = booker.book1D(
      name + " time_real",
      title + " processing time (real)",
      time_bins, 0., ranges.time_range
      )->getTH1F();
  time_real_->StatOverflows(true);
  time_real_->SetXTitle("processing time [ms]");
  time_real_->SetYTitle(y_title_ms.c_str());

  if (memory_usage::is_available())
  {
    allocated_ = booker.book1D(
        name + " allocated",
        title + " allocated memory",
        mem_bins, 0., ranges.memory_range
        )->getTH1F();
    allocated_->StatOverflows(true);
    allocated_->SetXTitle("memory [kB]");
    allocated_->SetYTitle(y_title_kB.c_str());

    deallocated_ = booker.book1D(
        name + " deallocated",
        title + " deallocated memory",
        mem_bins, 0., ranges.memory_range
        )->getTH1F();
    deallocated_->StatOverflows(true);
    deallocated_->SetXTitle("memory [kB]");
    deallocated_->SetYTitle(y_title_kB.c_str());
  }

  if (not byls)
    return;

  time_thread_byls_ = booker.bookProfile(
      name + " time_thread_byls",
      title + " processing time (cpu) vs. lumisection",
      lumisections, 0.5, lumisections + 0.5,
      time_bins, 0., std::numeric_limits<double>::infinity(),
      " ")->getTProfile();
  time_thread_byls_->StatOverflows(true);
  time_thread_byls_->SetXTitle("lumisection");
  time_thread_byls_->SetYTitle("processing time [ms]");

  time_real_byls_ = booker.bookProfile(
      name + " time_real_byls",
      title + " processing time (real) vs. lumisection",
      lumisections, 0.5, lumisections + 0.5,
      time_bins, 0., std::numeric_limits<double>::infinity(),
      " ")->getTProfile();
  time_real_byls_->StatOverflows(true);
  time_real_byls_->SetXTitle("lumisection");
  time_real_byls_->SetYTitle("processing time [ms]");

  if (memory_usage::is_available())
  {
    allocated_byls_ = booker.bookProfile(
        name + " allocated_byls",
        title + " allocated memory vs. lumisection",
        lumisections, 0.5, lumisections + 0.5,
        mem_bins, 0., std::numeric_limits<double>::infinity(),
        " ")->getTProfile();
    allocated_byls_->StatOverflows(true);
    allocated_byls_->SetXTitle("lumisection");
    allocated_byls_->SetYTitle("memory [kB]");

    deallocated_byls_ = booker.bookProfile(
        name + " deallocated_byls",
        title + " deallocated memory vs. lumisection",
        lumisections, 0.5, lumisections + 0.5,
        mem_bins, 0., std::numeric_limits<double>::infinity(),
        " ")->getTProfile();
    deallocated_byls_->StatOverflows(true);
    deallocated_byls_->SetXTitle("lumisection");
    deallocated_byls_->SetYTitle("memory [kB]");
  }
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

  if (allocated_)
    allocated_->Fill(kB(data.allocated));

  if (allocated_byls_)
    allocated_byls_->Fill(lumisection, kB(data.allocated));

  if (deallocated_)
    deallocated_->Fill(kB(data.deallocated));

  if (deallocated_byls_)
    deallocated_byls_->Fill(lumisection, kB(data.deallocated));
}

void
FastTimerService::PlotsPerElement::fill_fraction(Resources const& data, Resources const& part, unsigned int lumisection)
{
  float total;
  float fraction;

  total     = ms(data.time_thread);
  fraction  = (total > 0.) ? (ms(part.time_thread) / total) : 0.;
  if (time_thread_)
    time_thread_->Fill(total, fraction);

  if (time_thread_byls_)
    time_thread_byls_->Fill(lumisection, total, fraction);

  total     = ms(data.time_real);
  fraction  = (total > 0.) ? (ms(part.time_real) / total) : 0.;
  if (time_real_)
    time_real_->Fill(total, fraction);

  if (time_real_byls_)
    time_real_byls_->Fill(lumisection, total, fraction);

  total     = kB(data.allocated);
  fraction  = (total > 0.) ? (kB(part.allocated) / total) : 0.;
  if (allocated_)
    allocated_->Fill(total, fraction);

  if (allocated_byls_)
    allocated_byls_->Fill(lumisection, total, fraction);

  total     = kB(data.deallocated);
  fraction  = (total > 0.) ? (kB(part.deallocated) / total) : 0.;
  if (deallocated_)
    deallocated_->Fill(total, fraction);

  if (deallocated_byls_)
    deallocated_byls_->Fill(lumisection, total, fraction);
}


FastTimerService::PlotsPerPath::PlotsPerPath() :
  total_(),
  module_counter_(nullptr),
  module_time_thread_total_(nullptr),
  module_time_real_total_(nullptr),
  module_allocated_total_(nullptr),
  module_deallocated_total_(nullptr)
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
  module_allocated_total_    = nullptr;
  module_deallocated_total_  = nullptr;
}

void
FastTimerService::PlotsPerPath::book(
    DQMStore::IBooker & booker,
    std::string const & prefixDir,
    ProcessCallGraph const& job,
    ProcessCallGraph::PathType const& path,
    PlotRanges const& ranges,
    unsigned int lumisections,
    bool byls)
{
  const std::string basedir = booker.pwd();
  //  booker.setCurrentFolder(basedir + "/path " + path.name_);
  booker.setCurrentFolder(basedir + "/" + prefixDir + path.name_);

  total_.book(booker, "path", path.name_, ranges, lumisections, byls);

  unsigned int bins = path.modules_and_dependencies_.size();
  module_counter_ = booker.book1DD(
      "module_counter",
      "module counter",
      bins + 1, -0.5, bins + 0.5
      )->getTH1D();
  module_counter_->SetYTitle("events");
  module_time_thread_total_ = booker.book1DD(
      "module_time_thread_total",
      "total module time (cpu)",
      bins, -0.5, bins - 0.5
      )->getTH1D();
  module_time_thread_total_->SetYTitle("processing time [ms]");
  module_time_real_total_ = booker.book1DD(
      "module_time_real_total",
      "total module time (real)",
      bins, -0.5, bins - 0.5
      )->getTH1D();
  module_time_real_total_->SetYTitle("processing time [ms]");
  if (memory_usage::is_available())
  {
    module_allocated_total_ = booker.book1DD(
        "module_allocated_total",
        "total allocated memory",
        bins, -0.5, bins - 0.5
        )->getTH1D();
    module_allocated_total_->SetYTitle("memory [kB]");
    module_deallocated_total_ = booker.book1DD(
        "module_deallocated_total",
        "total deallocated memory",
        bins, -0.5, bins - 0.5
        )->getTH1D();
    module_deallocated_total_->SetYTitle("memory [kB]");
  }
  for (unsigned int bin: boost::irange(0u, bins)) {
    auto const& module = job[path.modules_and_dependencies_[bin]];
    std::string const& label = module.scheduled_ ? module.module_.moduleLabel() : module.module_.moduleLabel() + " (unscheduled)";
    module_counter_          ->GetXaxis()->SetBinLabel(bin + 1, label.c_str());
    module_time_thread_total_->GetXaxis()->SetBinLabel(bin + 1, label.c_str());
    module_time_real_total_  ->GetXaxis()->SetBinLabel(bin + 1, label.c_str());
    if (memory_usage::is_available())
    {
      module_allocated_total_  ->GetXaxis()->SetBinLabel(bin + 1, label.c_str());
      module_deallocated_total_->GetXaxis()->SetBinLabel(bin + 1, label.c_str());
    }
  }
  module_counter_->GetXaxis()->SetBinLabel(bins + 1, "");

  booker.setCurrentFolder(basedir);
}

void
FastTimerService::PlotsPerPath::fill(ProcessCallGraph::PathType const& description, ResourcesPerJob const& data, ResourcesPerPath const& path, unsigned int ls)
{
  // fill the total path time
  total_.fill(path.total, ls);

  // fill the modules that actually ran and the total time spent in each od them
  for (unsigned int i = 0; i < path.last; ++i) {
    auto const& module = data.modules[description.modules_and_dependencies_[i]];
    if (module_counter_)
      module_counter_->Fill(i);

    if (module_time_thread_total_)
      module_time_thread_total_->Fill(i, ms(module.total.time_thread));

    if (module_time_real_total_)
      module_time_real_total_->Fill(i, ms(module.total.time_real));

    if (module_allocated_total_)
      module_allocated_total_->Fill(i, kB(module.total.allocated));

    if (module_deallocated_total_)
      module_deallocated_total_->Fill(i, kB(module.total.deallocated));
  }
  if (module_counter_ and path.status)
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
    PlotRanges const& event_ranges,
    PlotRanges const& path_ranges,
    unsigned int lumisections,
    bool bypath,
    bool byls)
{
  const std::string basedir = booker.pwd();
  event_.book(booker,
      "process " + process.name_, "process " + process.name_,
      event_ranges,
      lumisections,
      byls);
  if (bypath) {
    booker.setCurrentFolder(basedir + "/process " + process.name_ + " paths");
    for (unsigned int id: boost::irange(0ul, paths_.size()))
    {
      paths_[id].book(booker,"path ",
          job, process.paths_[id],
          path_ranges,
          lumisections,
          byls);
    }
    for (unsigned int id: boost::irange(0ul, endpaths_.size()))
    {
      endpaths_[id].book(booker,"endpath ",
          job, process.endPaths_[id],
          path_ranges,
          lumisections,
          byls);
    }
    booker.setCurrentFolder(basedir);
  }
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

FastTimerService::PlotsPerJob::PlotsPerJob(ProcessCallGraph const& job, std::vector<GroupOfModules> const& groups) :
  event_(),
  highlight_(groups.size()),
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
  for (auto & module: highlight_)
    module.reset();
  for (auto & module: modules_)
    module.reset();
  for (auto & process: processes_)
    process.reset();
}

void
FastTimerService::PlotsPerJob::book(
    DQMStore::IBooker & booker,
    ProcessCallGraph const& job,
    std::vector<GroupOfModules> const& groups,
    PlotRanges const& event_ranges,
    PlotRanges const& path_ranges,
    PlotRanges const& module_ranges,
    unsigned int lumisections,
    bool bymodule,
    bool bypath,
    bool byls)
{
  const std::string basedir = booker.pwd();

  // event summary plots
  event_.book(booker,
      "event", "Event",
      event_ranges,
      lumisections,
      byls);

  modules_[job.source().id()].book(booker,
      "source", "Source",
      module_ranges,
      lumisections,
      byls);

  // plot the time spent in few given groups of modules
  for (unsigned int group: boost::irange(0ul, groups.size())) {
    auto const & label = groups[group].label;
    highlight_[group].book(booker,
        "highlight " + label, "Highlight " + label,
        event_ranges,
        lumisections,
        byls);
  }

  // plots per subprocess (event, modules, paths and endpaths)
  for (unsigned int pid: boost::irange(0ul, job.processes().size())) {
    auto const& process = job.processDescription(pid);
    processes_[pid].book(booker,
        job, process,
        event_ranges,
        path_ranges,
        lumisections,
        bypath,
        byls);

    if (bymodule) {
      booker.setCurrentFolder(basedir + "/process " + process.name_ + " modules");
      for (unsigned int id: process.modules_)
      {
        auto const& module_name = job.module(id).moduleLabel();
        modules_[id].book(booker,
            module_name, module_name,
            module_ranges,
            lumisections,
            byls);
      }
      booker.setCurrentFolder(basedir);
    }
  }
}

void
FastTimerService::PlotsPerJob::fill(ProcessCallGraph const& job, ResourcesPerJob const& data, unsigned int ls)
{
  // fill total event plots
  event_.fill(data.total, ls);

  // fill highltight plots
  for (unsigned int group: boost::irange(0ul, highlight_.size()))
    highlight_[group].fill_fraction(data.total, data.highlight[group], ls);

  // fill modules plots
  for (unsigned int id: boost::irange(0ul, modules_.size()))
    modules_[id].fill(data.modules[id].total, ls);

  for (unsigned int pid: boost::irange(0ul, processes_.size()))
    processes_[pid].fill(job.processDescription(pid), data, data.processes[pid], ls);
}


///////////////////////////////////////////////////////////////////////////////

FastTimerService::FastTimerService(const edm::ParameterSet & config, edm::ActivityRegistry & registry) :
  // configuration
  callgraph_(),
  // job configuration
  concurrent_runs_(             0 ),
  concurrent_streams_(          0 ),
  concurrent_threads_(          0 ),
  print_event_summary_(         config.getUntrackedParameter<bool>(     "printEventSummary"        ) ),
  print_run_summary_(           config.getUntrackedParameter<bool>(     "printRunSummary"          ) ),
  print_job_summary_(           config.getUntrackedParameter<bool>(     "printJobSummary"          ) ),
  // dqm configuration
  module_id_(                   edm::ModuleDescription::invalidID() ),
  enable_dqm_(                  config.getUntrackedParameter<bool>(     "enableDQM"                ) ),
  enable_dqm_bymodule_(         config.getUntrackedParameter<bool>(     "enableDQMbyModule"        ) ),
  enable_dqm_bypath_(           config.getUntrackedParameter<bool>(     "enableDQMbyPath"          ) ),
  enable_dqm_byls_(             config.getUntrackedParameter<bool>(     "enableDQMbyLumiSection"   ) ),
  enable_dqm_bynproc_(          config.getUntrackedParameter<bool>(     "enableDQMbyProcesses"     ) ),
  dqm_event_ranges_(          { config.getUntrackedParameter<double>(   "dqmTimeRange"             ),              // ms
                                config.getUntrackedParameter<double>(   "dqmTimeResolution"        ),              // ms
                                config.getUntrackedParameter<double>(   "dqmMemoryRange"           ),              // kB
                                config.getUntrackedParameter<double>(   "dqmMemoryResolution"      ) } ),          // kB
  dqm_path_ranges_(           { config.getUntrackedParameter<double>(   "dqmPathTimeRange"         ),              // ms
                                config.getUntrackedParameter<double>(   "dqmPathTimeResolution"    ),              // ms
                                config.getUntrackedParameter<double>(   "dqmPathMemoryRange"       ),              // kB
                                config.getUntrackedParameter<double>(   "dqmPathMemoryResolution"  ) } ),          // kB
  dqm_module_ranges_(         { config.getUntrackedParameter<double>(   "dqmModuleTimeRange"       ),              // ms
                                config.getUntrackedParameter<double>(   "dqmModuleTimeResolution"  ),              // ms
                                config.getUntrackedParameter<double>(   "dqmModuleMemoryRange"     ),              // kB
                                config.getUntrackedParameter<double>(   "dqmModuleMemoryResolution") } ),          // kB
  dqm_lumisections_range_(      config.getUntrackedParameter<unsigned int>( "dqmLumiSectionsRange" ) ),
  dqm_path_(                    config.getUntrackedParameter<std::string>("dqmPath" ) ),
  // highlight configuration
  highlight_module_psets_(      config.getUntrackedParameter<std::vector<edm::ParameterSet>>("highlightModules") ),
  highlight_modules_(           highlight_module_psets_.size())         // filled in postBeginJob()
{
  registry.watchPreallocate(                this, & FastTimerService::preallocate );
  registry.watchPreBeginJob(                this, & FastTimerService::preBeginJob );
  registry.watchPostBeginJob(               this, & FastTimerService::postBeginJob );
  registry.watchPostEndJob(                 this, & FastTimerService::postEndJob );
  registry.watchPreGlobalBeginRun(          this, & FastTimerService::preGlobalBeginRun );
//registry.watchPostGlobalBeginRun(         this, & FastTimerService::postGlobalBeginRun );
//registry.watchPreGlobalEndRun(            this, & FastTimerService::preGlobalEndRun );
  registry.watchPostGlobalEndRun(           this, & FastTimerService::postGlobalEndRun );
  registry.watchPreStreamBeginRun(          this, & FastTimerService::preStreamBeginRun );
//registry.watchPostStreamBeginRun(         this, & FastTimerService::postStreamBeginRun );
//registry.watchPreStreamEndRun(            this, & FastTimerService::preStreamEndRun );
  registry.watchPostStreamEndRun(           this, & FastTimerService::postStreamEndRun );
//registry.watchPreGlobalBeginLumi(         this, & FastTimerService::preGlobalBeginLumi );
//registry.watchPostGlobalBeginLumi(        this, & FastTimerService::postGlobalBeginLumi );
//registry.watchPreGlobalEndLumi(           this, & FastTimerService::preGlobalEndLumi );
//registry.watchPostGlobalEndLumi(          this, & FastTimerService::postGlobalEndLumi );
  registry.watchPreStreamBeginLumi(         this, & FastTimerService::preStreamBeginLumi );
//registry.watchPostStreamBeginLumi(        this, & FastTimerService::postStreamBeginLumi );
//registry.watchPreStreamEndLumi(           this, & FastTimerService::preStreamEndLumi );
  registry.watchPostStreamEndLumi(          this, & FastTimerService::postStreamEndLumi );
//registry.watchPreEvent(                   this, & FastTimerService::preEvent );
  registry.watchPostEvent(                  this, & FastTimerService::postEvent );
  registry.watchPrePathEvent(               this, & FastTimerService::prePathEvent );
  registry.watchPostPathEvent(              this, & FastTimerService::postPathEvent );
  registry.watchPreSourceConstruction(      this, & FastTimerService::preSourceConstruction);
//registry.watchPostSourceConstruction(     this, & FastTimerService::postSourceConstruction);
//registry.watchPreSourceRun(               this, & FastTimerService::preSourceRun );
//registry.watchPostSourceRun(              this, & FastTimerService::postSourceRun );
//registry.watchPreSourceLumi(              this, & FastTimerService::preSourceLumi );
//registry.watchPostSourceLumi(             this, & FastTimerService::postSourceLumi );
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
//registry.watchPreModuleGlobalBeginRun(    this, & FastTimerService::preModuleGlobalBeginRun );
//registry.watchPostModuleGlobalBeginRun(   this, & FastTimerService::postModuleGlobalBeginRun );
//registry.watchPreModuleGlobalEndRun(      this, & FastTimerService::preModuleGlobalEndRun );
//registry.watchPostModuleGlobalEndRun(     this, & FastTimerService::postModuleGlobalEndRun );
//registry.watchPreModuleGlobalBeginLumi(   this, & FastTimerService::preModuleGlobalBeginLumi );
//registry.watchPostModuleGlobalBeginLumi(  this, & FastTimerService::postModuleGlobalBeginLumi );
//registry.watchPreModuleGlobalEndLumi(     this, & FastTimerService::preModuleGlobalEndLumi );
//registry.watchPostModuleGlobalEndLumi(    this, & FastTimerService::postModuleGlobalEndLumi );
//registry.watchPreModuleStreamBeginRun(    this, & FastTimerService::preModuleStreamBeginRun );
//registry.watchPostModuleStreamBeginRun(   this, & FastTimerService::postModuleStreamBeginRun );
//registry.watchPreModuleStreamEndRun(      this, & FastTimerService::preModuleStreamEndRun );
//registry.watchPostModuleStreamEndRun(     this, & FastTimerService::postModuleStreamEndRun );
//registry.watchPreModuleStreamBeginLumi(   this, & FastTimerService::preModuleStreamBeginLumi );
//registry.watchPostModuleStreamBeginLumi(  this, & FastTimerService::postModuleStreamBeginLumi );
//registry.watchPreModuleStreamEndLumi(     this, & FastTimerService::preModuleStreamEndLumi );
//registry.watchPostModuleStreamEndLumi(    this, & FastTimerService::postModuleStreamEndLumi );
//registry.watchPreModuleEventPrefetching(  this, & FastTimerService::preModuleEventPrefetching );
//registry.watchPostModuleEventPrefetching( this, & FastTimerService::postModuleEventPrefetching );
  registry.watchPreModuleEvent(             this, & FastTimerService::preModuleEvent );
  registry.watchPostModuleEvent(            this, & FastTimerService::postModuleEvent );
  registry.watchPreModuleEventDelayedGet(   this, & FastTimerService::preModuleEventDelayedGet );
  registry.watchPostModuleEventDelayedGet(  this, & FastTimerService::postModuleEventDelayedGet );
//registry.watchPreEventReadFromSource(     this, & FastTimerService::preEventReadFromSource );
//registry.watchPostEventReadFromSource(    this, & FastTimerService::postEventReadFromSource );
}

FastTimerService::~FastTimerService() = default;

double
FastTimerService::queryModuleTime_(edm::StreamID sid, unsigned int id) const
{
  // private version, assume "id" is valid
  auto const& stream = streams_[sid];
  auto const& module = stream.modules[id];
  return ms(module.total.time_real);
}

double
FastTimerService::querySourceTime(edm::StreamID sid) const
{
  return queryModuleTime_(sid, callgraph_.source().id());
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
  return queryModuleTime_(sid, md.id());
}

double
FastTimerService::queryModuleTime(edm::StreamID sid, unsigned int id) const
{
  if (id < callgraph_.size())
    return queryModuleTime_(sid, id);

  // FIXME issue a LogWarning, raise an exception, or return NaN
  return 0.;
}

double
FastTimerService::queryModuleTimeByLabel(edm::StreamID sid, std::string const& label) const
{
  for (unsigned int id: boost::irange(0u, callgraph_.size()))
    if (callgraph_.module(id).moduleLabel() == label)
      return queryModuleTime_(sid, id);

  // FIXME issue a LogWarning, raise an exception, or return NaN
  return 0.;
}

double
FastTimerService::queryModuleTimeByLabel(edm::StreamID sid, std::string const& process, const std::string & label) const
{
  for (unsigned int id: callgraph_.processDescription(process).modules_)
    if (callgraph_.module(id).moduleLabel() == label)
      return queryModuleTime_(sid, id);

  // FIXME issue a LogWarning, raise an exception, or return NaN
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

  // FIXME issue a LogWarning, raise an exception, or return NaN
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

  // FIXME issue a LogWarning, raise an exception, or return NaN
  return 0.;
}

double
FastTimerService::queryHighlightTime(edm::StreamID sid, std::string const& label) const
{
  auto const& stream = streams_[sid];
  for (unsigned int group: boost::irange(0ul, highlight_modules_.size()))
    if (highlight_modules_[group].label == label)
      return ms(stream.highlight[group].time_real);

  // FIXME issue a LogWarning, raise an exception, or return NaN
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

  // reset the run counters only during the main process being run
  if (isFirstSubprocess(gc)) {
    subprocess_global_run_check_[gc.runIndex()] = 0;
    run_summary_[gc.runIndex()].reset();
  }
}

void
FastTimerService::postGlobalBeginRun(edm::GlobalContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::preStreamBeginRun(edm::StreamContext const& sc)
{
  ignoredSignal(__func__);

  unsigned int sid = sc.streamID().value();

  // reset counters and book the DQM plots during the main process being run
  if (isFirstSubprocess(sc)) {
    subprocess_run_check_[sid] = 0;

    // book the DQM plots
    if (enable_dqm_) {
      // define a callback to book the MonitorElements
      auto bookTransactionCallback = [&, this] (DQMStore::IBooker & booker)
      {
        booker.setCurrentFolder(dqm_path_);
        stream_plots_[sid].book(booker, callgraph_,
            highlight_modules_,
            dqm_event_ranges_,
            dqm_path_ranges_,
            dqm_module_ranges_,
            dqm_lumisections_range_,
            enable_dqm_bymodule_,
            enable_dqm_bypath_,
            enable_dqm_byls_);
      };

      // book MonitorElements for this stream
      edm::Service<DQMStore>()->bookTransaction(bookTransactionCallback, sc.eventID().run(), sid, module_id_);
    }
  }
}


void
FastTimerService::preallocate(edm::service::SystemBounds const& bounds)
{
  concurrent_runs_    = bounds.maxNumberOfConcurrentRuns();
  concurrent_streams_ = bounds.maxNumberOfStreams();
  concurrent_threads_ = bounds.maxNumberOfThreads();

  if (enable_dqm_bynproc_)
    dqm_path_ += (boost::format("/Running %d processes") % concurrent_threads_).str();

  // allocate atomic variables to keep track of the completion of each step, process by process
  subprocess_event_check_       = std::make_unique<std::atomic<unsigned int>[]>(concurrent_streams_);
  for (unsigned int i = 0; i < concurrent_streams_; ++i)
    subprocess_event_check_[i] = 0;
  subprocess_lumisection_check_ = std::make_unique<std::atomic<unsigned int>[]>(concurrent_streams_);
  for (unsigned int i = 0; i < concurrent_streams_; ++i)
    subprocess_lumisection_check_[i] = 0;
  subprocess_run_check_         = std::make_unique<std::atomic<unsigned int>[]>(concurrent_streams_);
  for (unsigned int i = 0; i < concurrent_streams_; ++i)
    subprocess_run_check_[i] = 0;
  subprocess_global_run_check_  = std::make_unique<std::atomic<unsigned int>[]>(concurrent_streams_);
  for (unsigned int i = 0; i < concurrent_runs_; ++i)
    subprocess_global_run_check_[i] = 0;

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

  // module highlights
  for (unsigned int group: boost::irange(0ul, highlight_module_psets_.size())) {
    // copy and sort for faster search via std::binary_search
    auto labels = highlight_module_psets_[group].getUntrackedParameter<std::vector<std::string>>("modules");
    std::sort(labels.begin(), labels.end());

    highlight_modules_[group].label = highlight_module_psets_[group].getUntrackedParameter<std::string>("label");
    highlight_modules_[group].modules.reserve(labels.size());
    // convert the module labels in module ids
    for (unsigned int i = 0; i < modules; ++i) {
      auto const & label = callgraph_.module(i).moduleLabel();
      if (std::binary_search(labels.begin(), labels.end(), label))
        highlight_modules_[group].modules.push_back(i);
    }
  }
  highlight_module_psets_.clear();

  // allocate the resource counters for each stream, process, path and module
  ResourcesPerJob temp(callgraph_, highlight_modules_);
  streams_.resize(concurrent_streams_, temp);
  run_summary_.resize(concurrent_runs_, temp);
  job_summary_ = temp;

  // check that the DQMStore service is available
  if (enable_dqm_ and not edm::Service<DQMStore>().isAvailable()) {
    // the DQMStore is not available, disable all DQM plots
    enable_dqm_ = false;
    // FIXME issue a LogWarning ?
  }

  // allocate the structures to hold pointers to the DQM plots
  if (enable_dqm_)
    stream_plots_.resize(concurrent_streams_, PlotsPerJob(callgraph_, highlight_modules_));

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
  ignoredSignal(__func__);

  unsigned int sid = sc.streamID().value();

  // merge plots only after the last subprocess has run
  bool last = isLastSubprocess(subprocess_run_check_[sid]);
  if (enable_dqm_ and last) {
    DQMStore & store = * edm::Service<DQMStore>();
    store.mergeAndResetMEsRunSummaryCache(sc.eventID().run(), sid, module_id_);
  }
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
FastTimerService::preStreamBeginLumi(edm::StreamContext const& sc)
{
  ignoredSignal(__func__);

  // reset counters only during the main process transition
  if (isFirstSubprocess(sc)) {
    unsigned int sid = sc.streamID().value();
    subprocess_lumisection_check_[sid] = 0;
  }
}

void
FastTimerService::postStreamBeginLumi(edm::StreamContext const& sc)
{
  ignoredSignal(__func__);
}

void
FastTimerService::preStreamEndLumi(edm::StreamContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::postStreamEndLumi(edm::StreamContext const& sc) {
  ignoredSignal(__func__);

  unsigned int sid = sc.streamID().value();

  // merge plots only after the last subprocess has run
  bool last = isLastSubprocess(subprocess_lumisection_check_[sid]);
  if (enable_dqm_ and last) {
    DQMStore & store = * edm::Service<DQMStore>();
    store.mergeAndResetMEsLuminositySummaryCache(sc.eventID().run(),sc.eventID().luminosityBlock(),sid, module_id_);
  }
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

  // handle the summaries only after the last subprocess has run
  bool last = isLastSubprocess(subprocess_global_run_check_[gc.runIndex()]);
  if (print_run_summary_ and last) {
    edm::LogVerbatim out("FastReport");
    printSummary(out, run_summary_[gc.runIndex()], "Run");
  }
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
  if (print_job_summary_) {
    edm::LogVerbatim out("FastReport");
    printSummary(out, job_summary_, "Job");
  }
}


template <typename T>
void FastTimerService::printHeader(T& out, std::string const & label) const
{
  out << "FastReport ";
  if (label.size() < 60)
    for (unsigned int i = (60-label.size()) / 2; i > 0; --i)
      out << '-';
  out << ' ' << label << " Summary ";
  if (label.size() < 60)
    for (unsigned int i = (59-label.size()) / 2; i > 0; --i)
      out << '-';
  out << '\n';
}

template <typename T>
void FastTimerService::printEventHeader(T& out, std::string const & label) const
{
  out << "FastReport       CPU time      Real time      Allocated    Deallocated  " << label << "\n";
  //      FastReport  ######.### ms  ######.### ms  +######### kB  -######### kB  ...
}

template <typename T>
void FastTimerService::printEventLine(T& out, Resources const& data, std::string const & label) const
{
  out << boost::format("FastReport  %10.3f ms  %10.3f ms  %+10d kB  %+10d kB  %s\n") 
    % ms(data.time_thread)
    % ms(data.time_real)
    % kB(data.allocated)
    % kB(data.deallocated)
    % label;
}

template <typename T>
void FastTimerService::printEvent(T& out, ResourcesPerJob const& data) const
{
  printHeader(out, "Event");
  printEventHeader(out, "Modules");
  auto const& source_d = callgraph_.source();
  auto const& source   = data.modules[source_d.id()];
  printEventLine(out, source.total, source_d.moduleLabel());
  for (unsigned int i = 0; i < callgraph_.processes().size(); ++i) {
    auto const& proc_d = callgraph_.processDescription(i);
    auto const& proc   = data.processes[i];
    printEventLine(out, proc.total, "process " + proc_d.name_);
    for (unsigned int m: proc_d.modules_) {
      auto const& module_d = callgraph_.module(m);
      auto const& module   = data.modules[m];
      printEventLine(out, module.total, "  " + module_d.moduleLabel());
    }
  }
  printEventLine(out, data.total, "total");
  out << '\n';
  printEventHeader(out, "Processes and Paths");
  printEventLine(out, source.total, source_d.moduleLabel());
  for (unsigned int i = 0; i < callgraph_.processes().size(); ++i) {
    auto const& proc_d = callgraph_.processDescription(i);
    auto const& proc   = data.processes[i];
    printEventLine(out, proc.total, "process " + proc_d.name_);
    for (unsigned int p = 0; p < proc.paths.size(); ++p) {
      auto const& name = proc_d.paths_[p].name_;
      auto const& path = proc.paths[p];
      printEventLine(out, path.active, name + " (only scheduled modules)");
      printEventLine(out, path.total,  name + " (including dependencies)");
    }
    for (unsigned int p = 0; p < proc.endpaths.size(); ++p) {
      auto const& name = proc_d.endPaths_[p].name_;
      auto const& path = proc.endpaths[p];
      printEventLine(out, path.active, name + " (only scheduled modules)");
      printEventLine(out, path.total,  name + " (including dependencies)");
    }
  }
  printEventLine(out, data.total, "total");
  out << '\n';
  for (unsigned int group: boost::irange(0ul, highlight_modules_.size())) {
    printEventHeader(out, "Highlighted modules");
    for (unsigned int m: highlight_modules_[group].modules) {
      auto const& module_d = callgraph_.module(m);
      auto const& module   = data.modules[m];
      printEventLine(out, module.total, "  " + module_d.moduleLabel());
    }
    printEventLine(out, data.highlight[group], highlight_modules_[group].label);
    out << '\n';
  }
}

template <typename T>
void FastTimerService::printSummaryHeader(T& out, std::string const& label, bool detailed) const
{
  if (detailed)
    out << "FastReport   CPU time avg.      when run  Real time avg.      when run     Alloc, avg.      when run   Dealloc. avg.      when run  " << label;
    //      FastReport  ######.### ms  ######.### ms  ######.### ms  ######.### ms  +######### kB  +######### kB  -######### kB  -######### kB  ...
  else
    out << "FastReport   CPU time avg.                Real time avg.                   Alloc, avg.                 Dealloc. avg.                " << label;
    //      FastReport  ######.### ms                 ######.### ms                 +######### kB                 -######### kB                 ...
}

template <typename T>
void FastTimerService::printSummaryLine(T& out, Resources const& data, uint64_t events, std::string const& label) const
{
  out << boost::format("FastReport  %10.3f ms                 %10.3f ms                 %+10d kB                 %+10d kB                   %s\n")
    % (events ? ms(data.time_thread) / events : 0)
    % (events ? ms(data.time_real)   / events : 0)
    % (events ? kB(data.allocated)   / events : 0)
    % (events ? kB(data.deallocated) / events : 0)
    % label;
}

template <typename T>
void FastTimerService::printSummaryLine(T& out, Resources const& data, uint64_t events, uint64_t active, std::string const& label) const
{
  out << boost::format("FastReport  %10.3f ms  %10.3f ms  %10.3f ms  %10.3f ms  %+10d kB  %+10d kB  %+10d kB  %+10d kB  %s\n")
    % (events ? ms(data.time_thread) / events : 0) % (active ? ms(data.time_thread) / active : 0)
    % (events ? ms(data.time_real)   / events : 0) % (active ? ms(data.time_real)   / active : 0)
    % (events ? kB(data.allocated)   / events : 0) % (active ? kB(data.allocated)   / active : 0)
    % (events ? kB(data.deallocated) / events : 0) % (active ? kB(data.deallocated) / active : 0)
    % label;
}

template <typename T>
void FastTimerService::printSummary(T& out, ResourcesPerJob const& data, std::string const& label) const
{
  printHeader(out, label);
  printSummaryHeader(out, "Modules", true);
  auto const& source_d = callgraph_.source();
  auto const& source   = data.modules[source_d.id()];
  printSummaryLine(out, source.total, data.events, source.events, source_d.moduleLabel());
  for (unsigned int i = 0; i < callgraph_.processes().size(); ++i) {
    auto const& proc_d = callgraph_.processDescription(i);
    auto const& proc   = data.processes[i];
    printSummaryLine(out, proc.total, data.events, "process " + proc_d.name_);
    for (unsigned int m: proc_d.modules_) {
      auto const& module_d = callgraph_.module(m);
      auto const& module   = data.modules[m];
      printSummaryLine(out, module.total, data.events, module.events, module_d.moduleLabel());
    }
  }
  printSummaryLine(out, data.total, data.events, "total");
  out << '\n';
  printSummaryHeader(out, "Processes and Paths", false);
  printSummaryLine(out, source.total, data.events, source_d.moduleLabel());
  for (unsigned int i = 0; i < callgraph_.processes().size(); ++i) {
    auto const& proc_d = callgraph_.processDescription(i);
    auto const& proc   = data.processes[i];
    printSummaryLine(out, proc.total, data.events, "process " + proc_d.name_);
    for (unsigned int p = 0; p < proc.paths.size(); ++p) {
      auto const& name = proc_d.paths_[p].name_;
      auto const& path = proc.paths[p];
      printSummaryLine(out, path.active, data.events, name + " (only scheduled modules)");
      printSummaryLine(out, path.total,  data.events, name + " (including dependencies)");
    }
    for (unsigned int p = 0; p < proc.endpaths.size(); ++p) {
      auto const& name = proc_d.endPaths_[p].name_;
      auto const& path = proc.endpaths[p];
      printSummaryLine(out, path.active, data.events, name + " (only scheduled modules)");
      printSummaryLine(out, path.total,  data.events, name + " (including dependencies)");
    }
  }
  printSummaryLine(out, data.total, data.events, "total");
  out << '\n';
  for (unsigned int group: boost::irange(0ul, highlight_modules_.size())) {
    printSummaryHeader(out, "Highlighted modules", true);
    for (unsigned int m: highlight_modules_[group].modules) {
      auto const& module_d = callgraph_.module(m);
      auto const& module   = data.modules[m];
      printSummaryLine(out, module.total, data.events, module.events, module_d.moduleLabel());
    }
    printSummaryLine(out, data.highlight[group], data.events, highlight_modules_[group].label);
    out << '\n';
  }
}

// check if this is the first process being signalled
bool
FastTimerService::isFirstSubprocess(edm::StreamContext const& sc)
{
  return (not sc.processContext()->isSubProcess());
}

bool
FastTimerService::isFirstSubprocess(edm::GlobalContext const& gc)
{
  return (not gc.processContext()->isSubProcess());
}

// check if this is the last process being signalled
bool
FastTimerService::isLastSubprocess(std::atomic<unsigned int>& check)
{
  // release-acquire semantic guarantees that all writes in this and other threads are visible
  // after this operation; full sequentially-consistent ordering is (probably) not needed.
  unsigned int old_value = check.fetch_add(1, std::memory_order_acq_rel);
  return (old_value == callgraph_.processes().size() - 1);
}

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
    data += stream.modules[i].total;
  stream.total += data;

  // handle the summaries and fill the plots only after the last subprocess has run
  bool last = isLastSubprocess(subprocess_event_check_[sid]);
  if (not last)
    return;

  // highlighted modules
  for (unsigned int group: boost::irange(0ul, highlight_modules_.size()))
    for (unsigned int i: highlight_modules_[group].modules)
      stream.highlight[group] += stream.modules[i].total;

  // avoid concurrent access to the summary objects
  {
    std::lock_guard<std::mutex> guard(summary_mutex_);
    job_summary_ += stream;
    run_summary_[sc.runIndex()] += stream;
  }

  if (print_event_summary_) {
    edm::LogVerbatim out("FastReport");
    printEvent(out, stream);
  }

  if (enable_dqm_)
    stream_plots_[sid].fill(callgraph_, stream, sc.eventID().luminosityBlock());
}

void
FastTimerService::preSourceEvent(edm::StreamID sid)
{
  // clear the event counters
  auto & stream = streams_[sid];
  stream.reset();
  ++stream.events;

  subprocess_event_check_[sid] = 0;

  thread().measure();
}


void
FastTimerService::postSourceEvent(edm::StreamID sid)
{
  edm::ModuleDescription const& md = callgraph_.source();
  unsigned int id  = md.id();
  auto & stream = streams_[sid];

  thread().measure_and_store(stream.modules[id].total);
  ++stream.modules[id].events;
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
    data.active += module.total;
  }
  for (unsigned int i = 0; i < data.last; ++i) {
    auto const& module = stream.modules[path.modules_and_dependencies_[i]];
    data.total += module.total;
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

  thread().measure_and_store(stream.modules[id].total);
  ++stream.modules[id].events;
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
  ignoredSignal(__func__);
}

void
FastTimerService::postModuleEventPrefetching(edm::StreamContext const&, edm::ModuleCallingContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::preEventReadFromSource(edm::StreamContext const&, edm::ModuleCallingContext const&)
{
  ignoredSignal(__func__);
}

void
FastTimerService::postEventReadFromSource(edm::StreamContext const&, edm::ModuleCallingContext const&)
{
  ignoredSignal(__func__);
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


FastTimerService::Measurement &
FastTimerService::thread()
{
  return threads_.local();
}


// describe the module's configuration
void
FastTimerService::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>(        "printEventSummary",        false);
  desc.addUntracked<bool>(        "printRunSummary",          true);
  desc.addUntracked<bool>(        "printJobSummary",          true);
  desc.addUntracked<bool>(        "enableDQM",                true);
  desc.addUntracked<bool>(        "enableDQMbyModule",        false);
  desc.addUntracked<bool>(        "enableDQMbyPath",          false);
  desc.addUntracked<bool>(        "enableDQMbyLumiSection",   false);
  desc.addUntracked<bool>(        "enableDQMbyProcesses",     false);
  desc.addUntracked<double>(      "dqmTimeRange",             1000. );   // ms
  desc.addUntracked<double>(      "dqmTimeResolution",           5. );   // ms
  desc.addUntracked<double>(      "dqmMemoryRange",        1000000. );   // kB
  desc.addUntracked<double>(      "dqmMemoryResolution",      5000. );   // kB
  desc.addUntracked<double>(      "dqmPathTimeRange",          100. );   // ms
  desc.addUntracked<double>(      "dqmPathTimeResolution",       0.5);   // ms
  desc.addUntracked<double>(      "dqmPathMemoryRange",    1000000. );   // kB
  desc.addUntracked<double>(      "dqmPathMemoryResolution",  5000. );   // kB
  desc.addUntracked<double>(      "dqmModuleTimeRange",         40. );   // ms
  desc.addUntracked<double>(      "dqmModuleTimeResolution",     0.2);   // ms
  desc.addUntracked<double>(      "dqmModuleMemoryRange",   100000. );   // kB
  desc.addUntracked<double>(      "dqmModuleMemoryResolution", 500. );   // kB
  desc.addUntracked<unsigned>(    "dqmLumiSectionsRange",     2500  );   // ~ 16 hours
  desc.addUntracked<std::string>( "dqmPath",                  "HLT/TimerService");

  edm::ParameterSetDescription highlightModulesDescription;
  highlightModulesDescription.addUntracked<std::vector<std::string>>("modules", {});
  highlightModulesDescription.addUntracked<std::string>("label", "producers");
  desc.addVPSetUntracked("highlightModules", highlightModulesDescription, {});

  // # OBSOLETE - these parameters are ignored, they are left only not to break old configurations
  // they will not be printed in the generated cfi.py file
  desc.addOptionalNode(edm::ParameterDescription<bool>("useRealTimeClock",         true,  false), false)->setComment("This parameter is obsolete and will be ignored.");
  desc.addOptionalNode(edm::ParameterDescription<bool>("enableTimingPaths",        true,  false), false)->setComment("This parameter is obsolete and will be ignored.");
  desc.addOptionalNode(edm::ParameterDescription<bool>("enableTimingModules",      true,  false), false)->setComment("This parameter is obsolete and will be ignored.");
  desc.addOptionalNode(edm::ParameterDescription<bool>("enableTimingExclusive",    false, false), false)->setComment("This parameter is obsolete and will be ignored.");
  desc.addOptionalNode(edm::ParameterDescription<bool>("enableTimingSummary",      false, false), false)->setComment("This parameter is obsolete and will be ignored.");
  desc.addOptionalNode(edm::ParameterDescription<bool>("skipFirstPath",            false, false), false)->setComment("This parameter is obsolete and will be ignored.");
  desc.addOptionalNode(edm::ParameterDescription<bool>("enableDQMbyPathActive",    false, false), false)->setComment("This parameter is obsolete and will be ignored.");
  desc.addOptionalNode(edm::ParameterDescription<bool>("enableDQMbyPathTotal",     true,  false), false)->setComment("This parameter is obsolete and will be ignored.");
  desc.addOptionalNode(edm::ParameterDescription<bool>("enableDQMbyPathOverhead",  false, false), false)->setComment("This parameter is obsolete and will be ignored.");
  desc.addOptionalNode(edm::ParameterDescription<bool>("enableDQMbyPathDetails",   false, false), false)->setComment("This parameter is obsolete and will be ignored.");
  desc.addOptionalNode(edm::ParameterDescription<bool>("enableDQMbyPathCounters",  true,  false), false)->setComment("This parameter is obsolete and will be ignored.");
  desc.addOptionalNode(edm::ParameterDescription<bool>("enableDQMbyPathExclusive", false, false), false)->setComment("This parameter is obsolete and will be ignored.");
  desc.addOptionalNode(edm::ParameterDescription<bool>("enableDQMbyModuleType",    false, false), false)->setComment("This parameter is obsolete and will be ignored.");
  desc.addOptionalNode(edm::ParameterDescription<bool>("enableDQMSummary",         false, false), false)->setComment("This parameter is obsolete and will be ignored.");

  descriptions.add("FastTimerService", desc);
}

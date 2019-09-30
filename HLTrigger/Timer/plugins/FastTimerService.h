#ifndef FastTimerService_h
#define FastTimerService_h

// system headers
#include <unistd.h>

// C++ headers
#include <chrono>
#include <cmath>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>

// boost headers
#include <boost/chrono.hpp>

// tbb headers
#include <tbb/concurrent_unordered_set.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/task_scheduler_observer.h>

// CMSSW headers
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/HLTPathStatus.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "HLTrigger/Timer/interface/ProcessCallGraph.h"

/*
procesing time is divided into
 - source
 - event processing, sum of the time spent in all the modules
*/

class FastTimerService : public tbb::task_scheduler_observer {
public:
  FastTimerService(const edm::ParameterSet&, edm::ActivityRegistry&);
  ~FastTimerService() override = default;

private:
  void ignoredSignal(const std::string& signal) const;
  void unsupportedSignal(const std::string& signal) const;

  // these signal pairs are not guaranteed to happen in the same thread

  void preallocate(edm::service::SystemBounds const&);

  void preBeginJob(edm::PathsAndConsumesOfModulesBase const&, edm::ProcessContext const&);
  void postBeginJob();

  void postEndJob();

  void preGlobalBeginRun(edm::GlobalContext const&);
  void postGlobalBeginRun(edm::GlobalContext const&);

  void preGlobalEndRun(edm::GlobalContext const&);
  void postGlobalEndRun(edm::GlobalContext const&);

  void preStreamBeginRun(edm::StreamContext const&);
  void postStreamBeginRun(edm::StreamContext const&);

  void preStreamEndRun(edm::StreamContext const&);
  void postStreamEndRun(edm::StreamContext const&);

  void preGlobalBeginLumi(edm::GlobalContext const&);
  void postGlobalBeginLumi(edm::GlobalContext const&);

  void preGlobalEndLumi(edm::GlobalContext const&);
  void postGlobalEndLumi(edm::GlobalContext const&);

  void preStreamBeginLumi(edm::StreamContext const&);
  void postStreamBeginLumi(edm::StreamContext const&);

  void preStreamEndLumi(edm::StreamContext const&);
  void postStreamEndLumi(edm::StreamContext const&);

  void preEvent(edm::StreamContext const&);
  void postEvent(edm::StreamContext const&);

  void prePathEvent(edm::StreamContext const&, edm::PathContext const&);
  void postPathEvent(edm::StreamContext const&, edm::PathContext const&, edm::HLTPathStatus const&);

  void preModuleEventPrefetching(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postModuleEventPrefetching(edm::StreamContext const&, edm::ModuleCallingContext const&);

  // these signal pairs are guaranteed to be called within the same thread

  //void preOpenFile(std::string const&, bool);
  //void postOpenFile(std::string const&, bool);

  //void preCloseFile(std::string const&, bool);
  //void postCloseFile(std::string const&, bool);

  void preSourceConstruction(edm::ModuleDescription const&);
  //void postSourceConstruction(edm::ModuleDescription const&);

  void preSourceRun(edm::RunIndex);
  void postSourceRun(edm::RunIndex);

  void preSourceLumi(edm::LuminosityBlockIndex);
  void postSourceLumi(edm::LuminosityBlockIndex);

  void preSourceEvent(edm::StreamID);
  void postSourceEvent(edm::StreamID);

  //void preModuleConstruction(edm::ModuleDescription const&);
  //void postModuleConstruction(edm::ModuleDescription const&);

  //void preModuleBeginJob(edm::ModuleDescription const&);
  //void postModuleBeginJob(edm::ModuleDescription const&);

  //void preModuleEndJob(edm::ModuleDescription const&);
  //void postModuleEndJob(edm::ModuleDescription const&);

  //void preModuleBeginStream(edm::StreamContext const&, edm::ModuleCallingContext const&);
  //void postModuleBeginStream(edm::StreamContext const&, edm::ModuleCallingContext const&);

  //void preModuleEndStream(edm::StreamContext const&, edm::ModuleCallingContext const&);
  //void postModuleEndStream(edm::StreamContext const&, edm::ModuleCallingContext const&);

  void preModuleGlobalBeginRun(edm::GlobalContext const&, edm::ModuleCallingContext const&);
  void postModuleGlobalBeginRun(edm::GlobalContext const&, edm::ModuleCallingContext const&);

  void preModuleGlobalEndRun(edm::GlobalContext const&, edm::ModuleCallingContext const&);
  void postModuleGlobalEndRun(edm::GlobalContext const&, edm::ModuleCallingContext const&);

  void preModuleGlobalBeginLumi(edm::GlobalContext const&, edm::ModuleCallingContext const&);
  void postModuleGlobalBeginLumi(edm::GlobalContext const&, edm::ModuleCallingContext const&);

  void preModuleGlobalEndLumi(edm::GlobalContext const&, edm::ModuleCallingContext const&);
  void postModuleGlobalEndLumi(edm::GlobalContext const&, edm::ModuleCallingContext const&);

  void preModuleStreamBeginRun(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postModuleStreamBeginRun(edm::StreamContext const&, edm::ModuleCallingContext const&);

  void preModuleStreamEndRun(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postModuleStreamEndRun(edm::StreamContext const&, edm::ModuleCallingContext const&);

  void preModuleStreamBeginLumi(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postModuleStreamBeginLumi(edm::StreamContext const&, edm::ModuleCallingContext const&);

  void preModuleStreamEndLumi(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postModuleStreamEndLumi(edm::StreamContext const&, edm::ModuleCallingContext const&);

  void preModuleEventAcquire(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postModuleEventAcquire(edm::StreamContext const&, edm::ModuleCallingContext const&);

  void preModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const&);

  void preModuleEventDelayedGet(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postModuleEventDelayedGet(edm::StreamContext const&, edm::ModuleCallingContext const&);

  void preEventReadFromSource(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postEventReadFromSource(edm::StreamContext const&, edm::ModuleCallingContext const&);

  // inherited from TBB task_scheduler_observer
  void on_scheduler_entry(bool worker) final;
  void on_scheduler_exit(bool worker) final;

public:
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // forward declarations
  struct Resources;
  struct AtomicResources;

  // per-thread measurements
  struct Measurement {
  public:
    Measurement() noexcept;
    // take per-thread measurements
    void measure() noexcept;
    // take per-thread measurements, compute the delta with respect to the previous measurement, and store them in the argument
    void measure_and_store(Resources& store) noexcept;
    // take per-thread measurements, compute the delta with respect to the previous measurement, and add them to the argument
    void measure_and_accumulate(Resources& store) noexcept;
    void measure_and_accumulate(AtomicResources& store) noexcept;

  public:
#ifdef DEBUG_THREAD_CONCURRENCY
    std::thread::id id;
#endif  // DEBUG_THREAD_CONCURRENCY
    boost::chrono::thread_clock::time_point time_thread;
    boost::chrono::high_resolution_clock::time_point time_real;
    uint64_t allocated;
    uint64_t deallocated;
  };

  // highlight a group of modules
  struct GroupOfModules {
  public:
    std::string label;
    std::vector<unsigned int> modules;
  };

  // resources being monitored by the service
  struct Resources {
  public:
    Resources();
    void reset();
    Resources& operator+=(Resources const& other);
    Resources operator+(Resources const& other) const;

  public:
    boost::chrono::nanoseconds time_thread;
    boost::chrono::nanoseconds time_real;
    uint64_t allocated;
    uint64_t deallocated;
  };

  // atomic version of Resources
  struct AtomicResources {
  public:
    AtomicResources();
    AtomicResources(AtomicResources const& other);
    void reset();

    AtomicResources& operator=(AtomicResources const& other);
    AtomicResources& operator+=(AtomicResources const& other);
    AtomicResources operator+(AtomicResources const& other) const;

  public:
    std::atomic<boost::chrono::nanoseconds::rep> time_thread;
    std::atomic<boost::chrono::nanoseconds::rep> time_real;
    std::atomic<uint64_t> allocated;
    std::atomic<uint64_t> deallocated;
  };

  struct ResourcesPerModule {
  public:
    ResourcesPerModule() noexcept;
    void reset() noexcept;
    ResourcesPerModule& operator+=(ResourcesPerModule const& other);
    ResourcesPerModule operator+(ResourcesPerModule const& other) const;

  public:
    Resources total;
    unsigned events;
    bool has_acquire;  // whether this module has an acquire() method
  };

  struct ResourcesPerPath {
  public:
    void reset();
    ResourcesPerPath& operator+=(ResourcesPerPath const& other);
    ResourcesPerPath operator+(ResourcesPerPath const& other) const;

  public:
    Resources active;  // resources used by all modules on this path
    Resources total;   // resources used by all modules on this path, and their dependencies
    unsigned last;     // one-past-the last module that ran on this path
    bool status;       // whether the path accepted or rejected the event
  };

  struct ResourcesPerProcess {
  public:
    ResourcesPerProcess(ProcessCallGraph::ProcessType const& process);
    void reset();
    ResourcesPerProcess& operator+=(ResourcesPerProcess const& other);
    ResourcesPerProcess operator+(ResourcesPerProcess const& other) const;

  public:
    Resources total;
    std::vector<ResourcesPerPath> paths;
    std::vector<ResourcesPerPath> endpaths;
  };

  struct ResourcesPerJob {
  public:
    ResourcesPerJob() = default;
    ResourcesPerJob(ProcessCallGraph const& job, std::vector<GroupOfModules> const& groups);
    void reset();
    ResourcesPerJob& operator+=(ResourcesPerJob const& other);
    ResourcesPerJob operator+(ResourcesPerJob const& other) const;

  public:
    Resources total;
    AtomicResources overhead;
    Resources event;  // total time etc. spent between preSourceEvent and postEvent
    Measurement event_measurement;
    std::vector<Resources> highlight;
    std::vector<ResourcesPerModule> modules;
    std::vector<ResourcesPerProcess> processes;
    unsigned events;
  };

  // plot ranges and resolution
  struct PlotRanges {
    double time_range;
    double time_resolution;
    double memory_range;
    double memory_resolution;
  };

  // plots associated to each module or other element (path, process, etc)
  class PlotsPerElement {
  public:
    PlotsPerElement() = default;
    void book(dqm::reco::DQMStore::IBooker&,
              std::string const& name,
              std::string const& title,
              PlotRanges const& ranges,
              unsigned int lumisections,
              bool byls);
    void fill(Resources const&, unsigned int lumisection);
    void fill(AtomicResources const&, unsigned int lumisection);
    void fill_fraction(Resources const&, Resources const&, unsigned int lumisection);

  private:
    // resources spent in the module
    dqm::reco::MonitorElement* time_thread_;       // TH1F
    dqm::reco::MonitorElement* time_thread_byls_;  // TProfile
    dqm::reco::MonitorElement* time_real_;         // TH1F
    dqm::reco::MonitorElement* time_real_byls_;    // TProfile
    dqm::reco::MonitorElement* allocated_;         // TH1F
    dqm::reco::MonitorElement* allocated_byls_;    // TProfile
    dqm::reco::MonitorElement* deallocated_;       // TH1F
    dqm::reco::MonitorElement* deallocated_byls_;  // TProfile
  };

  // plots associated to each path or endpath
  class PlotsPerPath {
  public:
    PlotsPerPath() = default;
    void book(dqm::reco::DQMStore::IBooker&,
              std::string const&,
              ProcessCallGraph const&,
              ProcessCallGraph::PathType const&,
              PlotRanges const& ranges,
              unsigned int lumisections,
              bool byls);
    void fill(ProcessCallGraph::PathType const&,
              ResourcesPerJob const&,
              ResourcesPerPath const&,
              unsigned int lumisection);

  private:
    // resources spent in all the modules in the path, including their dependencies
    PlotsPerElement total_;

    // Note:
    //   a TH1F has 7 significant digits, while a 24-hour long run could process
    //   order of 10 billion events; a 64-bit long integer would work and might
    //   be better suited than a double, but there is no "TH1L" in ROOT.

    // how many times each module and their dependencies has run
    dqm::reco::MonitorElement* module_counter_;  // TH1D
    // resources spent in each module and their dependencies
    dqm::reco::MonitorElement* module_time_thread_total_;  // TH1D
    dqm::reco::MonitorElement* module_time_real_total_;    // TH1D
    dqm::reco::MonitorElement* module_allocated_total_;    // TH1D
    dqm::reco::MonitorElement* module_deallocated_total_;  // TH1D
  };

  class PlotsPerProcess {
  public:
    PlotsPerProcess(ProcessCallGraph::ProcessType const&);
    void book(dqm::reco::DQMStore::IBooker&,
              ProcessCallGraph const&,
              ProcessCallGraph::ProcessType const&,
              PlotRanges const& event_ranges,
              PlotRanges const& path_ranges,
              unsigned int lumisections,
              bool bypath,
              bool byls);
    void fill(ProcessCallGraph::ProcessType const&, ResourcesPerJob const&, ResourcesPerProcess const&, unsigned int ls);

  private:
    // resources spent in all the modules of the (sub)process
    PlotsPerElement event_;
    // resources spent in each path and endpath
    std::vector<PlotsPerPath> paths_;
    std::vector<PlotsPerPath> endpaths_;
  };

  class PlotsPerJob {
  public:
    PlotsPerJob(ProcessCallGraph const& job, std::vector<GroupOfModules> const& groups);
    void book(dqm::reco::DQMStore::IBooker&,
              ProcessCallGraph const&,
              std::vector<GroupOfModules> const&,
              PlotRanges const& event_ranges,
              PlotRanges const& path_ranges,
              PlotRanges const& module_ranges,
              unsigned int lumisections,
              bool bymodule,
              bool bypath,
              bool byls,
              bool transitions);
    void fill(ProcessCallGraph const&, ResourcesPerJob const&, unsigned int ls);
    void fill_run(AtomicResources const&);
    void fill_lumi(AtomicResources const&, unsigned int lumisection);

  private:
    // resources spent in all the modules of the job
    PlotsPerElement event_;
    PlotsPerElement event_ex_;
    PlotsPerElement overhead_;
    // resources spent in the modules' lumi and run transitions
    PlotsPerElement lumi_;
    PlotsPerElement run_;
    // resources spent in the highlighted modules
    std::vector<PlotsPerElement> highlight_;
    // resources spent in each module
    std::vector<PlotsPerElement> modules_;
    // resources spent in each (sub)process
    std::vector<PlotsPerProcess> processes_;
  };

  // keep track of the dependencies among modules
  ProcessCallGraph callgraph_;

  // per-stream information
  std::vector<ResourcesPerJob> streams_;

  // concurrent histograms and profiles
  std::unique_ptr<PlotsPerJob> plots_;

  // per-lumi and per-run information
  std::vector<AtomicResources> lumi_transition_;  // resources spent in the modules' global and stream lumi transitions
  std::vector<AtomicResources> run_transition_;   // resources spent in the modules' global and stream run transitions
  AtomicResources overhead_;                      // resources spent outside of the modules' transitions

  // summary data
  ResourcesPerJob job_summary_;               // whole event time accounting per-job
  std::vector<ResourcesPerJob> run_summary_;  // whole event time accounting per-run
  std::mutex summary_mutex_;                  // synchronise access to the summary objects across different threads

  // per-thread quantities, lazily allocated
  tbb::enumerable_thread_specific<Measurement, tbb::cache_aligned_allocator<Measurement>, tbb::ets_key_per_instance>
      threads_;

  // atomic variables to keep track of the completion of each step, process by process
  std::unique_ptr<std::atomic<unsigned int>[]> subprocess_event_check_;
  std::unique_ptr<std::atomic<unsigned int>[]> subprocess_global_lumi_check_;
  std::unique_ptr<std::atomic<unsigned int>[]> subprocess_global_run_check_;

  // retrieve the current thread's per-thread quantities
  Measurement& thread();

  // job configuration
  unsigned int concurrent_lumis_;
  unsigned int concurrent_runs_;
  unsigned int concurrent_streams_;
  unsigned int concurrent_threads_;

  // logging configuration
  const bool print_event_summary_;  // print the time spent in each process, path and module after every event
  const bool print_run_summary_;    // print the time spent in each process, path and module for each run
  const bool print_job_summary_;    // print the time spent in each process, path and module for the whole job

  // dqm configuration
  bool enable_dqm_;  // non const, depends on the availability of the DQMStore
  const bool enable_dqm_bymodule_;
  const bool enable_dqm_bypath_;
  const bool enable_dqm_byls_;
  const bool enable_dqm_bynproc_;
  const bool enable_dqm_transitions_;

  const PlotRanges dqm_event_ranges_;
  const PlotRanges dqm_path_ranges_;
  const PlotRanges dqm_module_ranges_;
  const unsigned int dqm_lumisections_range_;
  std::string dqm_path_;

  std::vector<edm::ParameterSet> highlight_module_psets_;  // non-const, cleared in postBeginJob()
  std::vector<GroupOfModules> highlight_modules_;          // non-const, filled in postBeginJob()

  // log unsupported signals
  mutable tbb::concurrent_unordered_set<std::string> unsupported_signals_;  // keep track of unsupported signals received

  // print the resource usage summary for en event, a run, or the while job
  template <typename T>
  void printHeader(T& out, std::string const& label) const;

  template <typename T>
  void printEventHeader(T& out, std::string const& label) const;

  template <typename T>
  void printEventLine(T& out, Resources const& data, std::string const& label) const;

  template <typename T>
  void printEventLine(T& out, AtomicResources const& data, std::string const& label) const;

  template <typename T>
  void printEvent(T& out, ResourcesPerJob const&) const;

  template <typename T>
  void printSummaryHeader(T& out, std::string const& label, bool detailed) const;

  template <typename T>
  void printPathSummaryHeader(T& out, std::string const& label) const;

  template <typename T>
  void printSummaryLine(T& out, Resources const& data, uint64_t events, std::string const& label) const;

  template <typename T>
  void printSummaryLine(T& out, Resources const& data, uint64_t events, uint64_t active, std::string const& label) const;

  template <typename T>
  void printPathSummaryLine(
      T& out, Resources const& data, Resources const& total, uint64_t events, std::string const& label) const;

  template <typename T>
  void printSummary(T& out, ResourcesPerJob const& data, std::string const& label) const;

  template <typename T>
  void printTransition(T& out, AtomicResources const& data, std::string const& label) const;

  // check if this is the first process being signalled
  bool isFirstSubprocess(edm::StreamContext const&);
  bool isFirstSubprocess(edm::GlobalContext const&);

  // check if this is the lest process being signalled
  bool isLastSubprocess(std::atomic<unsigned int>& check);
};

#endif  // ! FastTimerService_h

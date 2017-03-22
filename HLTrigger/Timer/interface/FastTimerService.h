#ifndef FastTimerService_h
#define FastTimerService_h

// C++ headers
#include <cmath>
#include <string>
#include <map>
#include <unordered_map>
#include <chrono>
#include <mutex>
#include <unistd.h>

// boost headers
#include <boost/chrono.hpp>

// tbb headers
#include <tbb/concurrent_unordered_set.h>

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
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/HLTPathStatus.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "HLTrigger/Timer/interface/FastTimer.h"
#include "HLTrigger/Timer/interface/ProcessCallGraph.h"


/*
procesing time is diveded into
 - source
 - pre-event processing overhead
 - event processing
 - post-event processing overhead

until lumi-processing and run-processing are taken into account, they will count as inter-event overhead

event processing time is diveded into
 - trigger processing (from the begin of the first path to the end of the last path)
 - trigger overhead
 - endpath processing (from the begin of the first endpath to the end of the last endpath)
 - endpath overhead
*/

/*
Assuming an HLT process with ~2500 modules and ~500 paths, tracking each step (with two calls per step, to start and stop the timer)
with std::chrono::high_resolution_clock gives a per-event overhead of 1 ms

Detailed informations on different timers can be extracted running $CMSSW_RELEASE_BASE/test/$SCRAM_ARCH/testChrono .


Timer per-call overhead on SLC5:

Linux 2.6.18-371.1.2.el5 x86_64
glibc version: 2.5
clock source: unknown
For each timer the resolution reported is the MINIMUM (MEDIAN) (MEAN +/- its STDDEV) of the increments measured during the test.

Performance of std::chrono::high_resolution_clock
        Average time per call:      317.0 ns
        Clock tick period:            1.0 ns
        Measured resolution:       1000.0 ns (median: 1000.0 ns) (sigma: 199.4 ns) (average: 1007.6 +/- 0.4 ns)


Timer per-call overhead on SLC6 (virtualized):

Linux 2.6.32-358.23.2.el6.x86_64 x86_64
glibc version: 2.12
clock source: kvm-clock
For each timer the resolution reported is the MINIMUM (MEDIAN) (MEAN +/- its STDDEV) of the increments measured during the test.

Performance of std::chrono::high_resolution_clock
        Average time per call:      351.2 ns
        Clock tick period:            1.0 ns
        Measured resolution:          1.0 ns (median: 358.0 ns) (sigma: 30360.8 ns) (average: 685.7 +/- 42.4 ns)
*/


class FastTimerService {
public:
  FastTimerService(const edm::ParameterSet &, edm::ActivityRegistry & );
  ~FastTimerService();

private:
  void unsupportedSignal(std::string signal) const;

  // these signal pairs are not guaranteed to happen in the same thread

  void preallocate(edm::service::SystemBounds const &);

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

  // these signal pairs are guaranteed to be called within the same thread

  //void preOpenFile(std::string const&, bool);
  //void postOpenFile(std::string const&, bool);

  //void preCloseFile(std::string const&, bool);
  //void postCloseFile(std::string const&, bool);

  void preSourceConstruction(edm::ModuleDescription const&);
  //void postSourceConstruction(edm::ModuleDescription const&);

  void preSourceRun();
  void postSourceRun();

  void preSourceLumi();
  void postSourceLumi();

  void preSourceEvent(edm::StreamID);
  void postSourceEvent(edm::StreamID);

  //void preModuleConstruction(edm::ModuleDescription const&);
  //void postModuleConstruction(edm::ModuleDescription const&);

  void preModuleBeginJob(edm::ModuleDescription const&);
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

  void preModuleEventPrefetching(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postModuleEventPrefetching(edm::StreamContext const&, edm::ModuleCallingContext const&);

  void preModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const&);

  void preModuleEventDelayedGet(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postModuleEventDelayedGet(edm::StreamContext const&, edm::ModuleCallingContext const&);

  void preEventReadFromSource(edm::StreamContext const&, edm::ModuleCallingContext const&);
  void postEventReadFromSource(edm::StreamContext const&, edm::ModuleCallingContext const&);

public:
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  // keep track of the dependencies among modules
  ProcessCallGraph m_callgraph;


  // resources being monitored by the service
  struct Resources {
    boost::chrono::nanoseconds time_thread;
    boost::chrono::nanoseconds time_real;

    Resources() :
      time_thread(boost::chrono::nanoseconds::zero()),
      time_real(boost::chrono::nanoseconds::zero())
    { }

    void reset() {
      time_thread = boost::chrono::nanoseconds::zero();
      time_real   = boost::chrono::nanoseconds::zero();
    }

    Resources & operator+=(Resources const & other) {
      time_thread += other.time_thread;
      time_real   += other.time_real;
      return *this;
    }

    Resources operator+(Resources const & other) const {
      Resources result;
      result.time_thread = time_thread + other.time_thread;
      result.time_real   = time_real   + other.time_real;
      return result;
    }
  };


  struct ResourcesPerPath {
    Resources active;
    Resources total;

    void reset() {
      active.reset();
      total.reset();
    }
  };

  struct ResourcesPerProcess {
    Resources                     total;
    std::vector<ResourcesPerPath> paths;
    std::vector<ResourcesPerPath> endpaths;

    void reset() {
      total.reset();
      for (auto & path: paths)
        path.reset();
      for (auto & path: endpaths)
        path.reset();
    }
  };

  struct ResourcesPerStream {
    Resources                        total;
    std::vector<Resources>           modules;
    std::vector<ResourcesPerProcess> processes;

    void reset() {
      total.reset();
      for (auto & module: modules)
        module.reset();
      for (auto & process: processes)
        process.reset();
    }
  };

  std::vector<ResourcesPerStream> streams_;


  // per-thread measurements
  struct Measurement {
    boost::chrono::thread_clock::time_point          time_thread;
    boost::chrono::high_resolution_clock::time_point time_real;

    void measure() {
      time_thread = boost::chrono::thread_clock::now();
      time_real   = boost::chrono::high_resolution_clock::now();
    }

    void measure(Resources & store) {
      auto new_time_thread = boost::chrono::thread_clock::now();
      auto new_time_real   = boost::chrono::high_resolution_clock::now();
      store.time_thread = new_time_thread - time_thread;
      store.time_real   = new_time_real   - time_real;
      time_thread = new_time_thread;
      time_real   = new_time_real;
    }
  };

  // per-thread quantities, indexed by a thread_local id
  std::vector<Measurement> threads_;

  // define a unique id per thread
  static unsigned int threadId();

  // retrieve the current thread's per-thread quantities
  Measurement & thread();



  // plots associated to each module
  struct PlotsPerModule {
    TH1F * active;          // time spent in the module
  };

  struct PlotsPerPath {
    TH1F * active;          // time spent in all the modules in the path
    TH1F * total;           // time spent in all the modules in the path, and their dependencies
    TH1F * module_counter;  // for each module in the path, track how many times it ran
    TH1F * module_active;   // for each module in the path, track the active time spent
    TH1F * module_total;    // for each module in the path, track the total time spent
  };

  struct PlotsPerProcess {
    TH1F * total;                               // time spent in all the modules of the (sub)process
    std::vector<PlotsPerPath>    paths;
    std::vector<PlotsPerPath>    endpaths;
  };

  struct PlotsPerStream {
    TH1F * total;                               // time spent in all the modules of the job
    std::vector<PlotsPerModule>  modules;
    std::vector<PlotsPerProcess> processes;
  };

  std::vector<PlotsPerStream> stream_plots_;











  /*

private:

  struct PathInfo;

  struct ModuleInfo {
    FastTimer                   timer;              // per-event timer
    double                      time_active;        // time actually spent in this module
    double                      summary_active;
    TH1F *                      dqm_active;
    PathInfo *                  run_in_path;        // the path inside which the module was atually active
    uint32_t                    counter;            // count how many times the module was scheduled to run

  public:
    ModuleInfo() :
      timer(),
      time_active(0.),
      summary_active(0.),
      dqm_active(nullptr),
      run_in_path(nullptr),
      counter(0)
    { }

    ~ModuleInfo() {
      reset();
    }

    // reset the timers and DQM plots
    void reset() {
      timer.reset();
      time_active = 0.;
      summary_active = 0.;
      // the DQM plots are owned by the DQMStore
      dqm_active = nullptr;
      run_in_path = nullptr;
      counter = 0;
    }
  };

  struct PathInfo {
    std::vector<ModuleInfo *>   modules;            // list of all modules contributing to the path (duplicate modules stored as null pointers)
    std::vector<unsigned int>   dependencies;       // list of all modules in the path and their dependencies
    std::vector<unsigned int>   last_dependencies;  // last entry in the vector of `dependencies' for each module in the path
    ModuleInfo *                first_module;       // first module actually run in this path
    FastTimer                   timer;              // per-event timer
    double                      time_active;        // time actually spent in this path
    double                      time_exclusive;     // time actually spent in this path, in modules that are not run on any other paths
    double                      time_overhead;      // time spente before, between or after modules
    double                      time_total;         // sum of the time spent in all modules which would have run in this path (plus overhead)
    double                      summary_active;
    double                      summary_overhead;
    double                      summary_total;
    uint32_t                    last_run;           // index of the last module run in this path, plus one
    uint32_t                    index;              // index of the Path or EndPath in the "schedule"
    bool                        accept;             // flag indicating if the path acepted the event
    TH1F *                      dqm_active;         // see time_active
    TH1F *                      dqm_exclusive;      // see time_exclusive
    TH1F *                      dqm_overhead;       // see time_overhead
    TH1F *                      dqm_total;          // see time_total
    TH1F *                      dqm_module_counter; // for each module in the path, track how many times it ran
    TH1F *                      dqm_module_active;  // for each module in the path, track the active time spent
    TH1F *                      dqm_module_total;   // for each module in the path, track the total time spent

  public:
    PathInfo() :
      modules(),
      dependencies(),
      last_dependencies(),
      first_module(nullptr),
      timer(),
      time_active(0.),
      time_exclusive(0.),
      time_overhead(0.),
      time_total(0.),
      summary_active(0.),
      summary_overhead(0.),
      summary_total(0.),
      last_run(0),
      index(0),
      accept(false),
      dqm_active(nullptr),
      dqm_exclusive(nullptr),
      dqm_overhead(nullptr),
      dqm_total(nullptr),
      dqm_module_counter(nullptr),
      dqm_module_active(nullptr),
      dqm_module_total(nullptr)
    { }

    ~PathInfo() {
      reset();
    }

    // reset the timers and DQM plots
    void reset() {
      first_module = nullptr;
      timer.reset();
      time_active = 0.;
      time_exclusive = 0.;
      time_overhead = 0.;
      time_total = 0.;
      summary_active = 0.;
      summary_overhead = 0.;
      summary_total = 0.;
      last_run = 0;
      index = 0;
      accept = false;

      // the DQM plots are owned by the DQMStore
      dqm_active = nullptr;
      dqm_overhead = nullptr;
      dqm_total = nullptr;
      dqm_module_counter = nullptr;
      dqm_module_active = nullptr;
      dqm_module_total = nullptr;
    }
  };

  // the vector is indexed by the peudo-process id, the map by the paths name
  template <typename T>
  using PathMap = std::vector<std::unordered_map<std::string, T>>;

  // key on ModuleDescription::id()
  template <typename T>
  using ModuleMap = std::vector<T>;

  */

  // timer configuration
  bool                                          m_use_realtime;
  bool                                          m_enable_timing_paths;
  bool                                          m_enable_timing_modules;
  bool                                          m_enable_timing_exclusive;
  const bool                                    m_enable_timing_summary;

  // dqm configuration
  bool                                          m_enable_dqm;                   // non const because the availability of the DQMStore can only be checked during the begin job
  const bool                                    m_enable_dqm_bypath_active;     // require per-path timers
  const bool                                    m_enable_dqm_bypath_total;      // require per-path and per-module timers
  const bool                                    m_enable_dqm_bypath_overhead;   // require per-path and per-module timers
  const bool                                    m_enable_dqm_bypath_details;    // require per-path and per-module timers
  const bool                                    m_enable_dqm_bypath_counters;
  const bool                                    m_enable_dqm_bypath_exclusive;
  const bool                                    m_enable_dqm_bymodule;          // require per-module timers
  const bool                                    m_enable_dqm_summary;
  const bool                                    m_enable_dqm_byls;
  const bool                                    m_enable_dqm_bynproc;

  unsigned int                                  m_concurrent_runs;
  unsigned int                                  m_concurrent_streams;
  unsigned int                                  m_concurrent_threads;
  unsigned int                                  module_id_;                     // pseudo module id for the FastTimerService, needed by the thread-safe DQMStore

  const double                                  m_dqm_eventtime_range;
  const double                                  m_dqm_eventtime_resolution;
  const double                                  m_dqm_pathtime_range;
  const double                                  m_dqm_pathtime_resolution;
  const double                                  m_dqm_moduletime_range;
  const double                                  m_dqm_moduletime_resolution;
  const uint32_t                                m_dqm_lumisections_range;
  std::string                                   m_dqm_path;

  /*
  struct ProcessDescription {
    std::string         name;
    std::string         first_path;             // the framework does not provide a pre/postPaths or pre/postEndPaths signal,
    std::string         last_path;              // so we emulate them keeping track of the first and last non-empty Path and EndPath
    std::string         first_endpath;
    std::string         last_endpath;
  };

  struct Timing {
    double              presource;              // time spent between the end of the previous Event, LumiSection or Run, and the beginning of the Source
    double              source;                 // time spent processing the Source
    double              preevent;               // time spent between the end of the Source and the new Event, LumiSection or Run
    double              event;                  // time spent processing the Event
    unsigned int        count;                  // number of processed events (used by the per-run and per-job accounting)

    Timing() :
      presource     (0.),
      source        (0.),
      preevent      (0.),
      event         (0.),
      count         (0)
    { }

    void reset() {
      presource     = 0.;
      source        = 0.;
      preevent      = 0.;
      event         = 0.;
      count         = 0;
    }

    Timing & operator+=(Timing const & other) {
      presource     += other.presource;
      source        += other.source;
      preevent      += other.preevent;
      event         += other.event;
      count         += other.count;

      return *this;
    }

    Timing operator+(Timing const & other) const {
      Timing result = *this;
      result += other;
      return result;
    }

  };

  struct TimingPerProcess {
    double preevent;                // time spent between the end of the Source and the new Event, Lumisection or Run
    double event;                   // time spent processing the Event
    double all_paths;               // time spent processing all Paths
    double all_endpaths;            // time spent processing all EndPaths
    double interpaths;              // time spent between the Paths (and EndPaths - i.e. the sum of all the entries in the following vector)

    TimingPerProcess() :
      preevent      (0.),
      event         (0.),
      all_paths     (0.),
      all_endpaths  (0.),
      interpaths    (0.)
    { }

    void reset() {
      preevent      = 0.;
      event         = 0.;
      all_paths     = 0.;
      all_endpaths  = 0.;
      interpaths    = 0.;
    }

    TimingPerProcess & operator+=(TimingPerProcess const & other) {
      preevent      += other.preevent;
      event         += other.event;
      all_paths     += other.all_paths;
      all_endpaths  += other.all_endpaths;
      interpaths    += other.interpaths;
      return *this;
    }

    TimingPerProcess operator+(TimingPerProcess const & other) const {
      TimingPerProcess result = *this;
      result += other;
      return result;
    }

  };

  // set of summary plots, over all subprocesses
  struct SummaryPlots {
    TH1F *     presource;
    TH1F *     source;
    TH1F *     preevent;
    TH1F *     event;

    SummaryPlots() :
      presource     (nullptr),
      source        (nullptr),
      preevent      (nullptr),
      event         (nullptr)
    { }

    void reset() {
      // the DQM plots are owned by the DQMStore
      presource     = nullptr;
      source        = nullptr;
      preevent      = nullptr;
      event         = nullptr;
    }

    void fill(Timing const & value) {
      // convert on the fly from seconds to ms
      presource     ->Fill( 1000. * value.presource );
      source        ->Fill( 1000. * value.source );
      preevent      ->Fill( 1000. * value.preevent );
      event         ->Fill( 1000. * value.event );
    }

  };

  // set of summary plots, per subprocess
  struct SummaryPlotsPerProcess {
    TH1F *     preevent;
    TH1F *     event;
    TH1F *     all_paths;
    TH1F *     all_endpaths;
    TH1F *     interpaths;

    SummaryPlotsPerProcess() :
      preevent      (nullptr),
      event         (nullptr),
      all_paths     (nullptr),
      all_endpaths  (nullptr),
      interpaths    (nullptr)
    { }

    void reset() {
      // the DQM plots are owned by the DQMStore
      preevent      = nullptr;
      event         = nullptr;
      all_paths     = nullptr;
      all_endpaths  = nullptr;
      interpaths    = nullptr;
    }

    void fill(TimingPerProcess const & value) {
      // convert on the fly from seconds to ms
      preevent      ->Fill( 1000. * value.preevent );
      event         ->Fill( 1000. * value.event );
      all_paths     ->Fill( 1000. * value.all_paths );
      all_endpaths  ->Fill( 1000. * value.all_endpaths );
      interpaths    ->Fill( 1000. * value.interpaths );
    }

  };

  // set of summary profiles vs. luminosity, over all subprocesses
  struct SummaryProfiles {
    TProfile * presource;
    TProfile * source;
    TProfile * preevent;
    TProfile * event;

    SummaryProfiles() :
      presource     (nullptr),
      source        (nullptr),
      preevent      (nullptr),
      event         (nullptr)
    { }

    void reset() {
      // the DQM plots are owned by the DQMStore
      presource     = nullptr;
      source        = nullptr;
      preevent      = nullptr;
      event         = nullptr;
    }

    void fill(double x, Timing const & value) {
      presource     ->Fill( x, 1000. * value.presource );
      source        ->Fill( x, 1000. * value.source );
      preevent      ->Fill( x, 1000. * value.preevent );
      event         ->Fill( x, 1000. * value.event );
    }

  };

  // set of summary profiles vs. luminosity, per subprocess
  struct SummaryProfilesPerProcess {
    TProfile * preevent;
    TProfile * event;
    TProfile * all_paths;
    TProfile * all_endpaths;
    TProfile * interpaths;

    SummaryProfilesPerProcess() :
      preevent      (nullptr),
      event         (nullptr),
      all_paths     (nullptr),
      all_endpaths  (nullptr),
      interpaths    (nullptr)
    { }

    ~SummaryProfilesPerProcess() {
      reset();
    }

    void reset() {
      // the DQM plots are owned by the DQMStore
      preevent      = nullptr;
      event         = nullptr;
      all_paths     = nullptr;
      all_endpaths  = nullptr;
      interpaths    = nullptr;
    }

    void fill(double x, TimingPerProcess const & value) {
      preevent      ->Fill( x, 1000. * value.preevent );
      event         ->Fill( x, 1000. * value.event );
      all_paths     ->Fill( x, 1000. * value.all_paths );
      all_endpaths  ->Fill( x, 1000. * value.all_endpaths );
      interpaths    ->Fill( x, 1000. * value.interpaths );
    }

  };

  // set of profile plots by path, per subprocess
  struct PathProfilesPerProcess {
    TProfile * active_time;
    TProfile * total_time;
    TProfile * exclusive_time;

    PathProfilesPerProcess() :
      active_time   (nullptr),
      total_time    (nullptr),
      exclusive_time(nullptr)
    {}

    ~PathProfilesPerProcess() {
      reset();
    }

    void reset() {
      // the DQM plots are owned by the DQMStore
      active_time    = nullptr;
      total_time     = nullptr;
      exclusive_time = nullptr;
    }

  };

  struct StreamData {
    // timers
    FastTimer                                       timer_event;                // track time spent in each event
    FastTimer                                       timer_source;               // track time spent in the source
    FastTimer                                       timer_paths;                // track time spent in all paths
    FastTimer                                       timer_endpaths;             // track time spent in all endpaths
    FastTimer                                       timer_path;                 // track time spent in each path
    FastTimer::Clock::time_point                    timer_last_path;            // record the stop of the last path run
    FastTimer::Clock::time_point                    timer_last_transition;      // record the last transition (end source, end event, end lumi, end run)

    // time accounting per-event
    Timing                                          timing;
    std::vector<TimingPerProcess>                   timing_perprocess;

    // overall plots
    SummaryPlots                                    dqm;                        // whole event summary plots
    SummaryProfiles                                 dqm_byls;                   // whole event plots vs. "luminosity"
    std::vector<SummaryPlotsPerProcess>             dqm_perprocess;             // per-process event summary plots
    std::vector<SummaryProfilesPerProcess>          dqm_perprocess_byls;        // per-process plots vs. "luminosity"

    // plots by path
    std::vector<PathProfilesPerProcess>             dqm_paths;

    // per-path, per-module and per-module-type accounting
    PathInfo *                                      current_path;
    ModuleInfo *                                    current_module;
    PathMap<PathInfo>                               paths;
    std::unordered_map<std::string, ModuleInfo>     modules;
    ModuleMap<ModuleInfo *>                         fast_modules;

    StreamData() :
      // timers
      timer_event(),
      timer_source(),
      timer_paths(),
      timer_endpaths(),
      timer_path(),
      timer_last_path(),
      timer_last_transition(),
      // time accounting per-event
      timing(),
      timing_perprocess(),
      // overall plots
      dqm(),
      dqm_byls(),
      dqm_perprocess(),
      dqm_perprocess_byls(),
      // plots by path
      dqm_paths(),
      // per-path, per-module and per-module-type accounting
      current_path(nullptr),
      current_module(nullptr),
      paths(),
      modules(),
      fast_modules()
    { }

    // called in FastTimerService::postStreamEndRun()
    void reset() {
      // timers
      timer_event.reset();
      timer_source.reset();
      timer_paths.reset();
      timer_endpaths.reset();
      timer_path.reset();
      timer_last_path = FastTimer::Clock::time_point();
      timer_last_transition = FastTimer::Clock::time_point();
      // time accounting per-event
      timing.reset();
      for (auto & timing: timing_perprocess)
        timing.reset();
      // overall plots
      dqm.reset();
      dqm_byls.reset();
      for (auto & plots: dqm_perprocess)
        plots.reset();
      for (auto & plots: dqm_perprocess_byls)
        plots.reset();
      // plots by path
      for (auto & plots: dqm_paths)
        plots.reset();
      // per-path, per-module and per-module-type accounting
      current_path              = nullptr;
      current_module            = nullptr;
      for (auto & map: paths)
        for (auto & keyval: map)
          keyval.second.reset();
      for (auto & keyval: modules)
        keyval.second.reset();
    }

  };

  // process descriptions
  std::vector<ProcessDescription>               m_process;

  // stream data
  std::vector<StreamData>                       m_stream;

  // summary data
  std::vector<Timing>                           m_run_summary;                  // whole event time accounting per-run
  Timing                                        m_job_summary;                  // whole event time accounting per-run
  std::vector<std::vector<TimingPerProcess>>    m_run_summary_perprocess;       // per-process time accounting per-job
  std::vector<TimingPerProcess>                 m_job_summary_perprocess;       // per-process time accounting per-job
  std::mutex                                    m_summary_mutex;                // synchronise access to the summary objects across different threads

  */

  // log unsupported signals
  mutable tbb::concurrent_unordered_set<std::string> m_unsupported_signals;     // keep track of unsupported signals received

  /*
  static
  double delta(FastTimer::Clock::time_point const & first, FastTimer::Clock::time_point const & second)
  {
    return std::chrono::duration_cast<std::chrono::duration<double>>(second - first).count();
  }

  // associate to a path all the modules it contains
  void fillPathMap(unsigned int pid, std::string const & name, std::vector<std::string> const & modules);

  // find the first and last non-empty paths, optionally skipping the first one
  std::pair<std::string,std::string> findFirstLast(unsigned int pid, std::vector<std::string> const & paths);

  // print a timing summary for the run or job
  void printSummary(Timing const & summary, std::string const & label) const;
  void printProcessSummary(Timing const & total, TimingPerProcess const & summary, std::string const & label, std::string const & process) const;
  */
};

#endif // ! FastTimerService_h

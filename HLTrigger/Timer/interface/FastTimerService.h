#ifndef FastTimerService_h
#define FastTimerService_h

// C++ headers
#include <cmath>
#include <string>
#include <map>
#include <unordered_map>
#include <chrono>
#include <unistd.h>

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
#include "HLTrigger/Timer/interface/FastTimer.h"


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

  // query the current module/path/event
  // Note: these functions incur in a "per-call timer overhead" (see above), currently of the order of 340ns
  double currentModuleTime(edm::StreamID) const;            // return the time spent since the last preModuleEvent() event
  double currentPathTime(edm::StreamID) const;              // return the time spent since the last prePathEvent() event
  double currentEventTime(edm::StreamID) const;             // return the time spent since the last preEvent() event

  // query the time spent in a module/path (available after it has run)
  double queryModuleTime(edm::StreamID, const edm::ModuleDescription &) const;
  double queryModuleTimeByLabel(edm::StreamID, const std::string &) const;
  double queryModuleTimeByType(edm::StreamID, const std::string &) const;
  double queryPathActiveTime(edm::StreamID, const std::string &) const;
  double queryPathExclusiveTime(edm::StreamID, const std::string &) const;
  double queryPathTotalTime(edm::StreamID, const std::string &) const;

  // query the time spent in the current event's
  //  - source        (available during event processing)
  //  - all paths     (available during endpaths)
  //  - all endpaths  (available after all endpaths have run)
  //  - processing    (available after the event has been processed)
  double querySourceTime(edm::StreamID) const;
  double queryPathsTime(edm::StreamID) const;
  double queryEndPathsTime(edm::StreamID) const;
  double queryEventTime(edm::StreamID) const;

  /* FIXME not yet implemented
  // try to assess the overhead which may not be included in the source, paths and event timers
  double queryPreSourceOverhead(edm::StreamID) const;       // time spent after the previous event's postEvent and this event's preSource
  double queryPreEventOverhead(edm::StreamID) const;        // time spent after this event's postSource and preEvent
  double queryPreEndPathsOverhead(edm::StreamID) const;     // time spent after the last path's postPathEvent and the first endpath's prePathEvent
  */

private:
  void preallocate(edm::service::SystemBounds const &);
  void postEndJob();
  void preGlobalBeginRun(edm::GlobalContext const &);
  void postGlobalEndRun(edm::GlobalContext const &);
  void preModuleBeginJob(edm::ModuleDescription const &);
  void preSourceEvent(  edm::StreamID );
  void postSourceEvent( edm::StreamID );
  void preEvent(edm::StreamContext const &);
  void postEvent(edm::StreamContext const &);
  void prePathEvent(edm::StreamContext const &, edm::PathContext const &);
  void postPathEvent(edm::StreamContext const &, edm::PathContext const &,edm:: HLTPathStatus const &);
  void preModuleEvent(edm::StreamContext const &, edm::ModuleCallingContext const &);
  void postModuleEvent(edm::StreamContext const &, edm::ModuleCallingContext const &);
  void preModuleEventDelayedGet(edm::StreamContext const &, edm::ModuleCallingContext const &);
  void postModuleEventDelayedGet(edm::StreamContext const &, edm::ModuleCallingContext const &);

public:
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

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
      // the DAQ destroys and re-creates the DQM and DQMStore services at each reconfigure, so we don't need to clean them up
      dqm_active = nullptr;
      run_in_path = nullptr;
      counter = 0;
    }
  };

  struct PathInfo {
    std::vector<ModuleInfo *>   modules;            // list of all modules contributing to the path (duplicate modules stored as null pointers)
    FastTimer                   timer;              // per-event timer
    double                      time_active;        // time actually spent in this path
    double                      time_exclusive;     // time actually spent in this path, in modules that are not run on any other paths
    double                      time_premodules;    // time spent between "begin path" and the first "begin module"
    double                      time_intermodules;  // time spent between modules
    double                      time_postmodules;   // time spent between the last "end module" and "end path"
    double                      time_overhead;      // sum of time_premodules, time_intermodules, time_postmodules
    double                      time_total;         // sum of the time spent in all modules which would have run in this path (plus overhead)
    double                      summary_active;
    double                      summary_premodules;
    double                      summary_intermodules;
    double                      summary_postmodules;
    double                      summary_overhead;
    double                      summary_total;
    uint32_t                    last_run;           // index of the last module run in this path
    uint32_t                    index;              // index of the Path or EndPath in the "schedule"
    bool                        accept;             // flag indicating if the path acepted the event
    TH1F *                      dqm_active;         // see time_active
    TH1F *                      dqm_exclusive;      // see time_exclusive
    TH1F *                      dqm_premodules;     // see time_premodules
    TH1F *                      dqm_intermodules;   // see time_intermodules
    TH1F *                      dqm_postmodules;    // see time_postmodules
    TH1F *                      dqm_overhead;       // see time_overhead
    TH1F *                      dqm_total;          // see time_total
    TH1F *                      dqm_module_counter; // for each module in the path, track how many times it ran
    TH1F *                      dqm_module_active;  // for each module in the path, track the active time spent
    TH1F *                      dqm_module_total;   // for each module in the path, track the total time spent

  public:
    PathInfo() :
      modules(),
      timer(),
      time_active(0.),
      time_exclusive(0.),
      time_premodules(0.),
      time_intermodules(0.),
      time_postmodules(0.),
      time_overhead(0.),
      time_total(0.),
      summary_active(0.),
      summary_premodules(0.),
      summary_intermodules(0.),
      summary_postmodules(0.),
      summary_overhead(0.),
      summary_total(0.),
      last_run(0),
      index(0),
      accept(false),
      dqm_active(nullptr),
      dqm_exclusive(nullptr),
      dqm_premodules(nullptr),
      dqm_intermodules(nullptr),
      dqm_postmodules(nullptr),
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
      modules.clear();
      timer.reset();
      time_active = 0.;
      time_exclusive = 0.;
      time_premodules = 0.;
      time_intermodules = 0.;
      time_postmodules = 0.;
      time_overhead = 0.;
      time_total = 0.;
      summary_active = 0.;
      summary_premodules = 0.;
      summary_intermodules = 0.;
      summary_postmodules = 0.;
      summary_overhead = 0.;
      summary_total = 0.;
      last_run = 0;
      index = 0;
      accept = false;

      // the DAQ destroys and re-creates the DQM and DQMStore services at each reconfigure, so we don't need to clean them up
      dqm_active = nullptr;
      dqm_premodules = nullptr;
      dqm_intermodules = nullptr;
      dqm_postmodules = nullptr;
      dqm_overhead = nullptr;
      dqm_total = nullptr;
      dqm_module_counter = nullptr;
      dqm_module_active = nullptr;
      dqm_module_total = nullptr;
    }
  };

  template <typename T> using PathMap   = std::unordered_map<std::string, T>;
  template <typename T> using ModuleMap = std::unordered_map<edm::ModuleDescription const *, T>;

  // timer configuration
  bool                                          m_use_realtime;
  bool                                          m_enable_timing_paths;
  bool                                          m_enable_timing_modules;
  bool                                          m_enable_timing_exclusive;
  const bool                                    m_enable_timing_summary;
  const bool                                    m_skip_first_path;

  // dqm configuration
  bool                                          m_enable_dqm;                   // non const because the availability of the DQMStore can only be checked during the begin job
  const bool                                    m_enable_dqm_bypath_active;     // require per-path timers
  const bool                                    m_enable_dqm_bypath_total;      // require per-path and per-module timers
  const bool                                    m_enable_dqm_bypath_overhead;   // require per-path and per-module timers
  const bool                                    m_enable_dqm_bypath_details;    // require per-path and per-module timers
  const bool                                    m_enable_dqm_bypath_counters;
  const bool                                    m_enable_dqm_bypath_exclusive;
  const bool                                    m_enable_dqm_bymodule;          // require per-module timers
  const bool                                    m_enable_dqm_bymoduletype;      // require per-module timers
  const bool                                    m_enable_dqm_summary;
  const bool                                    m_enable_dqm_byluminosity;
  const bool                                    m_enable_dqm_byls;
  const bool                                    m_enable_dqm_bynproc;

  bool                                          m_nproc_enabled;                // check if the plots by number of processes have been correctly enabled
  unsigned int                                  m_concurrent_runs;
  unsigned int                                  m_concurrent_streams;
  unsigned int                                  m_concurrent_threads;

  const double                                  m_dqm_eventtime_range;
  const double                                  m_dqm_eventtime_resolution;
  const double                                  m_dqm_pathtime_range;
  const double                                  m_dqm_pathtime_resolution;
  const double                                  m_dqm_moduletime_range;
  const double                                  m_dqm_moduletime_resolution;
  const double                                  m_dqm_luminosity_range;
  const double                                  m_dqm_luminosity_resolution;
  const uint32_t                                m_dqm_ls_range;
  const std::string                             m_dqm_path;
  const edm::InputTag                           m_luminosity_label;     // label of the per-Event luminosity EDProduct
  const std::vector<unsigned int>               m_supported_processes;  // possible number of concurrent processes

  // job configuration and caching
  std::string                                   m_first_path;           // the framework does not provide a pre-paths or pre-endpaths signal,
  std::string                                   m_last_path;            // so we emulate them keeping track of the first and last path and endpath
  std::string                                   m_first_endpath;
  std::string                                   m_last_endpath;
  bool                                          m_is_first_event;

  struct Timing {
    double              presource;              // time spent between the end of the previous Event, LumiSection or Run, and the beginning of the Source
    double              source;                 // time spent processing the Source
    double              preevent;               // time spent between the end of the Source and the new Event, Lumisection or Run
    double              event;                  // time spent processing the Event
    double              all_paths;              // time spent processing all Paths
    double              all_endpaths;           // time spent processing all EndPaths
    double              interpaths;             // time spent between the Paths (and EndPaths - i.e. the sum of all the entries in the following vector)
    std::vector<double> paths_interpaths;       // time spent between the beginning of the Event and the first Path, between Paths, and between the last (End)Path and the end of the Event
    unsigned int        count;                  // number of processed events (used by the per-run and per-job accounting)

    Timing() :
      presource     (0.),
      source        (0.),
      preevent      (0.),
      event         (0.),
      all_paths     (0.),
      all_endpaths  (0.),
      interpaths    (0.),
      paths_interpaths(),
      count         (0)
    { }

    Timing(std::vector<double>::size_type size) :
      presource     (0.),
      source        (0.),
      preevent      (0.),
      event         (0.),
      all_paths     (0.),
      all_endpaths  (0.),
      interpaths    (0.),
      paths_interpaths(size, 0.),
      count         (0)
    { }

    void reset() {
      presource     = 0.;
      source        = 0.;
      preevent      = 0.;
      event         = 0.;
      all_paths     = 0.;
      all_endpaths  = 0.;
      interpaths    = 0.;
      paths_interpaths.clear();
      count         = 0;
    }

    void reset(std::vector<double>::size_type size) {
      presource     = 0.;
      source        = 0.;
      preevent      = 0.;
      event         = 0.;
      all_paths     = 0.;
      all_endpaths  = 0.;
      interpaths    = 0.;
      paths_interpaths.assign(size, 0.);
      count         = 0;
    }

    Timing & operator+=(Timing const & other) {
      assert( paths_interpaths.size() == other.paths_interpaths.size() );

      presource     += other.presource;
      source        += other.source;
      preevent      += other.preevent;
      event         += other.event;
      all_paths     += other.all_paths;
      all_endpaths  += other.all_endpaths;
      interpaths    += other.interpaths;
      for (unsigned int i = 0; i < paths_interpaths.size(); ++i)
        paths_interpaths[i] += other.paths_interpaths[i];
      count         += other.count;

      return *this;
    }

    Timing operator+(Timing const & other) const {
      Timing result = *this;
      result += other;
      return result;
    }

  };

  std::vector<Timing>                           m_run_summary;          // time accounting per-run
  Timing                                        m_job_summary;          // time accounting per-job

  // DQM

  // set of summary plots
  struct SummaryPlots {
    TH1F *     presource;
    TH1F *     source;
    TH1F *     preevent;
    TH1F *     event;
    TH1F *     all_paths;
    TH1F *     all_endpaths;
    TH1F *     interpaths;

    SummaryPlots() :
      presource     (nullptr),
      source        (nullptr),
      preevent      (nullptr),
      event         (nullptr),
      all_paths     (nullptr),
      all_endpaths  (nullptr),
      interpaths    (nullptr)
    { }

    void reset() {
      presource     = nullptr;
      source        = nullptr;
      preevent      = nullptr;
      event         = nullptr;
      all_paths     = nullptr;
      all_endpaths  = nullptr;
      interpaths    = nullptr;
    }

    void fill(Timing const & value) {
      // convert on the fly from seconds to ms
      presource     ->Fill( 1000. * value.presource );
      source        ->Fill( 1000. * value.source );
      preevent      ->Fill( 1000. * value.preevent );
      event         ->Fill( 1000. * value.event );
      all_paths     ->Fill( 1000. * value.all_paths );
      all_endpaths  ->Fill( 1000. * value.all_endpaths );
      interpaths    ->Fill( 1000. * value.interpaths );
    }

  };

  struct SummaryProfiles {
    TProfile * presource;
    TProfile * source;
    TProfile * preevent;
    TProfile * event;
    TProfile * all_paths;
    TProfile * all_endpaths;
    TProfile * interpaths;

    SummaryProfiles() :
      presource     (nullptr),
      source        (nullptr),
      preevent      (nullptr),
      event         (nullptr),
      all_paths     (nullptr),
      all_endpaths  (nullptr),
      interpaths    (nullptr)
    { }

    void reset() {
      presource     = nullptr;
      source        = nullptr;
      preevent      = nullptr;
      event         = nullptr;
      all_paths     = nullptr;
      all_endpaths  = nullptr;
      interpaths    = nullptr;
    }

    void fill(double x, Timing const & value) {
      presource     ->Fill( x, 1000. * value.presource );
      source        ->Fill( x, 1000. * value.source );
      preevent      ->Fill( x, 1000. * value.preevent );
      event         ->Fill( x, 1000. * value.event );
      all_paths     ->Fill( x, 1000. * value.all_paths );
      all_endpaths  ->Fill( x, 1000. * value.all_endpaths );
      interpaths    ->Fill( x, 1000. * value.interpaths );
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

    // time accounting per-event
    Timing                                          timing;

    // overall plots
    SummaryPlots                                    dqm;                        // event summary plots
    SummaryProfiles                                 dqm_byls;                   // plots per lumisection
    SummaryProfiles                                 dqm_byluminosity;           // plots vs. instantaneous luminosity

    // plots to be summed over nodes with the same number of processes/threads
    SummaryPlots                                    dqm_nproc;                  // event summary plots
    SummaryProfiles                                 dqm_nproc_byls;             // plots per lumisection
    SummaryProfiles                                 dqm_nproc_byluminosity;     // plots vs. instantaneous luminosity

    // plots by path
    TProfile *                                      dqm_paths_active_time;
    TProfile *                                      dqm_paths_total_time;
    TProfile *                                      dqm_paths_exclusive_time;
    TProfile *                                      dqm_paths_interpaths;

    // per-path, per-module and per-module-type accounting
    PathInfo *                                      current_path;
    ModuleInfo *                                    current_module;
    ModuleInfo *                                    first_module_in_path;
    PathMap<PathInfo>                               paths;
    std::unordered_map<std::string, ModuleInfo>     modules;
    std::unordered_map<std::string, ModuleInfo>     moduletypes;
    ModuleMap<ModuleInfo *>                         fast_modules;               // these assume that ModuleDescription are stored in the same object through the whole job,
    ModuleMap<ModuleInfo *>                         fast_moduletypes;           // which is true only *after* the edm::Worker constructors have run

    StreamData() :
      // timers
      timer_event(),
      timer_source(),
      timer_paths(),
      timer_endpaths(),
      timer_path(),
      timer_last_path(),
      // time accounting per-event
      timing(),
      // overall plots
      dqm(),
      dqm_byls(),
      dqm_byluminosity(),
      // plots to be summed over nodes with the same number of processes/threads
      dqm_nproc(),
      dqm_nproc_byls(),
      dqm_nproc_byluminosity(),
      // plots by path
      dqm_paths_active_time(nullptr),
      dqm_paths_total_time(nullptr),
      dqm_paths_exclusive_time(nullptr),
      dqm_paths_interpaths(nullptr),
      // per-path, per-module and per-module-type accounting
      current_path(nullptr),
      current_module(nullptr),
      first_module_in_path(nullptr),
      paths(),
      modules(),
      moduletypes(),
      fast_modules(),
      fast_moduletypes()
    { }

  };

  std::vector<StreamData> m_stream;

  static
  double delta(FastTimer::Clock::time_point const & first, FastTimer::Clock::time_point const & second)
  {
    return std::chrono::duration_cast<std::chrono::duration<double>>(second - first).count();
  }

  // associate to a path all the modules it contains
  void fillPathMap(std::string const & name, std::vector<std::string> const & modules);

};

#endif // ! FastTimerService_h

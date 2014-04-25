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

  // reserve plots for timing vs. luminosity - these should only be called before any begin run (actually, before any preStreamBeginRun)
  unsigned int reserveLuminosityPlots(std::string const & name, std::string const & title, std::string const & label, double range, double resolution);
  unsigned int reserveLuminosityPlots(std::string && name, std::string && title, std::string && label, double range, double resolution);

  // set the event luminosity
  void setLuminosity(unsigned int stream_id, unsigned int luminosity_id, double value);

  // query the current module/path/event
  // Note: these functions incur in a "per-call timer overhead" (see above), currently of the order of 340ns
  double currentModuleTime(edm::StreamID) const;            // return the time spent since the last preModuleEvent() event
  double currentPathTime(edm::StreamID) const;              // return the time spent since the last prePathEvent() event
  double currentEventTime(edm::StreamID) const;             // return the time spent since the last preEvent() event

  // query the time spent in a module/path (available after it has run)
  double queryModuleTime(edm::StreamID, const edm::ModuleDescription &) const;
  double queryModuleTime(edm::StreamID, unsigned int id) const;
  double queryModuleTimeByLabel(edm::StreamID, const std::string &) const;
  double queryModuleTimeByType(edm::StreamID, const std::string &) const;
  /* FIXME re-implement taking into account subprocesses
  double queryPathActiveTime(edm::StreamID, const std::string &) const;
  double queryPathExclusiveTime(edm::StreamID, const std::string &) const;
  double queryPathTotalTime(edm::StreamID, const std::string &) const;
  */

  // query the time spent in the current event's
  //  - source        (available during event processing)
  //  - all paths     (available during endpaths)
  //  - all endpaths  (available after all endpaths have run)
  //  - processing    (available after the event has been processed)
  double querySourceTime(edm::StreamID) const;
  double queryEventTime(edm::StreamID) const;
  /* FIXME re-implement taking into account subprocesses
  double queryPathsTime(edm::StreamID) const;
  double queryEndPathsTime(edm::StreamID) const;
  */

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
  void preStreamBeginRun(edm::StreamContext const &);
  void postStreamBeginRun(edm::StreamContext const &);
  void postStreamEndRun(edm::StreamContext const &);
  void postStreamBeginLumi(edm::StreamContext const &);
  void postStreamEndLumi(edm::StreamContext const& );
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

  struct LuminosityDescription {
    std::string name;
    std::string title;
    std::string label;
    double      range;
    double      resolution;

    LuminosityDescription(std::string const & _name, std::string const & _title, std::string const & _label, double _range, double _resolution) :
      name(_name),
      title(_title),
      label(_label),
      range(_range),
      resolution(_resolution)
    { }

    LuminosityDescription(std::string && _name, std::string && _title, std::string const & _label, double _range, double _resolution) :
      name(_name),
      title(_title),
      label(_label),
      range(_range),
      resolution(_resolution)
    { }
  };


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
    ModuleInfo *                first_module;       // first module actually run in this path
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
    uint32_t                    last_run;           // index of the last module run in this path, plus one
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
      first_module(nullptr),
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
      first_module = nullptr;
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

      // the DQM plots are owned by the DQMStore
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

  // the vector is indexed by the peudo-process id, the paths by their path name
  template <typename T>
  using PathMap = std::vector<std::unordered_map<std::string, T>>;

  // key on ModuleDescription::id()
  template <typename T>
  using ModuleMap = std::vector<T>;

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
  const bool                                    m_enable_dqm_byls;
  const bool                                    m_enable_dqm_bynproc;

  unsigned int                                  m_concurrent_runs;
  unsigned int                                  m_concurrent_streams;
  unsigned int                                  m_concurrent_threads;
  unsigned int                                  m_module_id;                    // pseudo module id for the FastTimerService, needed by the thread-safe DQMStore

  const double                                  m_dqm_eventtime_range;
  const double                                  m_dqm_eventtime_resolution;
  const double                                  m_dqm_pathtime_range;
  const double                                  m_dqm_pathtime_resolution;
  const double                                  m_dqm_moduletime_range;
  const double                                  m_dqm_moduletime_resolution;
  std::string                                   m_dqm_path;

  // job configuration and caching
  bool                                          m_is_first_event;


  struct ProcessDescription {
    std::string         name;
    std::string         first_path;           // the framework does not provide a pre/postPaths or pre/postEndPaths signal,
    std::string         last_path;            // so we emulate them keeping track of the first and last Path and EndPath
    std::string         first_endpath;
    std::string         last_endpath;
    edm::ParameterSetID pset;
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
    double              preevent;               // time spent between the end of the Source and the new Event, Lumisection or Run
    double              event;                  // time spent processing the Event
    double              all_paths;              // time spent processing all Paths
    double              all_endpaths;           // time spent processing all EndPaths
    double              interpaths;             // time spent between the Paths (and EndPaths - i.e. the sum of all the entries in the following vector)
    std::vector<double> paths_interpaths;       // time spent between the beginning of the Event and the first Path, between Paths, and between the last (End)Path and the end of the Event

    TimingPerProcess() :
      preevent      (0.),
      event         (0.),
      all_paths     (0.),
      all_endpaths  (0.),
      interpaths    (0.),
      paths_interpaths()
    { }

    void reset() {
      preevent      = 0.;
      event         = 0.;
      all_paths     = 0.;
      all_endpaths  = 0.;
      interpaths    = 0.;
      paths_interpaths.assign(paths_interpaths.size(), 0.);
    }

    TimingPerProcess & operator+=(TimingPerProcess const & other) {
      assert( paths_interpaths.size() == other.paths_interpaths.size() );

      preevent      += other.preevent;
      event         += other.event;
      all_paths     += other.all_paths;
      all_endpaths  += other.all_endpaths;
      interpaths    += other.interpaths;
      for (unsigned int i = 0; i < paths_interpaths.size(); ++i)
        paths_interpaths[i] += other.paths_interpaths[i];
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
    TProfile * interpaths;

    PathProfilesPerProcess() :
      active_time   (nullptr),
      total_time    (nullptr),
      exclusive_time(nullptr),
      interpaths    (nullptr)
    {}

    ~PathProfilesPerProcess() {
      reset();
    }

    void reset() {
      // the DQM plots are owned by the DQMStore
      active_time    = nullptr;
      total_time     = nullptr;
      exclusive_time = nullptr;
      interpaths     = nullptr;
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

    // luminosity per event
    std::vector<double>                             luminosity;

    // overall plots
    SummaryPlots                                    dqm;                            // whole event summary plots
    std::vector<SummaryProfiles>                    dqm_byluminosity;               // whole event plots vs. "luminosity"
    std::vector<SummaryPlotsPerProcess>             dqm_perprocess;                 // per-process event summary plots
    std::vector<std::vector<SummaryProfilesPerProcess>> dqm_perprocess_byluminosity;    // per-process plots vs. "luminosity"

    // plots by path
    std::vector<PathProfilesPerProcess>             dqm_paths;

    // per-path, per-module and per-module-type accounting
    PathInfo *                                      current_path;
    ModuleInfo *                                    current_module;
    PathMap<PathInfo>                               paths;
    std::unordered_map<std::string, ModuleInfo>     modules;
    std::unordered_map<std::string, ModuleInfo>     moduletypes;
    ModuleMap<ModuleInfo *>                         fast_modules;               // these assume that module ids are constant throughout the whole job,
    ModuleMap<ModuleInfo *>                         fast_moduletypes;

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
      // luminosity per event
      luminosity(),
      // overall plots
      dqm(),
      dqm_byluminosity(),
      dqm_perprocess(),
      dqm_perprocess_byluminosity(),
      // plots by path
      dqm_paths(),
      // per-path, per-module and per-module-type accounting
      current_path(nullptr),
      current_module(nullptr),
      paths(),
      modules(),
      moduletypes(),
      fast_modules(),
      fast_moduletypes()
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
      // luminosity per event
      for (auto & lumi: luminosity)
        lumi = 0;
      // overall plots
      dqm.reset();
      for (auto & plots: dqm_byluminosity)
        plots.reset();
      for (auto & perprocess_plots: dqm_perprocess)
        perprocess_plots.reset();
      for (auto & process_plots: dqm_perprocess_byluminosity)
        for (auto & plots: process_plots)
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
      for (auto & keyval: moduletypes)
        keyval.second.reset();
    }

  };

  // process descriptions
  std::vector<ProcessDescription>               m_process;

  // description of the luminosity axes
  std::vector<LuminosityDescription>            m_dqm_luminosity;

  // stream data
  std::vector<StreamData>                       m_stream;

  // summary data
  std::vector<Timing>                           m_run_summary;                  // whole event time accounting per-run
  Timing                                        m_job_summary;                  // whole event time accounting per-run
  std::vector<std::vector<TimingPerProcess>>    m_run_summary_perprocess;       // per-process time accounting per-job
  std::vector<TimingPerProcess>                 m_job_summary_perprocess;       // per-process time accounting per-job

  static
  double delta(FastTimer::Clock::time_point const & first, FastTimer::Clock::time_point const & second)
  {
    return std::chrono::duration_cast<std::chrono::duration<double>>(second - first).count();
  }

  // associate to a path all the modules it contains
  void fillPathMap(unsigned int pid, std::string const & name, std::vector<std::string> const & modules);

  // print a timing summary for the run or job
  void printSummary(Timing const & summary, std::string const & label) const;
  void printProcessSummary(Timing const & total, TimingPerProcess const & summary, std::string const & label, std::string const & process) const;

  // assign a "process id" to a process, given its ProcessContext
  static
  unsigned int processID(edm::ProcessContext const *);

};

#endif // ! FastTimerService_h

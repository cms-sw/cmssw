#ifndef FastTimerService_h
#define FastTimerService_h

//system headers
#ifdef __linux
#include <time.h>
#else
typedef int clockid_t;
#endif

/* Darwin system headers */
#if defined(__APPLE__) || defined(__MACH__)
#include <mach/clock.h>
#include <mach/mach.h>
#endif // defined(__APPLE__) || defined(__MACH__)

// C++ headers
#include <cmath>
#include <string>
#include <map>
#include <tr1/unordered_map>
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
Timer per-call overhead (on lxplus, SLC 5.7):

Test results for "clock_gettime(CLOCK_THREAD_CPUTIME_ID, & value)"
 - average time per call: 340 ns
 - resolution:              ? ns

Test results for "clock_gettime(CLOCK_PROCESS_CPUTIME_ID, & value)"
 - average time per call: 360 ns
 - resolution:              ? ns

Test results for "clock_gettime(CLOCK_REALTIME, & value)"
 - average time per call: 320 ns
 - resolution:              1 us

Test results for "getrusage(RUSAGE_SELF, & value)"
 - average time per call: 380 ns
 - resolution:              1 ms

Test results for "gettimeofday(& value, NULL)"
 - average time per call:  65 ns
 - resolution:              1 us

Assuming an HLT process with ~2500 modules and ~500 paths, tracking each step
with clock_gettime(CLOCK_THREAD_CPUTIME_ID) gives a per-event overhead of 2 ms

Detailed informations on different timers can be extracted running $CMSSW_RELEASE_BASE/test/$SCRAM_ARCH/testTimer .
*/


class FastTimerService {
public:
  FastTimerService(const edm::ParameterSet &, edm::ActivityRegistry & );
  ~FastTimerService();

  // query the current module/path/event
  // Note: these functions incur in a "per-call timer overhead" (see above), currently of the order of 340ns
  double currentModuleTime() const;         // return the time spent since the last preModule() event
  double currentPathTime() const;           // return the time spent since the last preProcessPath() event
  double currentEventTime() const;          // return the time spent since the last preProcessEvent() event

  // query the time spent in a module/path (available after it has run)
  double queryModuleTime(const edm::ModuleDescription &) const;
  double queryPathActiveTime(const std::string &) const;
  double queryPathTotalTime(const std::string &) const;

  // query the time spent in the current event's
  //  - source        (available during event processing)
  //  - all paths     (available during endpaths)
  //  - all endpaths  (available after all endpaths have run)
  //  - processing    (available after the event has been processed)
  double querySourceTime() const;
  double queryPathsTime() const;
  double queryEndPathsTime() const;
  double queryEventTime() const;

  /* FIXME not yet implemented
  // try to assess the overhead which may not be included in the source, paths and event timers
  double queryPreSourceOverhead() const;    // time spent after the previous event's postProcessEvent and this event's preSource
  double queryPreEventOverhead() const;     // time spent after this event's postSource and preProcessEvent
  double queryPreEndPathsOverhead() const;  // time spent after the last path's postProcessPath and the first endpath's preProcessPath
  */

private:
  void postBeginJob();
  void postEndJob();
  void preModuleBeginJob( edm::ModuleDescription const & );
  void preProcessEvent( edm::EventID const &, edm::Timestamp const & );
  void postProcessEvent( edm::Event const &, edm::EventSetup const & );
  void preSource();
  void postSource();
  void prePathBeginRun( std::string const & );
  void preProcessPath(  std::string const & );
  void postProcessPath( std::string const &, edm::HLTPathStatus const & );
  void preModule(  edm::ModuleDescription const & );
  void postModule( edm::ModuleDescription const & );

  // needed for the DAQ when reconfiguring between runs
  void reset();

public:
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:

  struct ModuleInfo {
    double                      time_active;        // per-event timer: time actually spent in this module
    double                      summary_active;
    TH1F *                      dqm_active;
    bool                        has_just_run;       // flag set to check if a module was active inside a particular path, or not
    bool                        is_exclusive;       // flag set to check if a module has been run only once

  public:
    ModuleInfo() :
      time_active(0.),
      summary_active(0.),
      dqm_active(0),
      has_just_run(false),
      is_exclusive(false)
    { }

    ~ModuleInfo() {
      reset();
    }

    // reset the timers and DQM plots
    void reset() {
      time_active = 0.;
      summary_active = 0.;
      // the DAQ destroys and re-creates the DQM and DQMStore services at each reconfigure, so we don't need to clean them up
      dqm_active = 0;
      has_just_run = false;
      is_exclusive = false;
    }
  };

  struct PathInfo {
    std::vector<ModuleInfo *>   modules;            // list of all modules contributing to the path (duplicate modules stored as null pointers)
    double                      time_active;        // per-event timer: time actually spent in this path
    double                      time_premodules;    // per-event timer: time spent between "begin path" and the first "begin module"
    double                      time_intermodules;  // per-event timer: time spent between modules
    double                      time_postmodules;   // per-event timer: time spent between the last "end module" and "end path"
    double                      time_overhead;      // per-event timer: sum of time_premodules, time_intermodules, time_postmodules
    double                      time_total;         // per-event timer: sum of the time spent in all modules which would have run in this path (plus overhead)
    double                      summary_active;
    double                      summary_premodules;
    double                      summary_intermodules;
    double                      summary_postmodules;
    double                      summary_overhead;
    double                      summary_total;
    uint32_t                    last_run;
    uint32_t                    index;              // index of the Path or EndPath in the "schedule"
    TH1F *                      dqm_active;
    TH1F *                      dqm_exclusive;
    TH1F *                      dqm_premodules;
    TH1F *                      dqm_intermodules;
    TH1F *                      dqm_postmodules;
    TH1F *                      dqm_overhead;
    TH1F *                      dqm_total;
    TH1F *                      dqm_module_counter; // for each module in the path, track how many times it ran
    TH1F *                      dqm_module_active;  // for each module in the path, track the active time spent 
    TH1F *                      dqm_module_total;   // for each module in the path, track the total time spent 

  public:
    PathInfo() :
      modules(),
      time_active(0.),
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
      dqm_active(0),
      dqm_exclusive(0),
      dqm_premodules(0),
      dqm_intermodules(0),
      dqm_postmodules(0),
      dqm_overhead(0),
      dqm_total(0),
      dqm_module_counter(0),
      dqm_module_active(0),
      dqm_module_total(0)
    { }

    ~PathInfo() {
      reset();
    }

    // reset the timers and DQM plots
    void reset() {
      modules.clear();
      time_active = 0.;
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

      // the DAQ destroys and re-creates the DQM and DQMStore services at each reconfigure, so we don't need to clean them up
      dqm_active = 0; 
      dqm_premodules = 0;
      dqm_intermodules = 0;
      dqm_postmodules = 0;
      dqm_overhead = 0;
      dqm_total = 0;
      dqm_module_counter = 0;
      dqm_module_active = 0;
      dqm_module_total = 0;
    }
  };

  template <typename T> class PathMap   : public std::tr1::unordered_map<std::string, T> {};
  template <typename T> class ModuleMap : public std::tr1::unordered_map<edm::ModuleDescription const *, T> {};

  // timer configuration
  const clockid_t                               m_timer_id;             // the default is to use CLOCK_THREAD_CPUTIME_ID, unless useRealTimeClock is set, which will use CLOCK_REALTIME
  bool                                          m_is_cpu_bound;         // if the process is not bound to a single CPU, per-thread or per-process measuerements may be unreliable
  bool                                          m_enable_timing_paths;
  bool                                          m_enable_timing_modules;
  bool                                          m_enable_timing_exclusive;
  const bool                                    m_enable_timing_summary;
  const bool                                    m_skip_first_path;

  // dqm configuration
  const bool                                    m_enable_dqm;
  const bool                                    m_enable_dqm_bypath_active;     // require per-path timers
  const bool                                    m_enable_dqm_bypath_total;      // require per-path and per-module timers
  const bool                                    m_enable_dqm_bypath_overhead;   // require per-path and per-module timers
  const bool                                    m_enable_dqm_bypath_details;    // require per-path and per-module timers
  const bool                                    m_enable_dqm_bypath_counters;
  const bool                                    m_enable_dqm_bypath_exclusive;
  const bool                                    m_enable_dqm_bymodule;          // require per-module timers
  const bool                                    m_enable_dqm_bylumi;
  const double                                  m_dqm_eventtime_range;
  const double                                  m_dqm_eventtime_resolution;
  const double                                  m_dqm_pathtime_range;
  const double                                  m_dqm_pathtime_resolution;
  const double                                  m_dqm_moduletime_range;
  const double                                  m_dqm_moduletime_resolution;
  const uint32_t                                m_dqm_lumi_range;
  std::string                                   m_dqm_path;

  // job configuration and caching
  std::string const *                           m_first_path;           // the framework does not provide a pre-paths or pre-endpaths signal,
  std::string const *                           m_last_path;            // so we emulate them keeping track of the first and last path and endpath
  std::string const *                           m_first_endpath;
  std::string const *                           m_last_endpath;
  bool                                          m_is_first_module;      // helper to measure the time spent between the beginning of the path and the execution of the first module

  // per-event accounting
  double                                        m_event;
  double                                        m_source;
  double                                        m_all_paths;
  double                                        m_all_endpaths;
  double                                        m_interpaths;

  // per-job summary
  unsigned int                                  m_summary_events;       // number of events
  double                                        m_summary_event;
  double                                        m_summary_source;
  double                                        m_summary_all_paths;
  double                                        m_summary_all_endpaths;
  double                                        m_summary_interpaths;

  // DQM
  DQMStore *                                    m_dqms;
  TH1F *                                        m_dqm_event;
  TH1F *                                        m_dqm_source;
  TH1F *                                        m_dqm_all_paths;
  TH1F *                                        m_dqm_all_endpaths;
  TH1F *                                        m_dqm_interpaths;
  TProfile *                                    m_dqm_paths_active_time;
  TProfile *                                    m_dqm_paths_total_time;
  TProfile *                                    m_dqm_paths_exclusive_time;
  TProfile *                                    m_dqm_paths_interpaths;

  // per-lumisection plots
  TProfile *                                    m_dqm_bylumi_event;
  TProfile *                                    m_dqm_bylumi_source;
  TProfile *                                    m_dqm_bylumi_all_paths;
  TProfile *                                    m_dqm_bylumi_all_endpaths;
  TProfile *                                    m_dqm_bylumi_interpaths;

  // per-path and per-module accounting
  PathInfo *                                    m_current_path;
  PathMap<PathInfo>                             m_paths;
  ModuleMap<ModuleInfo>                         m_modules;              // this assumes that ModuleDescription are stored in the same object through the whole job,
                                                                        // which is true only *after* the edm::Worker constructors have run
  std::vector<PathInfo *>                       m_cache_paths;
  std::vector<ModuleInfo *>                     m_cache_modules;


  // timers
  std::pair<struct timespec, struct timespec>   m_timer_event;          // track time spent in each event
  std::pair<struct timespec, struct timespec>   m_timer_source;         // track time spent in the source
  std::pair<struct timespec, struct timespec>   m_timer_paths;          // track time spent in all paths
  std::pair<struct timespec, struct timespec>   m_timer_endpaths;       // track time spent in all endpaths
  std::pair<struct timespec, struct timespec>   m_timer_path;           // track time spent in each path
  std::pair<struct timespec, struct timespec>   m_timer_module;         // track time spent in each module
  struct timespec                               m_timer_first_module;   // record the start of the first active module in a path, if any

#if defined(__APPLE__) || defined (__MACH__)
  clock_serv_t m_clock_port;
#endif // defined(__APPLE__) || defined(__MACH__)

  void gettime(struct timespec & stamp) const
  {
#if defined(_POSIX_TIMERS) && _POSIX_TIMERS >= 0
    clock_gettime(m_timer_id, & stamp);
#else
// special cases which do not support _POSIX_TIMERS
#if defined(__APPLE__) || defined (__MACH__)
    mach_timespec_t timespec;
    clock_get_time(m_clock_port, &timespec);
    stamp.tv_sec  = timespec.tv_sec;
    stamp.tv_nsec = timespec.tv_nsec;
#endif // defined(__APPLE__) || defined(__MACH__)
#endif // defined(_POSIX_TIMERS) && _POSIX_TIMERS >= 0
  }

  void start(std::pair<struct timespec, struct timespec> & times) const
  {
    gettime(times.first);
  }

  void stop(std::pair<struct timespec, struct timespec> & times) const
  {
    gettime(times.second);
  }

  static
  double delta(const struct timespec & first, const struct timespec & second)
  {
    if (second.tv_nsec > first.tv_nsec)
      return (double) (second.tv_sec - first.tv_sec) + (double) (second.tv_nsec - first.tv_nsec) / (double) 1e9;
    else
      return (double) (second.tv_sec - first.tv_sec) - (double) (first.tv_nsec - second.tv_nsec) / (double) 1e9;
  }

  static
  double delta(const std::pair<struct timespec, struct timespec> & times)
  {
    return delta(times.first, times.second);
  }

  // find the module description associated to a module, by name
  edm::ModuleDescription const * findModuleDescription(const std::string & module) const;

  // associate to a path all the modules it contains
  void fillPathMap(std::string const & name, std::vector<std::string> const & modules);

};

#endif // ! FastTimerService_h

#ifndef Framework_Schedule_h
#define Framework_Schedule_h

/*
  Author: Jim Kowalkowski  28-01-06

  $Id: Schedule.h,v 1.21 2007/03/22 06:07:18 wmtan Exp $

  A class for creating a schedule based on paths in the configuration file.
  The schedule is maintained as a sequence of paths.
  After construction, events can be fed to the object and passed through
  all the modules in the schedule.  All accounting about processing
  of events by modules and paths is contained here or in object held
  by containment.

  The trigger results producer and product are generated and managed here.
  This class also manages endpaths and calls to endjob and beginjob.
  Endpaths are just treated as a simple list of modules that need to
  do processing of the event and do not participate in trigger path
  activities.

  This class requires the high-level process pset.  It expects a pset
  with name @trigger_paths to be present and also uses @process_name.
  If the high-level pset contains an "options" pset, then the
  following optional parameters can be present:
  bool wantSummary = true/false   # default false
  bool makeTriggerResults = true/false # default false

  wantSummary indicates whether or not the pass/fail/error stats
  for modules and paths should be printed at the end-of-job or not.

  If makeTriggerResults is true, then a TriggerResults object will
  always be inserted into the event for any schedule.  If this is false,
  then a TriggerResults object will be placed into the event only if
  there is an instance of a filter module present in any path.  If
  TriggerResults are needed, the producer is always the first module
  in the endpath.  The TriggerResultInserter is given a fixed label
  of "TriggerResults".

  The Schedule prints a warning if output modules are present in paths.
  They belong in endpaths.  The Schedule moves them to the endpath.

  Processing of an event happens by pushing the event through the Paths.
  The scheduler performs the reset() on each of the workers independent
  of the Path objects. 

  ------------------------

  About Paths:
  Paths fit into three categories:
  1) trigger paths that contribute directly to saved trigger bits
  2) nontrigger paths that do not contribute to saved trigger bits
  3) end paths
  The Schedule hold these paths in two data structures:
  1) main path list (both trigger/nontrigger paths maintained here)
  2) end path list

  Standard path processing (trigger/nontrigger) always precedes endpath
  processing.  The order of the trigger/nontrigger paths from the input
  specification is preserved in the main paths list.

  ------------------------

  The Schdule expects the following untracked PSet to be in the main
  process PSet (it is inserted automatically by the PSet builder).

  untracked PSet @trigger_paths = {
    vstring @paths = { p1, p2, p3, ... }
    vstring @end_paths = { e1, e2, e3, ... }
  }

  The first thing the scheduler does is to create a new @trigger_paths
  PSet that separates the trigger and nontrigger paths from the @paths.
  The new untracked PSet looks like this

  untracked PSet {
    vstring @paths = { p1, p2, p3, ... }     # all the regular paths
    vstring @trigger_paths = { p1, p2, ... } # subset of @paths
    vstring @end_paths = { e1, e2, e3, ... } # all the others
  }

  The @trigger_paths must be a subset of the @paths.
  The @trigger_paths are currently saved is the TriggerResults EDProduct.
  The number of trigger bits is the number of names in @trigger_paths.
  Each name in @trigger_paths has a corresponding trigger bit in the 
  TriggerResults.

*/

#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/WorkerInPath.h"
#include "FWCore/Framework/src/Path.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/src/WorkerRegistry.h"

#include "boost/shared_ptr.hpp"

#include <string>
#include <vector>
#include <set>

#include "FWCore/Framework/src/RunStopwatch.h"

namespace edm
{
  class UnscheduledCallProducer;
  class OutputWorker;
  class Schedule
  {
  public:
    typedef std::vector<std::string> vstring;
    typedef std::vector<Path> TrigPaths;
    typedef std::vector<Path> NonTrigPaths;
    typedef boost::shared_ptr<HLTGlobalStatus> TrigResPtr;
    typedef boost::shared_ptr<Worker> WorkerPtr;
    typedef boost::shared_ptr<ActivityRegistry> ActivityRegistryPtr;
    typedef std::set<Worker*> AllWorkers;
    typedef std::pair<int, OutputWorker const*> OneOutputWorker;
    typedef std::vector<OneOutputWorker> AllOutputWorkers;
    typedef std::vector<Worker*> Workers;
    typedef std::vector<WorkerInPath> PathWorkers;

    Schedule(ParameterSet const& processDesc,
	     edm::service::TriggerNamesService& tns,
	     WorkerRegistry& wregistry,
	     ProductRegistry& pregistry,
	     ActionTable& actions,
	     ActivityRegistryPtr areg);

    enum State { Ready=0, Running, Latched };

    void runOneEvent(EventPrincipal& eventPrincipal, 
		     EventSetup const& eventSetup,
		     BranchActionType const& branchActionType);

    void beginJob(EventSetup const&);
    void endJob();

    std::pair<double,double> timeCpuReal() const 
    {
      return std::pair<double,double>(stopwatch_->cpuTime(),stopwatch_->realTime());
    }

    /// Return a vector allowing const access to all the
    /// ModuleDescriptions for this Schedule.

    /// *** N.B. *** Ownership of the ModuleDescriptions is *not*
    /// *** passed to the caller. Do not call delete on these
    /// *** pointers!
    std::vector<ModuleDescription const*> getAllModuleDescriptions() const;

    /// Return the number of events this Schedule has tried to process
    /// (inclues both successes and failures, including failures due
    /// to exceptions during processing).
    int totalEvents() const
    {
      return total_events_;
    }

    /// Return the number of events which have been passed by one or
    /// more trigger paths.
    int totalEventsPassed() const
    {
      return total_passed_;
    }

    /// Return the number of events that have not passed any trigger.
    /// (N.B. totalEventsFailed() + totalEventsPassed() == totalEvents()
    int totalEventsFailed() const
    {
      return totalEvents() - totalEventsPassed();
    }

    /// Turn end_paths "off" if "active" is false;
    /// turn end_paths "on" if "active" is true.
    void enableEndPaths(bool active);

    /// Return true if end_paths are active, and false if they are
    /// inactive.
    bool endPathsEnabled() const;

    /// Return the trigger report information on paths,
    /// modules-in-path, modules-in-endpath, and modules.
    void getTriggerReport(TriggerReport& rep) const;      

    /// Return whether a module has reached its maximum count.
    bool const terminate() const;

  private:
    AllWorkers::const_iterator workersBegin() const 
    { return all_workers_.begin(); }

    AllWorkers::const_iterator workersEnd() const 
    { return all_workers_.end(); }

    AllWorkers::iterator workersBegin() 
    { return  all_workers_.begin(); }

    AllWorkers::iterator workersEnd() 
    { return all_workers_.end(); }

    void resetAll();
    bool runTriggerPaths(EventPrincipal&, EventSetup const&, BranchActionType const&);
    void runEndPaths(EventPrincipal&, EventSetup const&, BranchActionType const&);

    void setupOnDemandSystem(EventPrincipal& ep, EventSetup const& es);

    void fillWorkers(const std::string& name, PathWorkers& out);
    bool fillTrigPath(int bitpos,const std::string& name, TrigResPtr);
    void fillEndPath(int bitpos,const std::string& name);
    void handleWronglyPlacedModules();

    ParameterSet        pset_;
    WorkerRegistry*     worker_reg_;
    ProductRegistry*    prod_reg_;
    ActionTable*        act_table_;
    std::string         processName_;
    ParameterSet        trig_pset_;  // why is this kept?
    ActivityRegistryPtr act_reg_;

    State state_;
    vstring trig_name_list_;
    vstring path_name_list_;
    vstring               end_path_name_list_;
    std::set<std::string> trig_name_set_;

    TrigResPtr   results_;
    TrigResPtr   nontrig_results_;
    TrigResPtr   endpath_results_;

    WorkerPtr   results_inserter_;
    AllWorkers  all_workers_;
    AllOutputWorkers  all_output_workers_;
    TrigPaths   trig_paths_;
    TrigPaths   end_paths_;

    PathWorkers tmp_wrongly_placed_;

    bool                             wantSummary_;
    bool                             makeTriggerResults_;
    int                              total_events_;
    int                              total_passed_;
    RunStopwatch::StopwatchPointer   stopwatch_;

    boost::shared_ptr<UnscheduledCallProducer> unscheduled_;
    std::vector<boost::shared_ptr<Provenance> >     demandBranches_;

    volatile bool       endpathsAreActive_;
  };
}

#endif

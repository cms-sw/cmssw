#ifndef FWCore_Framework_Schedule_h
#define FWCore_Framework_Schedule_h

/*
  Author: Jim Kowalkowski  28-01-06

  $Id: Schedule.h,v 1.45 2008/05/12 18:14:07 wmtan Exp $

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

  This class requires the high-level process pset.  It uses @process_name.
  If the high-level pset contains an "options" pset, then the
  following optional parameter can be present:
  bool wantSummary = true/false   # default false

  wantSummary indicates whether or not the pass/fail/error stats
  for modules and paths should be printed at the end-of-job.

  A TriggerResults object will always be inserted into the event
  for any schedule.  The producer of the TriggerResults EDProduct
  is always the first module in the endpath.  The TriggerResultInserter
  is given a fixed label of "TriggerResults".

  The Schedule throws an exception if an output modules is present
  in a path.  It belongs in an endpath.

  The Schedule throws an exception if a filter is present
  in an endpath.  It belongs in a path.

  The Schedule issues a warning if a producer is present
  in an endpath.  It belongs in a path. 

  Processing of an event happens by pushing the event through the Paths.
  The scheduler performs the reset() on each of the workers independent
  of the Path objects. 

  ------------------------

  About Paths:
  Paths fit into two categories:
  1) trigger paths that contribute directly to saved trigger bits
  2) end paths
  The Schedule holds these paths in two data structures:
  1) main path list
  2) end path list

  Trigger path processing always precedes endpath processing.
  The order of the paths from the input configuration is
  preserved in the main paths list.

  ------------------------

  The Schedule uses the TriggerNamesService to get the names of the
  trigger paths and end paths. When a TriggerResults object is created
  the results are stored in the same order as the trigger names from
  TriggerNamesService.

*/

#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Common/interface/HLTGlobalStatus.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Framework/src/Path.h"
#include "FWCore/Framework/src/RunStopwatch.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/UnscheduledHandler.h"
#include "FWCore/Framework/src/Worker.h"

#include "boost/shared_ptr.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <set>

namespace edm {
  namespace service {
    class TriggerNamesService;
  }
  class ActivityRegistry;
  class EventSetup;
  class OutputWorker;
  class UnscheduledCallProducer;
  class RunStopwatch;
  class WorkerInPath;
  class WorkerRegistry;
  class Schedule {
  public:
    typedef std::vector<std::string> vstring;
    typedef std::vector<Path> TrigPaths;
    typedef std::vector<Path> NonTrigPaths;
    typedef boost::shared_ptr<HLTGlobalStatus> TrigResPtr;
    typedef boost::shared_ptr<Worker> WorkerPtr;
    typedef boost::shared_ptr<ActivityRegistry> ActivityRegistryPtr;
    typedef std::vector<Worker*> AllWorkers;
    typedef std::vector<OutputWorker*> AllOutputWorkers;

    typedef std::vector<Worker*> Workers;

    typedef std::vector<WorkerInPath> PathWorkers;

    Schedule(ParameterSet const& processDesc,
	     edm::service::TriggerNamesService& tns,
	     WorkerRegistry& wregistry,
	     ProductRegistry& pregistry,
	     ActionTable& actions,
	     ActivityRegistryPtr areg);

    enum State { Ready=0, Running, Latched };

    template <typename T>
    void runOneEvent(T& principal, 
		     EventSetup const& eventSetup,
		     BranchActionType const& branchActionType);

    void beginJob(EventSetup const&);
    void endJob();

    // Write the luminosity block
    void writeLumi(LuminosityBlockPrincipal const& lbp);

    // Write the run
    void writeRun(RunPrincipal const& rp);

    // Call closeFile() on all OutputModules.
    void closeOutputFiles();

    // Call openNewFileIfNeeded() on all OutputModules
    void openNewOutputFilesIfNeeded();

    // Call openFiles() on all OutputModules
    void openOutputFiles(FileBlock & fb);

    // Call respondToOpenInputFile() on all Modules
    void respondToOpenInputFile(FileBlock const& fb);

    // Call respondToCloseInputFile() on all Modules
    void respondToCloseInputFile(FileBlock const& fb);

    // Call respondToOpenOutputFiles() on all Modules
    void respondToOpenOutputFiles(FileBlock const& fb);

    // Call respondToCloseOutputFiles() on all Modules
    void respondToCloseOutputFiles(FileBlock const& fb);

    // Call shouldWeCloseFile() on all OutputModules.
    bool shouldWeCloseOutput() const;

    bool anyOutputModules() const {return !all_output_workers_.empty();}

    std::pair<double,double> timeCpuReal() const {
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
    int totalEvents() const {
      return total_events_;
    }

    /// Return the number of events which have been passed by one or
    /// more trigger paths.
    int totalEventsPassed() const {
      return total_passed_;
    }

    /// Return the number of events that have not passed any trigger.
    /// (N.B. totalEventsFailed() + totalEventsPassed() == totalEvents()
    int totalEventsFailed() const {
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

    ///  Clear all the counters in the trigger report.
    void clearCounters();

    class CallPrePost {
    public:
      CallPrePost(ActivityRegistry* a, EventPrincipal* ep, EventSetup const* es);
      CallPrePost(ActivityRegistry* a, LuminosityBlockPrincipal* ep, EventSetup const* es) :
        a_(0), ep_(0), es_(0) {}
      CallPrePost(ActivityRegistry* a, RunPrincipal* ep, EventSetup const* es) :
        a_(0), ep_(0), es_(0) {}
      ~CallPrePost(); 

    private:
      // We own none of these resources.
      ActivityRegistry*  a_;
      EventPrincipal*    ep_;
      EventSetup const*  es_;
    };

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

    template <typename T>
    bool runTriggerPaths(T&, EventSetup const&, BranchActionType const&);

    template <typename T>
    void runEndPaths(T&, EventSetup const&, BranchActionType const&);

    void setupOnDemandSystem(EventPrincipal& principal, EventSetup const& es);

    void reportSkipped(EventPrincipal const& ep) const;
    void reportSkipped(LuminosityBlockPrincipal const&) const {}
    void reportSkipped(RunPrincipal const&) const {}

    void fillWorkers(std::string const& name, PathWorkers& out);
    void fillTrigPath(int bitpos, std::string const& name, TrigResPtr);
    void fillEndPath(int bitpos, std::string const& name);

    void limitOutput();

    void addToAllWorkers(Worker* w);

    ParameterSet        pset_;
    WorkerRegistry*     worker_reg_;
    ProductRegistry*    prod_reg_;
    ActionTable*        act_table_;
    std::string         processName_;
    ActivityRegistryPtr act_reg_;

    State state_;
    vstring trig_name_list_;
    vstring               end_path_name_list_;

    TrigResPtr   results_;
    TrigResPtr   endpath_results_;

    WorkerPtr                results_inserter_;
    AllWorkers               all_workers_;
    AllOutputWorkers         all_output_workers_;
    TrigPaths                trig_paths_;
    TrigPaths                end_paths_;

    bool                             wantSummary_;
    int                              total_events_;
    int                              total_passed_;
    RunStopwatch::StopwatchPointer   stopwatch_;

    boost::shared_ptr<UnscheduledCallProducer> unscheduled_;
    std::vector<boost::shared_ptr<ConstBranchDescription const> >  demandBranches_;

    volatile bool       endpathsAreActive_;
  };

  // -----------------------------
  // run_one_event is a functor that has bound a specific
  // Principal and Event Setup, and can be called with a Path, to
  // execute Path::runOneEvent for that event
    
  template <typename T>
  class run_one_event {
  public:
    typedef void result_type;
    run_one_event(T& principal, EventSetup const& setup, BranchActionType const& branchActionType) :
      ep(principal), es(setup), bat(branchActionType) {};

      void operator()(Path& p) {p.runOneEvent(ep, es, bat);}

  private:      
    T&   ep;
    EventSetup const& es;
    BranchActionType const& bat;
  };

  class UnscheduledCallProducer : public UnscheduledHandler {
  public:
    UnscheduledCallProducer() : UnscheduledHandler(), labelToWorkers_() {}
    void addWorker(Worker* aWorker) {
      assert(0 != aWorker);
      labelToWorkers_[aWorker->description().moduleLabel_]=aWorker;
    }
  private:
    virtual bool tryToFillImpl(std::string const& moduleLabel,
			       EventPrincipal& event,
			       const EventSetup& eventSetup) {
      std::map<std::string, Worker*>::const_iterator itFound =
        labelToWorkers_.find(moduleLabel);
      if(itFound != labelToWorkers_.end()) {
	  // Unscheduled reconstruction has no accepted definition
	  // (yet) of the "current path". We indicate this by passing
	  // a null pointer as the CurrentProcessingContext.
	  itFound->second->doWork(event, eventSetup, BranchActionEvent, 0);
	  return true;
      }
      return false;
    }
    std::map<std::string, Worker*> labelToWorkers_;
  };

  void
  inline
  Schedule::reportSkipped(EventPrincipal const& ep) const {
    Service<JobReport> reportSvc;
    reportSvc->reportSkippedEvent(ep.id().run(), ep.id().event());
  }
  
  template <typename T>
  void
  Schedule::runOneEvent(T& ep, EventSetup const& es, BranchActionType const& bat) {
    this->resetAll();
    state_ = Running;

    bool const isEvent = (bat == BranchActionEvent);

    // A RunStopwatch, but only if we are processing an event.
    std::auto_ptr<RunStopwatch> stopwatch(isEvent ? new RunStopwatch(stopwatch_) : 0);

    if (isEvent) {
      ++total_events_;
      setupOnDemandSystem(dynamic_cast<EventPrincipal &>(ep), es);
    }
    try {
      //If the CallPrePost object is used, it must live for the entire time the event is
      // being processed
      std::auto_ptr<CallPrePost> sentry;
      try {
        if (isEvent) {
 	  sentry = std::auto_ptr<CallPrePost>(new CallPrePost(act_reg_.get(), &ep, &es));
        }
        if (runTriggerPaths(ep, es, bat)) {
	  if (isEvent) ++total_passed_;
        }
        state_ = Latched;
	
        if (results_inserter_.get()) results_inserter_->doWork(ep, es, bat, 0);
      }
      catch(cms::Exception& e) {
        actions::ActionCodes action = (isEvent ? act_table_->find(e.rootCause()) : actions::Rethrow);
        assert (action != actions::IgnoreCompletely);
        assert (action != actions::FailPath);
        assert (action != actions::FailModule);
	if (action == actions::SkipEvent) {
            LogWarning(e.category())
              << "an exception occurred and all paths for the event are being skipped: \n"
              << e.what();
        } else {
 	  throw;
        }
      }

      if (endpathsAreActive_) runEndPaths(ep, es, bat);
    }
    catch(cms::Exception& ex) {
      actions::ActionCodes action = act_table_->find(ex.rootCause());
      assert (action != actions::SkipEvent);
      assert (action != actions::FailPath);
      assert (action != actions::FailModule);
      switch(action) {
      case actions::IgnoreCompletely: {
  	LogWarning(ex.category())
  	  << "exception being ignored for current event:\n"
  	  << ex.what();
  	break;
      }
      default: {
        state_ = Ready;
        throw edm::Exception(errors::EventProcessorFailure,
			     "EventProcessingStopped",ex)
	  << "an exception occurred during current event processing\n";
      }
      }
    }
    catch(...) {
      LogError("PassingThrough")
	<< "an exception occurred during current event processing\n";
      state_ = Ready;
      throw;
    }

    // next thing probably is not needed, the product insertion code clears it
    state_ = Ready;

  }

  template <typename T>
  bool
  Schedule::runTriggerPaths(T& ep, EventSetup const& es, BranchActionType const& bat) {
    for_all(trig_paths_, run_one_event<T>(ep, es, bat));
    return results_->accept();
  }

  template <typename T>
  void
  Schedule::runEndPaths(T& ep, EventSetup const& es, BranchActionType const& bat) {
    // Note there is no state-checking safety controlling the
    // activation/deactivation of endpaths.
    for_all(end_paths_, run_one_event<T>(ep, es, bat));

    // We could get rid of the functor run_one_event if we used
    // boost::lambda, but the use of lambda with member functions
    // which take multiple arguments, by both non-const and const
    // reference, seems much more obscure...
    //
    // using namespace boost::lambda;
    // for_all(end_paths_,
    //          bind(&Path::runOneEvent, 
    //               boost::lambda::_1, // qualification to avoid ambiguity
    //               var(ep),           //  pass by reference (not copy)
    //               constant_ref(es))); // pass by const-reference (not copy)
  }
  
}

#endif

#ifndef FWCore_Framework_Schedule_h
#define FWCore_Framework_Schedule_h

/*
  Author: Jim Kowalkowski  28-01-06

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

#include "DataFormats/Common/interface/HLTGlobalStatus.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/ExceptionActions.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/ExceptionHelpers.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "FWCore/Framework/interface/WorkerManager.h"
#include "FWCore/Framework/src/Path.h"
#include "FWCore/Framework/src/RunStopwatch.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/WorkerRegistry.h"
#include "FWCore/Framework/src/EarlyDeleteHelper.h"
#include "FWCore/MessageLogger/interface/ExceptionMessages.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "boost/shared_ptr.hpp"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <sstream>

namespace edm {

  namespace {
    template <typename T>
    class ScheduleSignalSentry {
    public:
      ScheduleSignalSentry(ActivityRegistry* a, typename T::MyPrincipal* principal, EventSetup const* es, typename T::Context const* context) :
        a_(a), principal_(principal), es_(es), context_(context) {
        if (a_) T::preScheduleSignal(a_, principal_, context_);
      }
      ~ScheduleSignalSentry() {
        if (a_) if (principal_) T::postScheduleSignal(a_, principal_, es_, context_);
      }

    private:
      // We own none of these resources.
      ActivityRegistry* a_;
      typename T::MyPrincipal* principal_;
      EventSetup const* es_;
      typename T::Context const* context_;
    };
  }

  namespace service {
    class TriggerNamesService;
  }
  class ActivityRegistry;
  class BranchIDListHelper;
  class EventSetup;
  class ExceptionCollector;
  class OutputModuleCommunicator;
  class ProcessContext;
  class RunStopwatch;
  class WorkerInPath;
  class PreallocationConfiguration;
  
  class Schedule {
  public:
    typedef std::vector<std::string> vstring;
    typedef std::vector<Path> TrigPaths;
    typedef std::vector<Path> NonTrigPaths;
    typedef boost::shared_ptr<HLTGlobalStatus> TrigResPtr;
    typedef boost::shared_ptr<Worker> WorkerPtr;
    typedef std::vector<Worker*> AllWorkers;
    typedef std::vector<boost::shared_ptr<OutputModuleCommunicator>> AllOutputModuleCommunicators;

    typedef std::vector<Worker*> Workers;

    typedef std::vector<WorkerInPath> PathWorkers;

    Schedule(ParameterSet& proc_pset,
             service::TriggerNamesService& tns,
             ProductRegistry& pregistry,
             BranchIDListHelper& branchIDListHelper,
             ExceptionToActionTable const& actions,
             boost::shared_ptr<ActivityRegistry> areg,
             boost::shared_ptr<ProcessConfiguration> processConfiguration,
             const ParameterSet* subProcPSet,
             PreallocationConfiguration const& config,
             ProcessContext const* processContext);

    enum State { Ready = 0, Running, Latched };

    template <typename T>
    void processOneEvent(typename T::MyPrincipal& principal,
                         EventSetup const& eventSetup,
                         bool cleaningUpAfterException = false);

    template <typename T>
    void processOneGlobal(typename T::MyPrincipal& principal,
                          EventSetup const& eventSetup,
                          bool cleaningUpAfterException = false);

    template <typename T>
    void processOneStream(typename T::MyPrincipal& principal,
                          EventSetup const& eventSetup,
                          bool cleaningUpAfterException = false);

    void beginJob(ProductRegistry const&);
    void endJob(ExceptionCollector & collector);
    
    void beginStream();
    void endStream();

    // Write the luminosity block
    void writeLumi(LuminosityBlockPrincipal const& lbp, ProcessContext const*);

    // Write the run
    void writeRun(RunPrincipal const& rp, ProcessContext const*);

    // Call closeFile() on all OutputModules.
    void closeOutputFiles();

    // Call openNewFileIfNeeded() on all OutputModules
    void openNewOutputFilesIfNeeded();

    // Call openFiles() on all OutputModules
    void openOutputFiles(FileBlock& fb);

    // Call respondToOpenInputFile() on all Modules
    void respondToOpenInputFile(FileBlock const& fb);

    // Call respondToCloseInputFile() on all Modules
    void respondToCloseInputFile(FileBlock const& fb);

    // Call shouldWeCloseFile() on all OutputModules.
    bool shouldWeCloseOutput() const;

    void preForkReleaseResources();
    void postForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren);

    StreamID streamID() const { return streamID_; }
    
    std::pair<double, double> timeCpuReal() const {
      return std::pair<double, double>(stopwatch_->cpuTime(), stopwatch_->realTime());
    }

    /// Return a vector allowing const access to all the
    /// ModuleDescriptions for this Schedule.

    /// *** N.B. *** Ownership of the ModuleDescriptions is *not*
    /// *** passed to the caller. Do not call delete on these
    /// *** pointers!
    std::vector<ModuleDescription const*> getAllModuleDescriptions() const;

    ///adds to oLabelsToFill the labels for all paths in the process
    void availablePaths(std::vector<std::string>& oLabelsToFill) const;

    ///adds to oLabelsToFill in execution order the labels of all modules in path iPathLabel
    void modulesInPath(std::string const& iPathLabel,
                       std::vector<std::string>& oLabelsToFill) const;

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

    /// Return whether each output module has reached its maximum count.
    bool terminate() const;

    ///  Clear all the counters in the trigger report.
    void clearCounters();

    /// clone the type of module with label iLabel but configure with iPSet.
    /// Returns true if successful.
    bool changeModule(std::string const& iLabel, ParameterSet const& iPSet);

    /// returns the collection of pointers to workers
    AllWorkers const& allWorkers() const {
      return workerManager_.allWorkers();
    }

    /// returns the action table
    ExceptionToActionTable const& actionTable() const {
      return workerManager_.actionTable();
    }

  private:

    void resetAll();

    template <typename T>
    bool runTriggerPaths(typename T::MyPrincipal&, EventSetup const&, typename T::Context const*);

    template <typename T>
    void runEndPaths(typename T::MyPrincipal&, EventSetup const&, typename T::Context const*);

    void reportSkipped(EventPrincipal const& ep) const;
    void reportSkipped(LuminosityBlockPrincipal const&) const {}
    void reportSkipped(RunPrincipal const&) const {}

    void reduceParameterSet(ParameterSet& proc_pset,
                            vstring& modulesInConfig,
                            std::set<std::string> const& modulesInConfigSet,
                            vstring& labelsOnTriggerPaths,
                            vstring& shouldBeUsedLabels,
                            std::map<std::string, std::vector<std::pair<std::string, int> > >& outputModulePathPositions);

    void fillWorkers(ParameterSet& proc_pset,
                     ProductRegistry& preg,
                     boost::shared_ptr<ProcessConfiguration const> processConfiguration,
                     std::string const& name, bool ignoreFilters, PathWorkers& out,
                     vstring* labelsOnPaths);
    void fillTrigPath(ParameterSet& proc_pset,
                      ProductRegistry& preg,
                      boost::shared_ptr<ProcessConfiguration const> processConfiguration,
                      int bitpos, std::string const& name, TrigResPtr,
                      vstring* labelsOnTriggerPaths);
    void fillEndPath(ParameterSet& proc_pset,
                     ProductRegistry& preg,
                     boost::shared_ptr<ProcessConfiguration const> processConfiguration,
                     int bitpos, std::string const& name);

    void limitOutput(ParameterSet const& proc_pset, BranchIDLists const& branchIDLists);

    void addToAllWorkers(Worker* w);
    
    void resetEarlyDelete();
    void initializeEarlyDelete(edm::ParameterSet const& opts,
                               edm::ProductRegistry const& preg, 
                               edm::ParameterSet const* subProcPSet);

    WorkerManager            workerManager_;
    boost::shared_ptr<ActivityRegistry>           actReg_;

    State                    state_;
    vstring                  trig_name_list_;
    vstring                  end_path_name_list_;

    TrigResPtr               results_;
    TrigResPtr               endpath_results_;

    WorkerPtr                results_inserter_;
    AllOutputModuleCommunicators         all_output_communicators_;
    TrigPaths                trig_paths_;
    TrigPaths                end_paths_;
    std::vector<int>         empty_trig_paths_;
    vstring                  empty_trig_path_names_;

    //For each branch that has been marked for early deletion
    // keep track of how many modules are left that read this data but have
    // not yet been run in this event
    std::vector<std::pair<BranchID,unsigned int>> earlyDeleteBranchToCount_;
    //NOTE the following is effectively internal data for each EarlyDeleteHelper
    // but putting it into one vector makes for better allocation as well as
    // faster iteration when used to reset the earlyDeleteBranchToCount_
    // Each EarlyDeleteHelper hold a begin and end range into this vector. The values
    // of this vector correspond to indexes into earlyDeleteBranchToCount_ so 
    // tell which EarlyDeleteHelper is associated with which BranchIDs.
    std::vector<unsigned int> earlyDeleteHelperToBranchIndicies_;
    //There is one EarlyDeleteHelper per Module which are reading data that
    // has been marked for early deletion
    std::vector<EarlyDeleteHelper> earlyDeleteHelpers_;

    bool                           wantSummary_;
    int                            total_events_;
    int                            total_passed_;
    RunStopwatch::StopwatchPointer stopwatch_;

    StreamID                streamID_;
    StreamContext           streamContext_;
    volatile bool           endpathsAreActive_;
  };

  void
  inline
  Schedule::reportSkipped(EventPrincipal const& ep) const {
    Service<JobReport> reportSvc;
    reportSvc->reportSkippedEvent(ep.id().run(), ep.id().event());
  }

  template <typename T>
  void Schedule::processOneEvent(typename T::MyPrincipal& ep,
                                 EventSetup const& es,
                                 bool cleaningUpAfterException) {
    this->resetAll();
    for (int empty_trig_path : empty_trig_paths_) {
      results_->at(empty_trig_path) = HLTPathStatus(hlt::Pass, 0);
    }
    state_ = Running;

    T::setStreamContext(streamContext_, ep);
    ScheduleSignalSentry<T> sentry(actReg_.get(), &ep, &es, &streamContext_);

    // A RunStopwatch, but only if we are processing an event.
    RunStopwatch stopwatch(stopwatch_);

    // This call takes care of the unscheduled processing.
    workerManager_.processOneOccurrence<T>(ep, es, streamID_, &streamContext_, &streamContext_, cleaningUpAfterException);

    ++total_events_;
    try {
      try {
        try {
          if (runTriggerPaths<T>(ep, es, &streamContext_)) {
            ++total_passed_;
          }
          state_ = Latched;
        }
        catch(cms::Exception& e) {
          exception_actions::ActionCodes action = actionTable().find(e.category());
          assert (action != exception_actions::IgnoreCompletely);
          assert (action != exception_actions::FailPath);
          if (action == exception_actions::SkipEvent) {
            edm::printCmsExceptionWarning("SkipEvent", e);
          } else {
            throw;
          }
        }

        try {
          CPUTimer timer;
          ParentContext parentContext(&streamContext_);
          if (results_inserter_.get()) results_inserter_->doWork<T>(ep, es, nullptr, &timer,streamID_, parentContext, &streamContext_);
        }
        catch (cms::Exception & ex) {
          if (T::isEvent_) {
            ex.addContext("Calling produce method for module TriggerResultInserter");
          }
	  std::ostringstream ost;
          ost << "Processing " << ep.id();
          ex.addContext(ost.str());
          throw;
        }

        if (endpathsAreActive_) runEndPaths<T>(ep, es, &streamContext_);
        resetEarlyDelete();
      }
      catch (cms::Exception& e) { throw; }
      catch(std::bad_alloc& bda) { convertException::badAllocToEDM(); }
      catch (std::exception& e) { convertException::stdToEDM(e); }
      catch(std::string& s) { convertException::stringToEDM(s); }
      catch(char const* c) { convertException::charPtrToEDM(c); }
      catch (...) { convertException::unknownToEDM(); }
    }
    catch(cms::Exception& ex) {
      if (ex.context().empty()) {
        addContextAndPrintException("Calling function Schedule::processOneEvent", ex, cleaningUpAfterException);
      } else {
        addContextAndPrintException("", ex, cleaningUpAfterException);
      }
      state_ = Ready;
      throw;
    }
    // next thing probably is not needed, the product insertion code clears it
    state_ = Ready;
  }

  template <typename T>
  void Schedule::processOneStream(typename T::MyPrincipal& ep,
                                  EventSetup const& es,
                                  bool cleaningUpAfterException) {
    this->resetAll();
    for (int empty_trig_path : empty_trig_paths_) {
      results_->at(empty_trig_path) = HLTPathStatus(hlt::Pass, 0);
    }
    state_ = Running;

    T::setStreamContext(streamContext_, ep);
    ScheduleSignalSentry<T> sentry(actReg_.get(), &ep, &es, &streamContext_);

    // This call takes care of the unscheduled processing.
    workerManager_.processOneOccurrence<T>(ep, es, streamID_, &streamContext_, &streamContext_, cleaningUpAfterException);

    try {
      try {
        runTriggerPaths<T>(ep, es, &streamContext_);
        state_ = Latched;

        if (endpathsAreActive_) runEndPaths<T>(ep, es, &streamContext_);
      }
      catch (cms::Exception& e) { throw; }
      catch(std::bad_alloc& bda) { convertException::badAllocToEDM(); }
      catch (std::exception& e) { convertException::stdToEDM(e); }
      catch(std::string& s) { convertException::stringToEDM(s); }
      catch(char const* c) { convertException::charPtrToEDM(c); }
      catch (...) { convertException::unknownToEDM(); }
    }
    catch(cms::Exception& ex) {
      if (ex.context().empty()) {
        addContextAndPrintException("Calling function Schedule::processOneStream", ex, cleaningUpAfterException);
      } else {
        addContextAndPrintException("", ex, cleaningUpAfterException);
      }
      state_ = Ready;
      throw;
    }
    // next thing probably is not needed, the product insertion code clears it
    state_ = Ready;
  }
  template <typename T>
  void
  Schedule::processOneGlobal(typename T::MyPrincipal& ep,
                                 EventSetup const& es,
                                 bool cleaningUpAfterException) {
    this->resetAll();
    for (int empty_trig_path : empty_trig_paths_) {
      results_->at(empty_trig_path) = HLTPathStatus(hlt::Pass, 0);
    }
    state_ = Running;

    GlobalContext globalContext = T::makeGlobalContext(ep, streamContext_.processContext());

    ScheduleSignalSentry<T> sentry(actReg_.get(), &ep, &es, &globalContext);

    // This call takes care of the unscheduled processing.
    workerManager_.processOneOccurrence<T>(ep, es, streamID_, &globalContext, &globalContext, cleaningUpAfterException);

    try {
      try {
        runTriggerPaths<T>(ep, es, &globalContext);
        state_ = Latched;

        if (endpathsAreActive_) runEndPaths<T>(ep, es, &globalContext);
      }
      catch (cms::Exception& e) { throw; }
      catch(std::bad_alloc& bda) { convertException::badAllocToEDM(); }
      catch (std::exception& e) { convertException::stdToEDM(e); }
      catch(std::string& s) { convertException::stringToEDM(s); }
      catch(char const* c) { convertException::charPtrToEDM(c); }
      catch (...) { convertException::unknownToEDM(); }
    }
    catch(cms::Exception& ex) {
      if (ex.context().empty()) {
        addContextAndPrintException("Calling function Schedule::processOneGlobal", ex, cleaningUpAfterException);
      } else {
        addContextAndPrintException("", ex, cleaningUpAfterException);
      }
      state_ = Ready;
      throw;
    }
    // next thing probably is not needed, the product insertion code clears it
    state_ = Ready;
  }

  template <typename T>
  bool
  Schedule::runTriggerPaths(typename T::MyPrincipal& ep, EventSetup const& es, typename T::Context const* context) {
    for(auto& p : trig_paths_) {
      p.processOneOccurrence<T>(ep, es, streamID_, context);
    }
    return results_->accept();
  }

  template <typename T>
  void
  Schedule::runEndPaths(typename T::MyPrincipal& ep, EventSetup const& es, typename T::Context const* context) {
    // Note there is no state-checking safety controlling the
    // activation/deactivation of endpaths.
    for(auto& p : end_paths_) {
      p.processOneOccurrence<T>(ep, es, streamID_, context);
    }
  }
}

#endif

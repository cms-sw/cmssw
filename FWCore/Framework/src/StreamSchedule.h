#ifndef FWCore_Framework_StreamSchedule_h
#define FWCore_Framework_StreamSchedule_h

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
  The StreamSchedule holds these paths in two data structures:
  1) main path list
  2) end path list

  Trigger path processing always precedes endpath processing.
  The order of the paths from the input configuration is
  preserved in the main paths list.

  ------------------------

  The StreamSchedule uses the TriggerNamesService to get the names of the
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
#include "FWCore/Framework/interface/UnscheduledCallProducer.h"
#include "FWCore/Framework/interface/WorkerManager.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/src/Path.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/WorkerRegistry.h"
#include "FWCore/Framework/src/EarlyDeleteHelper.h"
#include "FWCore/MessageLogger/interface/ExceptionMessages.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <sstream>

namespace edm {

  class ActivityRegistry;
  class BranchIDListHelper;
  class EventSetup;
  class ExceptionCollector;
  class OutputModuleCommunicator;
  class ProcessContext;
  class UnscheduledCallProducer;
  class WorkerInPath;
  class ModuleRegistry;
  class TriggerResultInserter;
  class PreallocationConfiguration;

  namespace service {
    class TriggerNamesService;
  }

  namespace {
    template <typename T>
    class StreamScheduleSignalSentry {
    public:
      StreamScheduleSignalSentry(ActivityRegistry* a, typename T::Context const* context) :
        a_(a), context_(context), allowThrow_(false) {
        if (a_) T::preScheduleSignal(a_, context_);
      }
      ~StreamScheduleSignalSentry() noexcept(false) {
        try {
          if (a_) { T::postScheduleSignal(a_, context_); }
        } catch(...) {
          if(allowThrow_) {throw;}
        }
      }
      
      void allowThrow() {
        allowThrow_ = true;
      }

    private:
      // We own none of these resources.
      ActivityRegistry* a_;
      typename T::Context const* context_;
      bool allowThrow_;
    };
  }
  
  class StreamSchedule {
  public:
    typedef std::vector<std::string> vstring;
    typedef std::vector<Path> TrigPaths;
    typedef std::vector<Path> NonTrigPaths;
    typedef std::shared_ptr<HLTGlobalStatus> TrigResPtr;
    typedef std::shared_ptr<Worker> WorkerPtr;
    typedef std::vector<Worker*> AllWorkers;
    typedef std::vector<std::shared_ptr<OutputModuleCommunicator> > AllOutputModuleCommunicators;

    typedef std::vector<Worker*> Workers;

    typedef std::vector<WorkerInPath> PathWorkers;

    StreamSchedule(TriggerResultInserter* inserter,
                   std::shared_ptr<ModuleRegistry>,
                   ParameterSet& proc_pset,
                   service::TriggerNamesService& tns,
                   PreallocationConfiguration const& prealloc,
                   ProductRegistry& pregistry,
                   BranchIDListHelper& branchIDListHelper,
                   ExceptionToActionTable const& actions,
                   std::shared_ptr<ActivityRegistry> areg,
                   std::shared_ptr<ProcessConfiguration> processConfiguration,
                   bool allowEarlyDelete,
                   StreamID streamID,
                   ProcessContext const* processContext);
    
    StreamSchedule(StreamSchedule const&) = delete;

    template <typename T>
    void processOneEvent(typename T::MyPrincipal& principal,
                         EventSetup const& eventSetup,
                         bool cleaningUpAfterException = false);

    template <typename T>
    void processOneStream(typename T::MyPrincipal& principal,
                          EventSetup const& eventSetup,
                          bool cleaningUpAfterException = false);

    void beginStream();
    void endStream();

    StreamID streamID() const { return streamID_; }
    
    /// Return a vector allowing const access to all the
    /// ModuleDescriptions for this StreamSchedule.

    /// *** N.B. *** Ownership of the ModuleDescriptions is *not*
    /// *** passed to the caller. Do not call delete on these
    /// *** pointers!
    std::vector<ModuleDescription const*> getAllModuleDescriptions() const;

    ///adds to oLabelsToFill the labels for all paths in the process
    void availablePaths(std::vector<std::string>& oLabelsToFill) const;

    ///adds to oLabelsToFill in execution order the labels of all modules in path iPathLabel
    void modulesInPath(std::string const& iPathLabel,
                       std::vector<std::string>& oLabelsToFill) const;

    /// Return the number of events this StreamSchedule has tried to process
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

    ///  Clear all the counters in the trigger report.
    void clearCounters();

    /// clone the type of module with label iLabel but configure with iPSet.
    void replaceModule(maker::ModuleHolder* iMod, std::string const& iLabel);

    /// returns the collection of pointers to workers
    AllWorkers const& allWorkers() const {
      return workerManager_.allWorkers();
    }
    
    unsigned int numberOfUnscheduledModules() const {
      return number_of_unscheduled_modules_;
    }
    
  private:
    //Sentry class to only send a signal if an
    // exception occurs. An exception is identified
    // by the destructor being called without first
    // calling completedSuccessfully().
    class SendTerminationSignalIfException {
    public:
      SendTerminationSignalIfException(edm::ActivityRegistry* iReg, edm::StreamContext const* iContext):
      reg_(iReg),
      context_(iContext){}
      ~SendTerminationSignalIfException() {
        if(reg_) {
          reg_->preStreamEarlyTerminationSignal_(*context_,TerminationOrigin::ExceptionFromThisContext);
        }
      }
      void completedSuccessfully() {
        reg_ = nullptr;
      }
    private:
      edm::ActivityRegistry* reg_;
      StreamContext const* context_;
    };

    /// returns the action table
    ExceptionToActionTable const& actionTable() const {
      return workerManager_.actionTable();
    }
    

    void resetAll();

    template <typename T>
    bool runTriggerPaths(typename T::MyPrincipal&, EventSetup const&, typename T::Context const*);

    template <typename T>
    void runEndPaths(typename T::MyPrincipal&, EventSetup const&, typename T::Context const*);

    void reportSkipped(EventPrincipal const& ep) const;

    void fillWorkers(ParameterSet& proc_pset,
                     ProductRegistry& preg,
                     PreallocationConfiguration const* prealloc,
                     std::shared_ptr<ProcessConfiguration const> processConfiguration,
                     std::string const& name, bool ignoreFilters, PathWorkers& out,
                     vstring* labelsOnPaths);
    void fillTrigPath(ParameterSet& proc_pset,
                      ProductRegistry& preg,
                      PreallocationConfiguration const* prealloc,
                      std::shared_ptr<ProcessConfiguration const> processConfiguration,
                      int bitpos, std::string const& name, TrigResPtr,
                      vstring* labelsOnTriggerPaths);
    void fillEndPath(ParameterSet& proc_pset,
                     ProductRegistry& preg,
                     PreallocationConfiguration const* prealloc,
                     std::shared_ptr<ProcessConfiguration const> processConfiguration,
                     int bitpos, std::string const& name);

    void addToAllWorkers(Worker* w);
    
    void resetEarlyDelete();
    void initializeEarlyDelete(ModuleRegistry & modReg,
                               edm::ParameterSet const& opts,
                               edm::ProductRegistry const& preg, 
                               bool allowEarlyDelete);

    WorkerManager            workerManager_;
    std::shared_ptr<ActivityRegistry>           actReg_;

    vstring                  trig_name_list_;
    vstring                  end_path_name_list_;

    TrigResPtr               results_;

    WorkerPtr                results_inserter_;
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

    int                            total_events_;
    int                            total_passed_;
    unsigned int                   number_of_unscheduled_modules_;
    
    StreamID                streamID_;
    StreamContext           streamContext_;
    volatile bool           endpathsAreActive_;
  };

  void
  inline
  StreamSchedule::reportSkipped(EventPrincipal const& ep) const {
    Service<JobReport> reportSvc;
    reportSvc->reportSkippedEvent(ep.id().run(), ep.id().event());
  }

  template <typename T>
  void StreamSchedule::processOneEvent(typename T::MyPrincipal& ep,
                                 EventSetup const& es,
                                 bool cleaningUpAfterException) {
    this->resetAll();
    for (int empty_trig_path : empty_trig_paths_) {
      results_->at(empty_trig_path) = HLTPathStatus(hlt::Pass, 0);
    }

    T::setStreamContext(streamContext_, ep);
    StreamScheduleSignalSentry<T> sentry(actReg_.get(), &streamContext_);

    SendTerminationSignalIfException terminationSentry(actReg_.get(), &streamContext_);
    // This call takes care of the unscheduled processing.
    workerManager_.processOneOccurrence<T>(ep, es, streamID_, &streamContext_, &streamContext_, cleaningUpAfterException);

    ++total_events_;
    try {
      convertException::wrap([&]() {
        try {
          if (runTriggerPaths<T>(ep, es, &streamContext_)) {
            ++total_passed_;
          }
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
          ParentContext parentContext(&streamContext_);
          if (results_inserter_.get()) results_inserter_->doWork<T>(ep, es, streamID_, parentContext, &streamContext_);
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
      });
    }
    catch(cms::Exception& ex) {
      if (ex.context().empty()) {
        addContextAndPrintException("Calling function StreamSchedule::processOneEvent", ex, cleaningUpAfterException);
      } else {
        addContextAndPrintException("", ex, cleaningUpAfterException);
      }
      throw;
    }
    terminationSentry.completedSuccessfully();
    
    //If we got here no other exception has happened so we can propogate any Service related exceptions
    sentry.allowThrow();
  }

  template <typename T>
  void StreamSchedule::processOneStream(typename T::MyPrincipal& ep,
                                  EventSetup const& es,
                                  bool cleaningUpAfterException) {
    this->resetAll();

    T::setStreamContext(streamContext_, ep);
    StreamScheduleSignalSentry<T> sentry(actReg_.get(), &streamContext_);

    SendTerminationSignalIfException terminationSentry(actReg_.get(), &streamContext_);

    // This call takes care of the unscheduled processing.
    workerManager_.processOneOccurrence<T>(ep, es, streamID_, &streamContext_, &streamContext_, cleaningUpAfterException);

    try {
      convertException::wrap([&]() {
        runTriggerPaths<T>(ep, es, &streamContext_);

        if (endpathsAreActive_) runEndPaths<T>(ep, es, &streamContext_);
      });
    }
    catch(cms::Exception& ex) {
      if (ex.context().empty()) {
        addContextAndPrintException("Calling function StreamSchedule::processOneStream", ex, cleaningUpAfterException);
      } else {
        addContextAndPrintException("", ex, cleaningUpAfterException);
      }
      throw;
    }
    terminationSentry.completedSuccessfully();

    //If we got here no other exception has happened so we can propogate any Service related exceptions
    sentry.allowThrow();
  }

  template <typename T>
  bool
  StreamSchedule::runTriggerPaths(typename T::MyPrincipal& ep, EventSetup const& es, typename T::Context const* context) {
    for(auto& p : trig_paths_) {
      p.processOneOccurrence<T>(ep, es, streamID_, context);
    }
    return results_->accept();
  }

  template <typename T>
  void
  StreamSchedule::runEndPaths(typename T::MyPrincipal& ep, EventSetup const& es, typename T::Context const* context) {
    // Note there is no state-checking safety controlling the
    // activation/deactivation of endpaths.
    for(auto& p : end_paths_) {
      p.processOneOccurrence<T>(ep, es, streamID_, context);
    }
  }
}

#endif

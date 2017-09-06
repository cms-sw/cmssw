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
#include "FWCore/Concurrency/interface/FunctorTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <sstream>
#include <atomic>

namespace edm {

  class ActivityRegistry;
  class BranchIDListHelper;
  class EventSetup;
  class ExceptionCollector;
  class ExceptionToActionTable;
  class OutputModuleCommunicator;
  class ProcessContext;
  class UnscheduledCallProducer;
  class WorkerInPath;
  class ModuleRegistry;
  class TriggerResultInserter;
  class PathStatusInserter;
  class EndPathStatusInserter;
  class PreallocationConfiguration;
  class WaitingTaskHolder;

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
      ActivityRegistry* a_; // We do not use propagate_const because the registry itself is mutable.
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
    typedef std::shared_ptr<HLTGlobalStatus const> TrigResConstPtr;
    typedef std::shared_ptr<Worker> WorkerPtr;
    typedef std::vector<Worker*> AllWorkers;

    typedef std::vector<Worker*> Workers;

    typedef std::vector<WorkerInPath> PathWorkers;

    StreamSchedule(std::shared_ptr<TriggerResultInserter> inserter,
                   std::vector<edm::propagate_const<std::shared_ptr<PathStatusInserter>>>& pathStatusInserters,
                   std::vector<edm::propagate_const<std::shared_ptr<EndPathStatusInserter>>>& endPathStatusInserters,
                   std::shared_ptr<ModuleRegistry>,
                   ParameterSet& proc_pset,
                   service::TriggerNamesService const& tns,
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

    void processOneEventAsync(WaitingTaskHolder iTask,
                              EventPrincipal& ep,
                              EventSetup const& es,
                              std::vector<edm::propagate_const<std::shared_ptr<PathStatusInserter>>>& pathStatusInserters);

    template <typename T>
    void processOneStreamAsync(WaitingTaskHolder iTask,
                               typename T::MyPrincipal& principal,
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

    void moduleDescriptionsInPath(std::string const& iPathLabel,
                                  std::vector<ModuleDescription const*>& descriptions,
                                  unsigned int hint) const;

    void moduleDescriptionsInEndPath(std::string const& iEndPathLabel,
                                     std::vector<ModuleDescription const*>& descriptions,
                                     unsigned int hint) const;

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
    
    StreamContext const& context() const { return streamContext_;}
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
      edm::ActivityRegistry* reg_; // We do not use propagate_const because the registry itself is mutable.
      StreamContext const* context_;
    };

    /// returns the action table
    ExceptionToActionTable const& actionTable() const {
      return workerManager_.actionTable();
    }
    

    void resetAll();

    void finishedPaths(std::exception_ptr, WaitingTaskHolder,
                       EventPrincipal& ep, EventSetup const& es);
    std::exception_ptr finishProcessOneEvent(std::exception_ptr);
    
    void reportSkipped(EventPrincipal const& ep) const;

    void fillWorkers(ParameterSet& proc_pset,
                     ProductRegistry& preg,
                     PreallocationConfiguration const* prealloc,
                     std::shared_ptr<ProcessConfiguration const> processConfiguration,
                     std::string const& name, bool ignoreFilters, PathWorkers& out,
                     std::vector<std::string> const& endPathNames);
    void fillTrigPath(ParameterSet& proc_pset,
                      ProductRegistry& preg,
                      PreallocationConfiguration const* prealloc,
                      std::shared_ptr<ProcessConfiguration const> processConfiguration,
                      int bitpos, std::string const& name, TrigResPtr,
                      std::vector<std::string> const& endPathNames);
    void fillEndPath(ParameterSet& proc_pset,
                     ProductRegistry& preg,
                     PreallocationConfiguration const* prealloc,
                     std::shared_ptr<ProcessConfiguration const> processConfiguration,
                     int bitpos, std::string const& name,
                     std::vector<std::string> const& endPathNames);

    void addToAllWorkers(Worker* w);
    
    void resetEarlyDelete();
    void initializeEarlyDelete(ModuleRegistry & modReg,
                               edm::ParameterSet const& opts,
                               edm::ProductRegistry const& preg, 
                               bool allowEarlyDelete);

    TrigResConstPtr results() const {return get_underlying_safe(results_);}
    TrigResPtr& results() {return get_underlying_safe(results_);}

    void makePathStatusInserters(
      std::vector<edm::propagate_const<std::shared_ptr<PathStatusInserter>>>& pathStatusInserters,
      std::vector<edm::propagate_const<std::shared_ptr<EndPathStatusInserter>>>& endPathStatusInserters,
      ExceptionToActionTable const& actions);

    WorkerManager            workerManager_;
    std::shared_ptr<ActivityRegistry> actReg_; // We do not use propagate_const because the registry itself is mutable.

    edm::propagate_const<TrigResPtr> results_;

    edm::propagate_const<WorkerPtr> results_inserter_;
    std::vector<edm::propagate_const<WorkerPtr>> pathStatusInserterWorkers_;
    std::vector<edm::propagate_const<WorkerPtr>> endPathStatusInserterWorkers_;

    TrigPaths                trig_paths_;
    TrigPaths                end_paths_;
    std::vector<int>         empty_trig_paths_;
    std::vector<int>         empty_end_paths_;

    //For each branch that has been marked for early deletion
    // keep track of how many modules are left that read this data but have
    // not yet been run in this event
    std::vector<BranchToCount> earlyDeleteBranchToCount_;
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
    std::atomic<bool>       skippingEvent_;
  };

  void
  inline
  StreamSchedule::reportSkipped(EventPrincipal const& ep) const {
    Service<JobReport> reportSvc;
    reportSvc->reportSkippedEvent(ep.id().run(), ep.id().event());
  }

  template <typename T>
  void StreamSchedule::processOneStreamAsync(WaitingTaskHolder iHolder,
                                             typename T::MyPrincipal& ep,
                                             EventSetup const& es,
                                             bool cleaningUpAfterException) {
    ServiceToken token = ServiceRegistry::instance().presentToken();

    T::setStreamContext(streamContext_, ep);

    auto id = ep.id();
    auto doneTask = make_waiting_task(tbb::task::allocate_root(),
                                      [this,iHolder, id,cleaningUpAfterException,token](std::exception_ptr const* iPtr) mutable
    {
      ServiceRegistry::Operate op(token);
      std::exception_ptr excpt;
      if(iPtr) {
        excpt = *iPtr;
        //add context information to the exception and print message
        try {
          convertException::wrap([&]() {
            std::rethrow_exception(excpt);
          });
        } catch(cms::Exception& ex) {
          //TODO: should add the transition type info
          std::ostringstream ost;
          if(ex.context().empty()) {
            ost<<"Processing "<<T::transitionName()<<" "<<id;
          }
          addContextAndPrintException(ost.str().c_str(), ex, cleaningUpAfterException);
          excpt = std::current_exception();
        }
        
        actReg_->preStreamEarlyTerminationSignal_(streamContext_,TerminationOrigin::ExceptionFromThisContext);
      }
      
      try {
        T::postScheduleSignal(actReg_.get(), &streamContext_);
      } catch(...) {
        if(not excpt) {
          excpt = std::current_exception();
        }
      }
      iHolder.doneWaiting(excpt);
      
    });
    
    auto task = make_functor_task(tbb::task::allocate_root(), [this,doneTask,&ep,&es,cleaningUpAfterException,token] () mutable {
      ServiceRegistry::Operate op(token);
      T::preScheduleSignal(actReg_.get(), &streamContext_);
      WaitingTaskHolder h(doneTask);

      workerManager_.resetAll();
      for(auto& p : end_paths_) {
        p.runAllModulesAsync<T>(doneTask, ep, es, streamID_, &streamContext_);
      }

      for(auto& p : trig_paths_) {
        p.runAllModulesAsync<T>(doneTask, ep, es, streamID_, &streamContext_);
      }
      
      workerManager_.processOneOccurrenceAsync<T>(doneTask,
                                                  ep, es, streamID_, &streamContext_, &streamContext_);
    });
    
    if(streamID_.value() == 0) {
      //Enqueueing will start another thread if there is only
      // one thread in the job. Having stream == 0 use spawn
      // avoids starting up another thread when there is only one stream.
      tbb::task::spawn( *task);
    } else {
      tbb::task::enqueue( *task);
    }
  }
}

#endif

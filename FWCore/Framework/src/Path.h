#ifndef FWCore_Framework_Path_h
#define FWCore_Framework_Path_h

/*
  Author: Jim Kowalkowski 28-01-06

  An object of this type represents one path in a job configuration.
  It holds the assigned bit position and the list of workers that are
  an event must pass through when this parh is processed.  The workers
  are held in WorkerInPath wrappers so that per path execution statistics
  can be kept for each worker.
*/

#include "FWCore/Framework/src/WorkerInPath.h"
#include "FWCore/Framework/src/Worker.h"
#include "DataFormats/Common/interface/HLTenums.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/make_sentry.h"

#include <memory>

#include <string>
#include <vector>
#include <map>
#include <exception>
#include <sstream>

namespace edm {
  class EventPrincipal;
  class EventSetup;
  class ModuleDescription;
  class PathStatusInserter;
  class RunPrincipal;
  class LuminosityBlockPrincipal;
  class EarlyDeleteHelper;
  class StreamContext;
  class StreamID;
  class WaitingTask;

  class Path {
  public:
    typedef hlt::HLTState State;

    typedef std::vector<WorkerInPath> WorkersInPath;
    typedef WorkersInPath::size_type        size_type;
    typedef std::shared_ptr<HLTGlobalStatus> TrigResPtr;

    Path(int bitpos, std::string const& path_name,
         WorkersInPath const& workers,
         TrigResPtr trptr,
         ExceptionToActionTable const& actions,
         std::shared_ptr<ActivityRegistry> reg,
         StreamContext const* streamContext,
         std::atomic<bool>* stopProcessEvent,
         PathContext::PathType pathType);

    Path(Path const&);

    template <typename T>
    void runAllModulesAsync(WaitingTask*,
                            typename T::MyPrincipal const&,
                            EventSetup  const&,
                            StreamID const&,
                            typename T::Context const*);

    void processOneOccurrenceAsync(WaitingTask*, EventPrincipal const&, EventSetup const&, StreamID const&, StreamContext const*);
    
    int bitPosition() const { return bitpos_; }
    std::string const& name() const { return pathContext_.pathName(); }

    void clearCounters();

    int timesRun() const { return timesRun_; }
    int timesPassed() const { return timesPassed_; }
    int timesFailed() const { return timesFailed_; }
    int timesExcept() const { return timesExcept_; }
    //int abortWorker() const { return abortWorker_; }
    State state() const { return state_; }

    size_type size() const { return workers_.size(); }
    int timesVisited(size_type i) const { return workers_.at(i).timesVisited(); }
    int timesPassed (size_type i) const { return workers_.at(i).timesPassed() ; }
    int timesFailed (size_type i) const { return workers_.at(i).timesFailed() ; }
    int timesExcept (size_type i) const { return workers_.at(i).timesExcept() ; }
    Worker const* getWorker(size_type i) const { return workers_.at(i).getWorker(); }
    
    void setEarlyDeleteHelpers(std::map<const Worker*,EarlyDeleteHelper*> const&);

    void setPathStatusInserter(PathStatusInserter* pathStatusInserter,
                               Worker* pathStatusInserterWorker);

  private:

    // If you define this be careful about the pointer in the
    // PlaceInPathContext object in the contained WorkerInPath objects.
    Path const& operator=(Path const&) = delete; // stop default

    int timesRun_;
    int timesPassed_;
    int timesFailed_;
    int timesExcept_;
    //int abortWorker_;
    State state_;

    int bitpos_;
    TrigResPtr trptr_;
    std::shared_ptr<ActivityRegistry> actReg_; // We do not use propagate_const because the registry itself is mutable.
    ExceptionToActionTable const* act_table_;

    WorkersInPath workers_;
    std::vector<EarlyDeleteHelper*> earlyDeleteHelpers_;

    PathContext pathContext_;
    WaitingTaskList waitingTasks_;
    std::atomic<bool>* stopProcessingEvent_;

    PathStatusInserter* pathStatusInserter_;
    Worker* pathStatusInserterWorker_;

    // Helper functions
    // nwrwue = numWorkersRunWithoutUnhandledException (really!)
    bool handleWorkerFailure(cms::Exception & e,
                             int nwrwue,
                             bool isEvent,
                             bool begin,
                             BranchType branchType,
                             ModuleDescription const&,
                             std::string const& id);
    static void exceptionContext(cms::Exception & ex,
                                 bool isEvent,
                                 bool begin,
                                 BranchType branchType,
                                 ModuleDescription const&,
                                 std::string const& id,
                                 PathContext const&);
    void recordStatus(int nwrwue, bool isEvent);
    void updateCounters(bool succeed, bool isEvent);
    
    void finished(int iModuleIndex, bool iSucceeded, std::exception_ptr,
                  StreamContext const*,
                  EventPrincipal const& iEP,
                  EventSetup const& iES,
                  StreamID const& streamID);

    void handleEarlyFinish(EventPrincipal const&);
    void handleEarlyFinish(RunPrincipal const&) {}
    void handleEarlyFinish(LuminosityBlockPrincipal const&) {}
    
    //Handle asynchronous processing
    void workerFinished(std::exception_ptr const* iException,
                        unsigned int iModuleIndex,
                        EventPrincipal const& iEP, EventSetup const& iES,
                        StreamID const& iID, StreamContext const* iContext);
    void runNextWorkerAsync(unsigned int iNextModuleIndex,
                            EventPrincipal const&, EventSetup const&,
                            StreamID const&, StreamContext const*);

  };

  namespace {
    template <typename T>
    class PathSignalSentry {
    public:
      PathSignalSentry(ActivityRegistry *a,
                       int const& nwrwue,
                       hlt::HLTState const& state,
                       PathContext const* pathContext) :
        a_(a), nwrwue_(nwrwue), state_(state), pathContext_(pathContext) {
        if (a_) T::prePathSignal(a_, pathContext_);
      }
      ~PathSignalSentry() {
        HLTPathStatus status(state_, nwrwue_);
        if(a_) T::postPathSignal(a_, status, pathContext_);
      }
    private:
      ActivityRegistry* a_; // We do not use propagate_const because the registry itself is mutable.
      int const& nwrwue_;
      hlt::HLTState const& state_;
      PathContext const* pathContext_;
    };
  }

  template <typename T>
  void Path::runAllModulesAsync(WaitingTask* task,
                          typename T::MyPrincipal const& p,
                          EventSetup  const& es,
                          StreamID const& streamID,
                                typename T::Context const* context) {
    for(auto& worker: workers_) {
      worker.runWorkerAsync<T>(task,p,es,streamID,context);
    }
  }

}

#endif

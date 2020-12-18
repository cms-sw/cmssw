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
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
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
  class EventTransitionInfo;
  class ModuleDescription;
  class PathStatusInserter;
  class EarlyDeleteHelper;
  class StreamContext;
  class StreamID;

  class Path {
  public:
    typedef hlt::HLTState State;

    typedef std::vector<WorkerInPath> WorkersInPath;
    typedef WorkersInPath::size_type size_type;
    typedef std::shared_ptr<HLTGlobalStatus> TrigResPtr;

    Path(int bitpos,
         std::string const& path_name,
         WorkersInPath const& workers,
         TrigResPtr trptr,
         ExceptionToActionTable const& actions,
         std::shared_ptr<ActivityRegistry> reg,
         StreamContext const* streamContext,
         std::atomic<bool>* stopProcessEvent,
         PathContext::PathType pathType);

    Path(Path const&);

    Path& operator=(Path const&) = delete;

    template <typename T>
    void runAllModulesAsync(WaitingTaskHolder,
                            typename T::TransitionInfoType const&,
                            ServiceToken const&,
                            StreamID const&,
                            typename T::Context const*);

    void processOneOccurrenceAsync(
        WaitingTaskHolder, EventTransitionInfo const&, ServiceToken const&, StreamID const&, StreamContext const*);

    int bitPosition() const { return bitpos_; }
    std::string const& name() const { return pathContext_.pathName(); }

    void clearCounters();

    int timesRun() const { return timesRun_; }
    int timesPassed() const { return timesPassed_; }
    int timesFailed() const { return timesFailed_; }
    int timesExcept() const { return timesExcept_; }
    //int abortWorker() const { return abortWorker_; }

    size_type size() const { return workers_.size(); }
    int timesVisited(size_type i) const { return workers_.at(i).timesVisited(); }
    int timesPassed(size_type i) const { return workers_.at(i).timesPassed(); }
    int timesFailed(size_type i) const { return workers_.at(i).timesFailed(); }
    int timesExcept(size_type i) const { return workers_.at(i).timesExcept(); }
    Worker const* getWorker(size_type i) const { return workers_.at(i).getWorker(); }

    void setEarlyDeleteHelpers(std::map<const Worker*, EarlyDeleteHelper*> const&);

    void setPathStatusInserter(PathStatusInserter* pathStatusInserter, Worker* pathStatusInserterWorker);

  private:
    int timesRun_;
    int timesPassed_;
    int timesFailed_;
    int timesExcept_;
    //int abortWorker_;
    //When an exception happens, it is possible for multiple modules in a path to fail
    // and then try to change the state concurrently.
    std::atomic<bool> stateLock_ = false;
    CMS_THREAD_GUARD(stateLock_) int failedModuleIndex_;
    CMS_THREAD_GUARD(stateLock_) State state_;

    int const bitpos_;
    TrigResPtr const trptr_;
    // We do not use propagate_const because the registry itself is mutable.
    std::shared_ptr<ActivityRegistry> const actReg_;
    ExceptionToActionTable const* const act_table_;

    WorkersInPath workers_;

    PathContext pathContext_;
    WaitingTaskList waitingTasks_;
    std::atomic<bool>* const stopProcessingEvent_;
    std::atomic<unsigned int> modulesToRun_;

    PathStatusInserter* pathStatusInserter_;
    Worker* pathStatusInserterWorker_;

    // Helper functions
    // nwrwue = numWorkersRunWithoutUnhandledException (really!)
    bool handleWorkerFailure(cms::Exception& e,
                             int nwrwue,
                             bool isEvent,
                             bool begin,
                             BranchType branchType,
                             ModuleDescription const&,
                             std::string const& id) const;
    static void exceptionContext(cms::Exception& ex,
                                 bool isEvent,
                                 bool begin,
                                 BranchType branchType,
                                 ModuleDescription const&,
                                 std::string const& id,
                                 PathContext const&);
    void threadsafe_setFailedModuleInfo(int nwrwue, std::exception_ptr);
    void recordStatus(int nwrwue, hlt::HLTState state);
    void updateCounters(hlt::HLTState state);

    void finished(std::exception_ptr, StreamContext const*, EventTransitionInfo const&, StreamID const&);

    //Handle asynchronous processing
    void workerFinished(std::exception_ptr const*,
                        unsigned int iModuleIndex,
                        EventTransitionInfo const&,
                        ServiceToken const&,
                        StreamID const&,
                        StreamContext const*);
    void runNextWorkerAsync(unsigned int iNextModuleIndex,
                            EventTransitionInfo const&,
                            ServiceToken const&,
                            StreamID const&,
                            StreamContext const*);
  };

  namespace {
    template <typename T>
    class PathSignalSentry {
    public:
      PathSignalSentry(ActivityRegistry* a,
                       int const& nwrwue,
                       hlt::HLTState const& state,
                       PathContext const* pathContext)
          : a_(a), nwrwue_(nwrwue), state_(state), pathContext_(pathContext) {
        if (a_)
          T::prePathSignal(a_, pathContext_);
      }
      ~PathSignalSentry() {
        HLTPathStatus status(state_, nwrwue_);
        if (a_)
          T::postPathSignal(a_, status, pathContext_);
      }

    private:
      ActivityRegistry* a_;  // We do not use propagate_const because the registry itself is mutable.
      int const& nwrwue_;
      hlt::HLTState const& state_;
      PathContext const* pathContext_;
    };
  }  // namespace

  template <typename T>
  void Path::runAllModulesAsync(WaitingTaskHolder task,
                                typename T::TransitionInfoType const& info,
                                ServiceToken const& token,
                                StreamID const& streamID,
                                typename T::Context const* context) {
    for (auto& worker : workers_) {
      worker.runWorkerAsync<T>(task, info, token, streamID, context);
    }
  }

}  // namespace edm

#endif

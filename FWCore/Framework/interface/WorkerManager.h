#ifndef FWCore_Framework_WorkerManager_h
#define FWCore_Framework_WorkerManager_h

/*

*/

#include "FWCore/Framework/interface/ExceptionHelpers.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/UnscheduledCallProducer.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/WorkerRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

#include <memory>

#include <set>
#include <string>
#include <vector>

namespace edm {
  class EventTransitionInfo;
  class ExceptionCollector;
  class StreamID;
  class StreamContext;
  class ModuleRegistry;
  class PreallocationConfiguration;
  namespace eventsetup {
    class ESRecordsToProxyIndices;
  }
  class WorkerManager {
  public:
    typedef std::vector<Worker*> AllWorkers;

    WorkerManager(std::shared_ptr<ActivityRegistry> actReg, ExceptionToActionTable const& actions);
    WorkerManager(WorkerManager&&) = default;

    WorkerManager(std::shared_ptr<ModuleRegistry> modReg,
                  std::shared_ptr<ActivityRegistry> actReg,
                  ExceptionToActionTable const& actions);

    void deleteModuleIfExists(std::string const& moduleLabel);

    void addToUnscheduledWorkers(ParameterSet& pset,
                                 ProductRegistry& preg,
                                 PreallocationConfiguration const* prealloc,
                                 std::shared_ptr<ProcessConfiguration> processConfiguration,
                                 std::string label,
                                 std::set<std::string>& unscheduledLabels,
                                 std::vector<std::string>& shouldBeUsedLabels);

    template <typename T, typename U>
    void processOneOccurrence(typename T::TransitionInfoType&,
                              StreamID,
                              typename T::Context const* topContext,
                              U const* context,
                              bool cleaningUpAfterException = false);
    template <typename T, typename U>
    void processOneOccurrenceAsync(WaitingTask*,
                                   typename T::TransitionInfoType&,
                                   ServiceToken const&,
                                   StreamID,
                                   typename T::Context const* topContext,
                                   U const* context);

    template <typename T>
    void processAccumulatorsAsync(WaitingTask*,
                                  typename T::TransitionInfoType const&,
                                  ServiceToken const&,
                                  StreamID,
                                  ParentContext const&,
                                  typename T::Context const*);

    void setupResolvers(Principal& principal);
    void setupOnDemandSystem(EventTransitionInfo const&);

    void beginJob(ProductRegistry const& iRegistry, eventsetup::ESRecordsToProxyIndices const&);
    void endJob();
    void endJob(ExceptionCollector& collector);

    void beginStream(StreamID iID, StreamContext& streamContext);
    void endStream(StreamID iID, StreamContext& streamContext);

    AllWorkers const& allWorkers() const { return allWorkers_; }

    void addToAllWorkers(Worker* w);

    ExceptionToActionTable const& actionTable() const { return *actionTable_; }

    Worker* getWorker(ParameterSet& pset,
                      ProductRegistry& preg,
                      PreallocationConfiguration const* prealloc,
                      std::shared_ptr<ProcessConfiguration const> processConfiguration,
                      std::string const& label);

    void resetAll();

  private:
    WorkerRegistry workerReg_;
    ExceptionToActionTable const* actionTable_;
    AllWorkers allWorkers_;
    UnscheduledCallProducer unscheduled_;
    void const* lastSetupEventPrincipal_;
  };

  template <typename T, typename U>
  void WorkerManager::processOneOccurrence(typename T::TransitionInfoType& info,
                                           StreamID streamID,
                                           typename T::Context const* topContext,
                                           U const* context,
                                           bool cleaningUpAfterException) {
    this->resetAll();

    auto waitTask = make_empty_waiting_task();
    waitTask->increment_ref_count();
    processOneOccurrenceAsync<T, U>(
        waitTask.get(), info, ServiceRegistry::instance().presentToken(), streamID, topContext, context);
    waitTask->wait_for_all();
    if (waitTask->exceptionPtr() != nullptr) {
      try {
        convertException::wrap([&]() { std::rethrow_exception(*(waitTask->exceptionPtr())); });
      } catch (cms::Exception& ex) {
        if (ex.context().empty()) {
          addContextAndPrintException(
              "Calling function WorkerManager::processOneOccurrence", ex, cleaningUpAfterException);
        } else {
          addContextAndPrintException("", ex, cleaningUpAfterException);
        }
        throw;
      }
    }
  }

  template <typename T, typename U>
  void WorkerManager::processOneOccurrenceAsync(WaitingTask* task,
                                                typename T::TransitionInfoType& info,
                                                ServiceToken const& token,
                                                StreamID streamID,
                                                typename T::Context const* topContext,
                                                U const* context) {
    //make sure the unscheduled items see this run or lumi transition
    unscheduled_.runNowAsync<T, U>(task, info, token, streamID, topContext, context);
  }

  template <typename T>
  void WorkerManager::processAccumulatorsAsync(WaitingTask* task,
                                               typename T::TransitionInfoType const& info,
                                               ServiceToken const& token,
                                               StreamID streamID,
                                               ParentContext const& parentContext,
                                               typename T::Context const* context) {
    unscheduled_.runAccumulatorsAsync<T>(task, info, token, streamID, parentContext, context);
  }
}  // namespace edm

#endif

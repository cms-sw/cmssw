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
    void addToUnscheduledWorkers(ParameterSet& pset,
                                 ProductRegistry& preg,
                                 PreallocationConfiguration const* prealloc,
                                 std::shared_ptr<ProcessConfiguration> processConfiguration,
                                 std::string label,
                                 std::set<std::string>& unscheduledLabels,
                                 std::vector<std::string>& shouldBeUsedLabels);

    template <typename T, typename U>
    void processOneOccurrence(typename T::MyPrincipal& principal,
                              EventSetupImpl const& eventSetup,
                              StreamID streamID,
                              typename T::Context const* topContext,
                              U const* context,
                              bool cleaningUpAfterException = false);
    template <typename T, typename U>
    void processOneOccurrenceAsync(WaitingTask* task,
                                   typename T::MyPrincipal& principal,
                                   EventSetupImpl const& eventSetup,
                                   ServiceToken const& token,
                                   StreamID streamID,
                                   typename T::Context const* topContext,
                                   U const* context);

    template <typename T>
    void processAccumulatorsAsync(WaitingTask* task,
                                  typename T::MyPrincipal const& ep,
                                  EventSetupImpl const& es,
                                  ServiceToken const& token,
                                  StreamID streamID,
                                  ParentContext const& parentContext,
                                  typename T::Context const* context);

    void setupOnDemandSystem(Principal& principal, EventSetupImpl const& es);

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
  void WorkerManager::processOneOccurrence(typename T::MyPrincipal& ep,
                                           EventSetupImpl const& es,
                                           StreamID streamID,
                                           typename T::Context const* topContext,
                                           U const* context,
                                           bool cleaningUpAfterException) {
    this->resetAll();

    auto waitTask = make_empty_waiting_task();
    waitTask->increment_ref_count();
    processOneOccurrenceAsync<T, U>(
        waitTask.get(), ep, es, ServiceRegistry::instance().presentToken(), streamID, topContext, context);
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
                                                typename T::MyPrincipal& ep,
                                                EventSetupImpl const& es,
                                                ServiceToken const& token,
                                                StreamID streamID,
                                                typename T::Context const* topContext,
                                                U const* context) {
    //make sure the unscheduled items see this run or lumi transition
    unscheduled_.runNowAsync<T, U>(task, ep, es, token, streamID, topContext, context);
  }

  template <typename T>
  void WorkerManager::processAccumulatorsAsync(WaitingTask* task,
                                               typename T::MyPrincipal const& ep,
                                               EventSetupImpl const& es,
                                               ServiceToken const& token,
                                               StreamID streamID,
                                               ParentContext const& parentContext,
                                               typename T::Context const* context) {
    unscheduled_.runAccumulatorsAsync<T>(task, ep, es, token, streamID, parentContext, context);
  }
}  // namespace edm

#endif

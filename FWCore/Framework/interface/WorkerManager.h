#ifndef FWCore_Framework_WorkerManager_h
#define FWCore_Framework_WorkerManager_h

#include "FWCore/Common/interface/FWCoreCommonFwd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/UnscheduledCallProducer.h"
#include "FWCore/Framework/interface/WorkerRegistry.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistryfwd.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace edm {
  class ExceptionCollector;
  class ExceptionToActionTable;
  class ModuleRegistry;
  class ModuleTypeResolverMaker;
  class PreallocationConfiguration;
  class Worker;
  namespace eventsetup {
    class ESRecordsToProductResolverIndices;
  }

  class WorkerManager {
  public:
    typedef std::vector<Worker*> AllWorkers;

    WorkerManager(std::shared_ptr<ActivityRegistry> actReg,
                  ExceptionToActionTable const& actions,
                  ModuleTypeResolverMaker const* typeResolverMaker);
    WorkerManager(WorkerManager&&) = default;

    WorkerManager(std::shared_ptr<ModuleRegistry> modReg,
                  std::shared_ptr<ActivityRegistry> actReg,
                  ExceptionToActionTable const& actions);

    void deleteModuleIfExists(std::string const& moduleLabel);

    void addToUnscheduledWorkers(ParameterSet& pset,
                                 SignallingProductRegistryFiller& preg,
                                 PreallocationConfiguration const* prealloc,
                                 std::shared_ptr<ProcessConfiguration const> processConfiguration,
                                 std::string label,
                                 std::set<std::string>& unscheduledLabels,
                                 std::vector<std::string>& shouldBeUsedLabels);

    template <typename T, typename U>
    void processOneOccurrenceAsync(WaitingTaskHolder,
                                   typename T::TransitionInfoType&,
                                   ServiceToken const&,
                                   StreamID,
                                   typename T::Context const* topContext,
                                   U const* context) noexcept;

    template <typename T>
    void processAccumulatorsAsync(WaitingTaskHolder,
                                  typename T::TransitionInfoType const&,
                                  ServiceToken const&,
                                  StreamID,
                                  ParentContext const&,
                                  typename T::Context const*);

    void setupResolvers(Principal& principal);
    void setupOnDemandSystem(EventTransitionInfo const&);

    void beginJob(ProductRegistry const& iRegistry,
                  eventsetup::ESRecordsToProductResolverIndices const&,
                  ProcessBlockHelperBase const&,
                  GlobalContext const&);
    void endJob(ExceptionCollector&, GlobalContext const&);

    void beginStream(StreamID, StreamContext const&);
    void endStream(StreamID, StreamContext const&, ExceptionCollector&, std::mutex& collectorMutex) noexcept;

    AllWorkers const& allWorkers() const { return allWorkers_; }
    AllWorkers const& unscheduledWorkers() const { return unscheduled_.workers(); }

    void addToAllWorkers(Worker* w);

    ExceptionToActionTable const& actionTable() const { return *actionTable_; }

    Worker* getWorker(ParameterSet& pset,
                      SignallingProductRegistryFiller& preg,
                      PreallocationConfiguration const* prealloc,
                      std::shared_ptr<ProcessConfiguration const> processConfiguration,
                      std::string const& label);

    void resetAll();

    void releaseMemoryPostLookupSignal();

  private:
    WorkerRegistry workerReg_;
    ExceptionToActionTable const* actionTable_;
    AllWorkers allWorkers_;
    UnscheduledCallProducer unscheduled_;
    void const* lastSetupEventPrincipal_;
  };

  template <typename T, typename U>
  void WorkerManager::processOneOccurrenceAsync(WaitingTaskHolder task,
                                                typename T::TransitionInfoType& info,
                                                ServiceToken const& token,
                                                StreamID streamID,
                                                typename T::Context const* topContext,
                                                U const* context) noexcept {
    static_assert(!T::isEvent_);

    // Spawn them in reverse order. At least in the single threaded case that makes
    // them run in forward order (and more likely to with multiple threads).
    for (auto it = allWorkers_.rbegin(), itEnd = allWorkers_.rend(); it != itEnd; ++it) {
      Worker* worker = *it;

      ParentContext parentContext(context);

      // We do not need to run prefetching here because this only handles
      // stream begin/end transitions for runs and lumis. There are no products
      // put into the runs or lumis in stream transitions, so there can be
      // no data dependencies which require prefetching. Prefetching is
      // needed for global transitions, but they are run elsewhere.
      // (One exception, the SecondaryEventProvider (used for mixing) sends
      // global begin/end run/lumi transitions through here. They shouldn't
      // need prefetching either and for some years nothing has been using
      // that part of the code anyway...)
      worker->doWorkNoPrefetchingAsync<T>(task, info, token, streamID, parentContext, topContext);
    }
  }

  template <typename T>
  void WorkerManager::processAccumulatorsAsync(WaitingTaskHolder task,
                                               typename T::TransitionInfoType const& info,
                                               ServiceToken const& token,
                                               StreamID streamID,
                                               ParentContext const& parentContext,
                                               typename T::Context const* context) {
    unscheduled_.runAccumulatorsAsync<T>(std::move(task), info, token, streamID, parentContext, context);
  }
}  // namespace edm

#endif

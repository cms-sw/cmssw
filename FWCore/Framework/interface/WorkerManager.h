#ifndef FWCore_Framework_WorkerManager_h
#define FWCore_Framework_WorkerManager_h

/*

*/

#include "FWCore/Common/interface/FWCoreCommonFwd.h"
#include "FWCore/Framework/interface/ExceptionHelpers.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/UnscheduledCallProducer.h"
#include "FWCore/Framework/interface/maker/Worker.h"
#include "FWCore/Framework/interface/WorkerRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
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
  class ModuleTypeResolverMaker;
  class PreallocationConfiguration;
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
                                 ProductRegistry& preg,
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
                                   U const* context);

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
                  ProcessBlockHelperBase const&);
    void endJob();
    void endJob(ExceptionCollector& collector);

    void beginStream(StreamID iID, StreamContext& streamContext);
    void endStream(StreamID iID, StreamContext& streamContext);

    AllWorkers const& allWorkers() const { return allWorkers_; }
    AllWorkers const& unscheduledWorkers() const { return unscheduled_.workers(); }

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
  void WorkerManager::processOneOccurrenceAsync(WaitingTaskHolder task,
                                                typename T::TransitionInfoType& info,
                                                ServiceToken const& token,
                                                StreamID streamID,
                                                typename T::Context const* topContext,
                                                U const* context) {
    //make sure the unscheduled items see this run or lumi transition
    unscheduled_.runNowAsync<T, U>(std::move(task), info, token, streamID, topContext, context);
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

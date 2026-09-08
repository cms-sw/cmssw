#ifndef FWCore_Framework_WorkerManager_h
#define FWCore_Framework_WorkerManager_h

#include "FWCore/Common/interface/FWCoreCommonFwd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/TransitionPhaseTypes.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
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
  class ExceptionToActionTable;
  class ModuleRegistry;
  class Worker;
  namespace eventsetup {
    class ESRecordsToProductResolverIndices;
  }

  template <typename TI, typename TP>
  class WorkerManagerCore {
  public:
    typedef std::vector<Worker*> AllWorkers;
    WorkerManagerCore(WorkerManagerCore&&) = default;

    WorkerManagerCore(std::shared_ptr<ModuleRegistry> modReg,
                      std::shared_ptr<ActivityRegistry> actReg,
                      ExceptionToActionTable const& actions);

    AllWorkers const& allWorkers() const { return allWorkers_; }

    void addToAllWorkers(Worker* w);

    ExceptionToActionTable const& actionTable() const { return *actionTable_; }

    template <typename T>
      requires requires(T const& x) { x.moduleDescription(); }
    Worker* getWorkerForModule(T const& module) {
      auto* worker = getWorkerForExistingModule(module.moduleDescription().moduleLabel());
      assert(worker != nullptr);
      assert(worker->matchesBaseClassPointer(static_cast<typename T::ModuleType const*>(&module)));
      return worker;
    }

    Worker* getWorkerForModule(edm::ModuleDescription const& iDescription) {
      auto* worker = getWorkerForExistingModule(iDescription.moduleLabel());
      assert(worker != nullptr);
      assert(worker->description() == &iDescription);
      return worker;
    }

    void resetAll();

  protected:
    Worker const* deleteModuleIfExists(std::string const& moduleLabel);
    AllWorkers& allWorkers() { return allWorkers_; }

    Worker* getWorkerForExistingModuleUnattached(std::string const& label) {
      return workerReg_.getWorkerFromExistingModule(label, actionTable_);
    }

    void setupResolvers(Principal& principal, UnscheduledAuxiliary const* aux);

  private:
    Worker* getWorkerForExistingModule(std::string const& label);

    WorkerRegistry<TI, TP> workerReg_;
    ExceptionToActionTable const* actionTable_;
    AllWorkers allWorkers_;
    void const* lastSetupPrincipal_;
  };

  template <typename TI, typename TP>
  class WorkerManager;

  template <typename TI>
  class WorkerManager<TI, TransitionPhaseGlobal> : private WorkerManagerCore<TI, TransitionPhaseGlobal> {
  public:
    WorkerManager(WorkerManager&&) = default;

    WorkerManager(std::shared_ptr<ModuleRegistry> modReg,
                  std::shared_ptr<ActivityRegistry> actReg,
                  ExceptionToActionTable const& actions);

    using WorkerManagerCore<TI, TransitionPhaseGlobal>::allWorkers;
    using WorkerManagerCore<TI, TransitionPhaseGlobal>::addToAllWorkers;
    using WorkerManagerCore<TI, TransitionPhaseGlobal>::getWorkerForModule;
    using WorkerManagerCore<TI, TransitionPhaseGlobal>::actionTable;
    using WorkerManagerCore<TI, TransitionPhaseGlobal>::resetAll;
    using WorkerManagerCore<TI, TransitionPhaseGlobal>::setupResolvers;
    using AllWorkers = typename WorkerManagerCore<TI, TransitionPhaseGlobal>::AllWorkers;

    //used by all but specialized for events
    void setupResolvers(Principal& principal);

    void deleteModuleIfExists(std::string const& moduleLabel);
  };

  template <typename TI>
  class WorkerManager<TI, TransitionPhaseStream> : private WorkerManagerCore<TI, TransitionPhaseStream> {
  public:
    WorkerManager(WorkerManager&&) = default;

    WorkerManager(std::shared_ptr<ModuleRegistry> modReg,
                  std::shared_ptr<ActivityRegistry> actReg,
                  ExceptionToActionTable const& actions);

    using WorkerManagerCore<TI, TransitionPhaseStream>::allWorkers;
    using WorkerManagerCore<TI, TransitionPhaseStream>::addToAllWorkers;
    using WorkerManagerCore<TI, TransitionPhaseStream>::getWorkerForModule;
    using WorkerManagerCore<TI, TransitionPhaseStream>::actionTable;
    using WorkerManagerCore<TI, TransitionPhaseStream>::setupResolvers;
    using WorkerManagerCore<TI, TransitionPhaseStream>::resetAll;
    using AllWorkers = typename WorkerManagerCore<TI, TransitionPhaseStream>::AllWorkers;

    //ONLY CALLED BY RUN/LUMI STREAM TRANSITIONS
    template <typename T, typename U>
      requires std::is_same_v<TI, typename T::TransitionInfoType> &&
               std::is_same_v<typename TransitionPhaseStream::ContextType, typename T::Context>
    void processOneOccurrenceAsync(WaitingTaskHolder task,
                                   TI& info,
                                   ServiceToken const& token,
                                   StreamID streamID,
                                   typename TransitionPhaseStream::ContextType const* topContext,
                                   U const* context) noexcept {
      {
        static_assert(!T::isEvent_);

        // Spawn them in reverse order. At least in the single threaded case that makes
        // them run in forward order (and more likely to with multiple threads).
        for (auto it = allWorkers().rbegin(), itEnd = allWorkers().rend(); it != itEnd; ++it) {
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
    }
    //used by all but specialized for events
    void setupResolvers(Principal& principal);

    void deleteModuleIfExists(std::string const& moduleLabel);
  };

  template <>
  class WorkerManager<EventTransitionInfo, TransitionPhaseStream>
      : private WorkerManagerCore<EventTransitionInfo, TransitionPhaseStream> {
  public:
    WorkerManager(WorkerManager&&) = default;

    WorkerManager(std::shared_ptr<ModuleRegistry> modReg,
                  std::shared_ptr<ActivityRegistry> actReg,
                  ExceptionToActionTable const& actions);

    using WorkerManagerCore<EventTransitionInfo, TransitionPhaseStream>::allWorkers;
    using WorkerManagerCore<EventTransitionInfo, TransitionPhaseStream>::addToAllWorkers;
    using WorkerManagerCore<EventTransitionInfo, TransitionPhaseStream>::getWorkerForModule;
    using WorkerManagerCore<EventTransitionInfo, TransitionPhaseStream>::actionTable;
    using WorkerManagerCore<EventTransitionInfo, TransitionPhaseStream>::setupResolvers;
    using WorkerManagerCore<EventTransitionInfo, TransitionPhaseStream>::resetAll;
    using AllWorkers = typename WorkerManagerCore<EventTransitionInfo, TransitionPhaseStream>::AllWorkers;
    //ONLY USED BY EVENTS
    void addToUnscheduledWorkers(ModuleDescription const& iDescription);

    //ONLY CALLED FOR EVENT TRANSITION
    template <typename T>
      requires std::is_same_v<EventTransitionInfo, typename T::TransitionInfoType> &&
               std::is_same_v<typename TransitionPhaseStream::ContextType, typename T::Context>
    void processAccumulatorsAsync(WaitingTaskHolder task,
                                  EventTransitionInfo const& info,
                                  ServiceToken const& token,
                                  StreamID streamID,
                                  ParentContext const& parentContext,
                                  StreamContext const* context) {
      {
        unscheduled_.runAccumulatorsAsync<T>(std::move(task), info, token, streamID, parentContext, context);
      }
    }
    //ONLY CALLED BY EVENTS
    void setupOnDemandSystem(EventTransitionInfo const&);

    //used by all but specialized for events
    void setupResolvers(Principal& principal);

    void deleteModuleIfExists(std::string const& moduleLabel);

    //ONLY CALLED BY EVENTS
    AllWorkers const& unscheduledWorkers() const { return unscheduled_.workers(); }

  private:
    UnscheduledCallProducer unscheduled_;
  };
}  // namespace edm

#endif

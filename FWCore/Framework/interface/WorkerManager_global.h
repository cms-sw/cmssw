#ifndef FWCore_Framework_WorkerManager_global_h
#define FWCore_Framework_WorkerManager_global_h

#include "WorkerManager.h"

namespace edm {
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

    //Called by SecondaryEventProvider
    template <typename T, typename U>
      requires std::is_same_v<TI, typename T::TransitionInfoType> &&
               std::is_same_v<typename TransitionPhaseGlobal::ContextType, typename T::Context>
    void processOneOccurrenceAsync(WaitingTaskHolder task,
                                   TI& info,
                                   ServiceToken const& token,
                                   StreamID streamID,
                                   typename TransitionPhaseGlobal::ContextType const* topContext,
                                   U const* context) noexcept {
      {
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
  class WorkerManager<EventTransitionInfo, TransitionPhaseGlobal>
      : private WorkerManagerCore<EventTransitionInfo, TransitionPhaseGlobal> {
  public:
    WorkerManager(WorkerManager&&) = default;

    WorkerManager(std::shared_ptr<ModuleRegistry> modReg,
                  std::shared_ptr<ActivityRegistry> actReg,
                  ExceptionToActionTable const& actions);

    using WorkerManagerCore<EventTransitionInfo, TransitionPhaseGlobal>::allWorkers;
    using WorkerManagerCore<EventTransitionInfo, TransitionPhaseGlobal>::addToAllWorkers;
    using WorkerManagerCore<EventTransitionInfo, TransitionPhaseGlobal>::getWorkerForModule;
    using WorkerManagerCore<EventTransitionInfo, TransitionPhaseGlobal>::actionTable;
    using WorkerManagerCore<EventTransitionInfo, TransitionPhaseGlobal>::setupResolvers;
    using WorkerManagerCore<EventTransitionInfo, TransitionPhaseGlobal>::resetAll;
    using AllWorkers = typename WorkerManagerCore<EventTransitionInfo, TransitionPhaseGlobal>::AllWorkers;
    //ONLY USED BY EVENTS
    void addToUnscheduledWorkers(ModuleDescription const& iDescription);

    //ONLY CALLED FOR EVENT TRANSITION
    void processAccumulatorsAsync(WaitingTaskHolder task,
                                  EventTransitionInfo const& info,
                                  ServiceToken const& token,
                                  StreamID streamID,
                                  ParentContext const& parentContext,
                                  StreamContext const* context) {
      {
        unscheduled_.runAccumulatorsAsync(std::move(task), info, token, streamID, parentContext, context);
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

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

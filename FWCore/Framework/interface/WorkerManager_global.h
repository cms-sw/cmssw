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
}  // namespace edm

#endif

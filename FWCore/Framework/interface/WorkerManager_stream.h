#ifndef FWCore_Framework_WorkerManager_stream_h
#define FWCore_Framework_WorkerManager_stream_h

#include "WorkerManager.h"

namespace edm {
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
}  // namespace edm

#endif

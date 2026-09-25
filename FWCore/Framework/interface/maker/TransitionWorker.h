#ifndef FWCore_Framework_maker_TransitionWorker_h
#define FWCore_Framework_maker_TransitionWorker_h
/*----------------------------------------------------------------------    
*/

#include "FWCore/Framework/interface/maker/Worker.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "FWCore/Framework/interface/TransitionPhaseTypes.h"
namespace edm {
  class EventTransitionInfo;
  class RunTransitionInfo;
  class LumiTransitionInfo;
  class TransitionPhaseGlobal;
  class TransitionPhaseStream;

  template <typename TI, TransitionActionType T>
  struct TransitionActionContext;
  template <>
  struct TransitionActionContext<EventTransitionInfo, TransitionActionGlobalBegin> {
    using ContextType = StreamContext;
  };
  template <typename TI>
  struct TransitionActionContext<TI, TransitionActionGlobalBegin> {
    using ContextType = GlobalContext;
  };
  template <typename TI>
  struct TransitionActionContext<TI, TransitionActionStreamBegin> {
    using ContextType = StreamContext;
  };
  template <typename TI>
  struct TransitionActionContext<TI, TransitionActionStreamEnd> {
    using ContextType = StreamContext;
  };
  template <typename TI>
  struct TransitionActionContext<TI, TransitionActionGlobalEnd> {
    using ContextType = GlobalContext;
  };
  template <typename TI>
  struct TransitionActionContext<TI, TransitionActionProcessBlockInput> {
    using ContextType = GlobalContext;
  };
  template <typename TI, typename TP>
  class TransitionWorker : public Worker {
  public:
    TransitionWorker(ModuleDescription const& iMD, ExceptionToActionTable const* iActions) : Worker(iMD, iActions) {}
    ~TransitionWorker() override = default;

    template <TransitionActionType T>
    void doWorkAsync(WaitingTaskHolder iTask,
                     TI const& iTransitionInfo,
                     ServiceToken const& iToken,
                     StreamID iStreamID,
                     ParentContext const& iParentContext,
                     typename TransitionActionContext<TI, T>::ContextType const* iContext) noexcept {
      this->template doWorkAsyncImpl<OccurrenceTraits<typename TI::PrincipalType, T> >(
          std::move(iTask), iTransitionInfo, iToken, iStreamID, iParentContext, iContext);
    }

    //called by processOneOccurrenceAsync which is only used for globals by the SecondaryEventProvider and
    // WokerManager<stream>::processOneOccurrenceAsync
    template <TransitionActionType T>
    void doWorkNoPrefetchingAsync(WaitingTaskHolder iTask,
                                  TI const& iTransitionInfo,
                                  ServiceToken const& iToken,
                                  StreamID iStreamID,
                                  ParentContext const& iParentContext,
                                  typename TransitionActionContext<TI, T>::ContextType const* iContext) noexcept {
      this->template doWorkNoPrefetchingAsyncImpl<OccurrenceTraits<typename TI::PrincipalType, T> >(
          std::move(iTask), iTransitionInfo, iToken, iStreamID, iParentContext, iContext);
    }
  };

  using StreamRunWorker = TransitionWorker<RunTransitionInfo, TransitionPhaseStream>;
  using StreamLumiWorker = TransitionWorker<LumiTransitionInfo, TransitionPhaseStream>;
  using GlobalRunWorker = TransitionWorker<RunTransitionInfo, TransitionPhaseGlobal>;
  using GlobalLumiWorker = TransitionWorker<LumiTransitionInfo, TransitionPhaseGlobal>;
  using GlobalEventWorker = TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>;
}  // namespace edm
#endif

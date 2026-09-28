#ifndef FWCore_Framework_maker_TransitionWorker_h
#define FWCore_Framework_maker_TransitionWorker_h
/*----------------------------------------------------------------------    
*/

#include "FWCore/Framework/interface/maker/Worker.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "FWCore/Framework/interface/TransitionEdge.h"
#include "FWCore/Framework/interface/TransitionPhaseTypes.h"

namespace edm {
  class EventTransitionInfo;
  class RunTransitionInfo;
  class LumiTransitionInfo;
  class TransitionPhaseGlobal;
  class TransitionPhaseStream;

  template <typename TI, TransitionActionType T>
  struct TransitionActionContextTrait;
  template <>
  struct TransitionActionContextTrait<EventTransitionInfo, TransitionActionGlobalBegin> {
    using ContextType = StreamContext;
  };
  template <typename TI>
  struct TransitionActionContextTrait<TI, TransitionActionGlobalBegin> {
    using ContextType = GlobalContext;
  };
  template <typename TI>
  struct TransitionActionContextTrait<TI, TransitionActionStreamBegin> {
    using ContextType = StreamContext;
  };
  template <typename TI>
  struct TransitionActionContextTrait<TI, TransitionActionStreamEnd> {
    using ContextType = StreamContext;
  };
  template <typename TI>
  struct TransitionActionContextTrait<TI, TransitionActionGlobalEnd> {
    using ContextType = GlobalContext;
  };
  template <typename TI>
  struct TransitionActionContextTrait<TI, TransitionActionProcessBlockInput> {
    using ContextType = GlobalContext;
  };

  template <typename TI, typename TP, TransitionEdge E>
  struct TransitionActionTrait;
  template <typename TI>
  struct TransitionActionTrait<TI, TransitionPhaseGlobal, TransitionEdge::kBegin> {
    static constexpr TransitionActionType value = TransitionActionGlobalBegin;
  };
  template <typename TI>
  struct TransitionActionTrait<TI, TransitionPhaseGlobal, TransitionEdge::kEnd> {
    static constexpr TransitionActionType value = TransitionActionGlobalEnd;
  };
  template <typename TI>
  struct TransitionActionTrait<TI, TransitionPhaseStream, TransitionEdge::kBegin> {
    static constexpr TransitionActionType value = TransitionActionStreamBegin;
  };
  template <typename TI>
  struct TransitionActionTrait<TI, TransitionPhaseStream, TransitionEdge::kEnd> {
    static constexpr TransitionActionType value = TransitionActionStreamEnd;
  };
  template <>
  struct TransitionActionTrait<InputProcessBlockTransitionInfo, TransitionPhaseGlobal, TransitionEdge::kBegin> {
    static constexpr TransitionActionType value = TransitionActionProcessBlockInput;
  };

  template <typename TI, typename TP>
  class TransitionWorker : public Worker {
  public:
    TransitionWorker(ModuleDescription const& iMD, ExceptionToActionTable const* iActions) : Worker(iMD, iActions) {}
    ~TransitionWorker() override = default;

    template <TransitionEdge E>
    void doWorkAsync(
        WaitingTaskHolder iTask,
        TI const& iTransitionInfo,
        ServiceToken const& iToken,
        StreamID iStreamID,
        ParentContext const& iParentContext,
        typename TransitionActionContextTrait<TI, TransitionActionTrait<TI, TP, E>::value>::ContextType const*
            iContext) noexcept {
      this->template doWorkAsyncImpl<
          OccurrenceTraits<typename TI::PrincipalType, TransitionActionTrait<TI, TP, E>::value> >(
          std::move(iTask), iTransitionInfo, iToken, iStreamID, iParentContext, iContext);
    }

    //called by processOneOccurrenceAsync which is only used for globals by the SecondaryEventProvider and
    // WokerManager<stream>::processOneOccurrenceAsync
    template <TransitionEdge E>
    void doWorkNoPrefetchingAsync(
        WaitingTaskHolder iTask,
        TI const& iTransitionInfo,
        ServiceToken const& iToken,
        StreamID iStreamID,
        ParentContext const& iParentContext,
        typename TransitionActionContextTrait<TI, TransitionActionTrait<TI, TP, E>::value>::ContextType const*
            iContext) noexcept {
      this->template doWorkNoPrefetchingAsyncImpl<
          OccurrenceTraits<typename TI::PrincipalType, TransitionActionTrait<TI, TP, E>::value> >(
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

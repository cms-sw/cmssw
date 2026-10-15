#ifndef FWCore_Framework_OccurrenceTraits_h
#define FWCore_Framework_OccurrenceTraits_h

/*----------------------------------------------------------------------

OccurrenceTraits:

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/TransitionActionType.h"
#include "FWCore/Framework/interface/TransitionEdge.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/ProcessBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/Framework/interface/TransitionPhaseTypes.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/Utilities/interface/RunIndex.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"
#include "FWCore/Utilities/interface/Transition.h"

#include <string>

namespace edm {

  class ProcessContext;

  template <typename T, TransitionActionType B>
  class OccurrenceTraits;

  template <>
  class OccurrenceTraits<EventPrincipal, TransitionActionGlobalBegin> {
  public:
    using MyPrincipal = EventPrincipal;
    using TransitionInfoType = EventTransitionInfo;
    using Context = StreamContext;
    using TransitionPhaseType = TransitionPhaseGlobal;
    static TransitionActionType constexpr transitionAction_ = TransitionActionGlobalBegin;
    static BranchType constexpr branchType_ = InEvent;
    static TransitionEdge constexpr transitionEdge_ = TransitionEdge::kBegin;
    static bool constexpr isEvent_ = true;
    static Transition constexpr transition_ = Transition::Event;

    static void setStreamContext(StreamContext& streamContext, MyPrincipal const& principal) {
      streamContext.setTransition(StreamContext::Transition::kEvent);
      streamContext.setEventID(principal.id());
      streamContext.setTimestamp(principal.time());
    }

    static void preScheduleSignal(ActivityRegistry* a, StreamContext const* streamContext) {
      a->preEventSignal_.emit(*streamContext);
    }
    static void postScheduleSignal(ActivityRegistry* a, StreamContext const* streamContext) {
      a->postEventSignal_.emit(*streamContext);
    }
    static void prePathSignal(ActivityRegistry* a, PathContext const* pathContext) {
      a->prePathEventSignal_.emit(*pathContext->streamContext(), *pathContext);
    }
    static void postPathSignal(ActivityRegistry* a, HLTPathStatus const& status, PathContext const* pathContext) {
      a->postPathEventSignal_.emit(*pathContext->streamContext(), *pathContext, status);
    }

    static const char* transitionName() { return "Event"; }
  };

  template <>
  class OccurrenceTraits<RunPrincipal, TransitionActionGlobalBegin> {
  public:
    using MyPrincipal = RunPrincipal;
    using TransitionInfoType = RunTransitionInfo;
    using TransitionPhaseType = TransitionPhaseGlobal;
    static TransitionActionType constexpr transitionAction_ = TransitionActionGlobalBegin;
    using Context = GlobalContext;
    static BranchType constexpr branchType_ = InRun;
    static TransitionEdge constexpr transitionEdge_ = TransitionEdge::kBegin;
    static bool constexpr isEvent_ = false;
    static Transition constexpr transition_ = Transition::BeginRun;

    static GlobalContext makeGlobalContext(MyPrincipal const& principal, ProcessContext const* processContext) {
      return GlobalContext(GlobalContext::Transition::kBeginRun,
                           LuminosityBlockID(principal.run(), 0),
                           principal.index(),
                           LuminosityBlockIndex::invalidLuminosityBlockIndex(),
                           principal.beginTime(),
                           processContext);
    }

    static void preScheduleSignal(ActivityRegistry* a, GlobalContext const* globalContext) {
      a->preGlobalBeginRunSignal_.emit(*globalContext);
    }
    static void postScheduleSignal(ActivityRegistry* a, GlobalContext const* globalContext) {
      a->postGlobalBeginRunSignal_.emit(*globalContext);
    }
    static void prePathSignal(ActivityRegistry*, PathContext const*) {}
    static void postPathSignal(ActivityRegistry*, HLTPathStatus const&, PathContext const*) {}
    static const char* transitionName() { return "global begin Run"; }
  };

  template <>
  class OccurrenceTraits<RunPrincipal, TransitionActionStreamBegin> {
  public:
    using MyPrincipal = RunPrincipal;
    using TransitionInfoType = RunTransitionInfo;
    using TransitionPhaseType = TransitionPhaseStream;
    static TransitionActionType constexpr transitionAction_ = TransitionActionStreamBegin;
    using Context = StreamContext;
    static BranchType constexpr branchType_ = InRun;
    static TransitionEdge constexpr transitionEdge_ = TransitionEdge::kBegin;
    static bool constexpr isEvent_ = false;
    static Transition constexpr transition_ = Transition::BeginRun;

    static void setStreamContext(StreamContext& streamContext, MyPrincipal const& principal) {
      streamContext.setTransition(StreamContext::Transition::kBeginRun);
      streamContext.setEventID(EventID(principal.run(), 0, 0));
      streamContext.setRunIndex(principal.index());
      streamContext.setLuminosityBlockIndex(LuminosityBlockIndex::invalidLuminosityBlockIndex());
      streamContext.setTimestamp(principal.beginTime());
    }

    static void preScheduleSignal(ActivityRegistry* a, StreamContext const* streamContext) {
      a->preStreamBeginRunSignal_.emit(*streamContext);
    }
    static void postScheduleSignal(ActivityRegistry* a, StreamContext const* streamContext) {
      a->postStreamBeginRunSignal_.emit(*streamContext);
    }
    static void prePathSignal(ActivityRegistry*, PathContext const*) {}
    static void postPathSignal(ActivityRegistry*, HLTPathStatus const&, PathContext const*) {}
    static const char* transitionName() { return "stream begin Run"; }
  };

  template <>
  class OccurrenceTraits<RunPrincipal, TransitionActionStreamEnd> {
  public:
    using MyPrincipal = RunPrincipal;
    using TransitionInfoType = RunTransitionInfo;
    using TransitionPhaseType = TransitionPhaseStream;
    static TransitionActionType constexpr transitionAction_ = TransitionActionStreamEnd;
    using Context = StreamContext;
    static BranchType constexpr branchType_ = InRun;
    static TransitionEdge constexpr transitionEdge_ = TransitionEdge::kEnd;
    static bool constexpr isEvent_ = false;
    static Transition constexpr transition_ = Transition::EndRun;

    static void setStreamContext(StreamContext& streamContext, MyPrincipal const& principal) {
      streamContext.setTransition(StreamContext::Transition::kEndRun);
      streamContext.setEventID(EventID(principal.run(), 0, 0));
      streamContext.setRunIndex(principal.index());
      streamContext.setLuminosityBlockIndex(LuminosityBlockIndex::invalidLuminosityBlockIndex());
      streamContext.setTimestamp(principal.endTime());
    }

    static void preScheduleSignal(ActivityRegistry* a, StreamContext const* streamContext) {
      a->preStreamEndRunSignal_.emit(*streamContext);
    }
    static void postScheduleSignal(ActivityRegistry* a, StreamContext const* streamContext) {
      a->postStreamEndRunSignal_.emit(*streamContext);
    }
    static void prePathSignal(ActivityRegistry*, PathContext const*) {}
    static void postPathSignal(ActivityRegistry*, HLTPathStatus const&, PathContext const*) {}
    static const char* transitionName() { return "stream end Run"; }
  };

  template <>
  class OccurrenceTraits<RunPrincipal, TransitionActionGlobalEnd> {
  public:
    using MyPrincipal = RunPrincipal;
    using TransitionInfoType = RunTransitionInfo;
    using TransitionPhaseType = TransitionPhaseGlobal;
    static constexpr TransitionActionType transitionAction_ = TransitionActionGlobalEnd;
    using Context = GlobalContext;
    static BranchType constexpr branchType_ = InRun;
    static TransitionEdge constexpr transitionEdge_ = TransitionEdge::kEnd;
    static bool constexpr isEvent_ = false;
    static Transition constexpr transition_ = Transition::EndRun;

    static GlobalContext makeGlobalContext(MyPrincipal const& principal, ProcessContext const* processContext) {
      return GlobalContext(GlobalContext::Transition::kEndRun,
                           LuminosityBlockID(principal.run(), 0),
                           principal.index(),
                           LuminosityBlockIndex::invalidLuminosityBlockIndex(),
                           principal.endTime(),
                           processContext);
    }

    static void preScheduleSignal(ActivityRegistry* a, GlobalContext const* globalContext) {
      a->preGlobalEndRunSignal_.emit(*globalContext);
    }
    static void postScheduleSignal(ActivityRegistry* a, GlobalContext const* globalContext) {
      a->postGlobalEndRunSignal_.emit(*globalContext);
    }
    static void prePathSignal(ActivityRegistry*, PathContext const*) {}
    static void postPathSignal(ActivityRegistry*, HLTPathStatus const&, PathContext const*) {}
    static const char* transitionName() { return "global end Run"; }
  };

  template <>
  class OccurrenceTraits<LuminosityBlockPrincipal, TransitionActionGlobalBegin> {
  public:
    using MyPrincipal = LuminosityBlockPrincipal;
    using TransitionInfoType = LumiTransitionInfo;
    using TransitionPhaseType = TransitionPhaseGlobal;
    static constexpr TransitionActionType transitionAction_ = TransitionActionGlobalBegin;
    using Context = GlobalContext;
    static BranchType constexpr branchType_ = InLumi;
    static TransitionEdge constexpr transitionEdge_ = TransitionEdge::kBegin;
    static bool constexpr isEvent_ = false;
    static Transition constexpr transition_ = Transition::BeginLuminosityBlock;

    static GlobalContext makeGlobalContext(MyPrincipal const& principal, ProcessContext const* processContext) {
      return GlobalContext(GlobalContext::Transition::kBeginLuminosityBlock,
                           principal.id(),
                           principal.runPrincipal().index(),
                           principal.index(),
                           principal.beginTime(),
                           processContext);
    }

    static void preScheduleSignal(ActivityRegistry* a, GlobalContext const* globalContext) {
      a->preGlobalBeginLumiSignal_.emit(*globalContext);
    }
    static void postScheduleSignal(ActivityRegistry* a, GlobalContext const* globalContext) {
      a->postGlobalBeginLumiSignal_.emit(*globalContext);
    }
    static void prePathSignal(ActivityRegistry*, PathContext const*) {}
    static void postPathSignal(ActivityRegistry*, HLTPathStatus const&, PathContext const*) {}
    static const char* transitionName() { return "global begin LuminosityBlock"; }
  };

  template <>
  class OccurrenceTraits<LuminosityBlockPrincipal, TransitionActionStreamBegin> {
  public:
    using MyPrincipal = LuminosityBlockPrincipal;
    using TransitionInfoType = LumiTransitionInfo;
    using TransitionPhaseType = TransitionPhaseStream;
    static constexpr TransitionActionType transitionAction_ = TransitionActionStreamBegin;
    using Context = StreamContext;
    static BranchType constexpr branchType_ = InLumi;
    static TransitionEdge constexpr transitionEdge_ = TransitionEdge::kBegin;
    static bool constexpr isEvent_ = false;
    static Transition constexpr transition_ = Transition::BeginLuminosityBlock;

    static void setStreamContext(StreamContext& streamContext, MyPrincipal const& principal) {
      streamContext.setTransition(StreamContext::Transition::kBeginLuminosityBlock);
      streamContext.setEventID(EventID(principal.run(), principal.luminosityBlock(), 0));
      streamContext.setRunIndex(principal.runPrincipal().index());
      streamContext.setLuminosityBlockIndex(principal.index());
      streamContext.setTimestamp(principal.beginTime());
    }

    static void preScheduleSignal(ActivityRegistry* a, StreamContext const* streamContext) {
      a->preStreamBeginLumiSignal_.emit(*streamContext);
    }
    static void postScheduleSignal(ActivityRegistry* a, StreamContext const* streamContext) {
      a->postStreamBeginLumiSignal_.emit(*streamContext);
    }
    static void prePathSignal(ActivityRegistry*, PathContext const*) {}
    static void postPathSignal(ActivityRegistry*, HLTPathStatus const&, PathContext const*) {}
    static const char* transitionName() { return "stream begin LuminosityBlock"; }
  };

  template <>
  class OccurrenceTraits<LuminosityBlockPrincipal, TransitionActionStreamEnd> {
  public:
    using MyPrincipal = LuminosityBlockPrincipal;
    using TransitionInfoType = LumiTransitionInfo;
    using TransitionPhaseType = TransitionPhaseStream;
    static constexpr TransitionActionType transitionAction_ = TransitionActionStreamEnd;
    using Context = StreamContext;
    static BranchType constexpr branchType_ = InLumi;
    static TransitionEdge constexpr transitionEdge_ = TransitionEdge::kEnd;
    static bool constexpr isEvent_ = false;
    static Transition constexpr transition_ = Transition::EndLuminosityBlock;

    static StreamContext const* context(StreamContext const* s, GlobalContext const*) { return s; }

    static void setStreamContext(StreamContext& streamContext, MyPrincipal const& principal) {
      streamContext.setTransition(StreamContext::Transition::kEndLuminosityBlock);
      streamContext.setEventID(EventID(principal.run(), principal.luminosityBlock(), 0));
      streamContext.setRunIndex(principal.runPrincipal().index());
      streamContext.setLuminosityBlockIndex(principal.index());
      streamContext.setTimestamp(principal.endTime());
    }

    static void preScheduleSignal(ActivityRegistry* a, StreamContext const* streamContext) {
      a->preStreamEndLumiSignal_.emit(*streamContext);
    }
    static void postScheduleSignal(ActivityRegistry* a, StreamContext const* streamContext) {
      a->postStreamEndLumiSignal_.emit(*streamContext);
    }
    static void prePathSignal(ActivityRegistry*, PathContext const*) {}
    static void postPathSignal(ActivityRegistry*, HLTPathStatus const&, PathContext const*) {}
    static const char* transitionName() { return "end stream LuminosityBlock"; }
  };

  template <>
  class OccurrenceTraits<LuminosityBlockPrincipal, TransitionActionGlobalEnd> {
  public:
    using MyPrincipal = LuminosityBlockPrincipal;
    using TransitionInfoType = LumiTransitionInfo;
    using TransitionPhaseType = TransitionPhaseGlobal;
    static constexpr TransitionActionType transitionAction_ = TransitionActionGlobalEnd;
    using Context = GlobalContext;
    static BranchType constexpr branchType_ = InLumi;
    static TransitionEdge constexpr transitionEdge_ = TransitionEdge::kEnd;
    static bool constexpr isEvent_ = false;
    static Transition constexpr transition_ = Transition::EndLuminosityBlock;

    static GlobalContext makeGlobalContext(MyPrincipal const& principal, ProcessContext const* processContext) {
      return GlobalContext(GlobalContext::Transition::kEndLuminosityBlock,
                           principal.id(),
                           principal.runPrincipal().index(),
                           principal.index(),
                           principal.beginTime(),
                           processContext);
    }

    static void preScheduleSignal(ActivityRegistry* a, GlobalContext const* globalContext) {
      a->preGlobalEndLumiSignal_.emit(*globalContext);
    }
    static void postScheduleSignal(ActivityRegistry* a, GlobalContext const* globalContext) {
      a->postGlobalEndLumiSignal_.emit(*globalContext);
    }
    static void prePathSignal(ActivityRegistry*, PathContext const*) {}
    static void postPathSignal(ActivityRegistry*, HLTPathStatus const&, PathContext const*) {}
    static const char* transitionName() { return "end global LuminosityBlock"; }
  };

  template <>
  class OccurrenceTraits<ProcessBlockPrincipal, TransitionActionGlobalBegin> {
  public:
    using MyPrincipal = ProcessBlockPrincipal;
    using TransitionInfoType = ProcessBlockTransitionInfo;
    using TransitionPhaseType = TransitionPhaseGlobal;
    static constexpr TransitionActionType transitionAction_ = TransitionActionGlobalBegin;
    using Context = GlobalContext;
    static BranchType constexpr branchType_ = InProcess;
    static TransitionEdge constexpr transitionEdge_ = TransitionEdge::kBegin;
    static bool constexpr isEvent_ = false;
    static Transition constexpr transition_ = Transition::BeginProcessBlock;

    static GlobalContext makeGlobalContext(MyPrincipal const& principal, ProcessContext const* processContext) {
      return GlobalContext(GlobalContext::Transition::kBeginProcessBlock,
                           LuminosityBlockID(),
                           RunIndex::invalidRunIndex(),
                           LuminosityBlockIndex::invalidLuminosityBlockIndex(),
                           Timestamp::invalidTimestamp(),
                           processContext);
    }

    static void preScheduleSignal(ActivityRegistry* a, GlobalContext const* globalContext) {
      a->preBeginProcessBlockSignal_.emit(*globalContext);
    }
    static void postScheduleSignal(ActivityRegistry* a, GlobalContext const* globalContext) {
      a->postBeginProcessBlockSignal_.emit(*globalContext);
    }
    static const char* transitionName() { return "begin ProcessBlock"; }
  };

  template <>
  class OccurrenceTraits<ProcessBlockPrincipal, TransitionActionProcessBlockInput> {
  public:
    using MyPrincipal = ProcessBlockPrincipal;
    using TransitionInfoType = InputProcessBlockTransitionInfo;
    using TransitionPhaseType = TransitionPhaseGlobal;
    static constexpr TransitionActionType transitionAction_ = TransitionActionProcessBlockInput;
    using Context = GlobalContext;
    static BranchType constexpr branchType_ = InProcess;
    static TransitionEdge constexpr transitionEdge_ = TransitionEdge::kBegin;
    static bool constexpr isEvent_ = false;
    static Transition constexpr transition_ = Transition::AccessInputProcessBlock;

    static GlobalContext makeGlobalContext(MyPrincipal const& principal, ProcessContext const* processContext) {
      return GlobalContext(GlobalContext::Transition::kAccessInputProcessBlock,
                           LuminosityBlockID(),
                           RunIndex::invalidRunIndex(),
                           LuminosityBlockIndex::invalidLuminosityBlockIndex(),
                           Timestamp::invalidTimestamp(),
                           processContext);
    }

    static void preScheduleSignal(ActivityRegistry* a, GlobalContext const* globalContext) {
      a->preAccessInputProcessBlockSignal_.emit(*globalContext);
    }
    static void postScheduleSignal(ActivityRegistry* a, GlobalContext const* globalContext) {
      a->postAccessInputProcessBlockSignal_.emit(*globalContext);
    }
    static const char* transitionName() { return "access input ProcessBlock"; }
  };

  template <>
  class OccurrenceTraits<ProcessBlockPrincipal, TransitionActionGlobalEnd> {
  public:
    using MyPrincipal = ProcessBlockPrincipal;
    using TransitionInfoType = ProcessBlockTransitionInfo;
    using TransitionPhaseType = TransitionPhaseGlobal;
    static constexpr TransitionActionType transitionAction_ = TransitionActionGlobalEnd;
    using Context = GlobalContext;
    static BranchType constexpr branchType_ = InProcess;
    static TransitionEdge constexpr transitionEdge_ = TransitionEdge::kEnd;
    static bool constexpr isEvent_ = false;
    static Transition constexpr transition_ = Transition::EndProcessBlock;

    static GlobalContext makeGlobalContext(MyPrincipal const& principal, ProcessContext const* processContext) {
      return GlobalContext(GlobalContext::Transition::kEndProcessBlock,
                           LuminosityBlockID(),
                           RunIndex::invalidRunIndex(),
                           LuminosityBlockIndex::invalidLuminosityBlockIndex(),
                           Timestamp::invalidTimestamp(),
                           processContext);
    }

    static void preScheduleSignal(ActivityRegistry* a, GlobalContext const* globalContext) {
      a->preEndProcessBlockSignal_.emit(*globalContext);
    }
    static void postScheduleSignal(ActivityRegistry* a, GlobalContext const* globalContext) {
      a->postEndProcessBlockSignal_.emit(*globalContext);
    }
    static const char* transitionName() { return "end ProcessBlock"; }
  };

}  // namespace edm
#endif

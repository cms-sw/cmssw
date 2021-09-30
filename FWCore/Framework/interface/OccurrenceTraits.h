#ifndef FWCore_Framework_OccurrenceTraits_h
#define FWCore_Framework_OccurrenceTraits_h

/*----------------------------------------------------------------------

OccurrenceTraits:

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/BranchActionType.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/ProcessBlockPrincipal.h"
#include "FWCore/Utilities/interface/RunIndex.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/TransitionInfoTypes.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"
#include "FWCore/Utilities/interface/Transition.h"

#include <string>

namespace edm {

  class ProcessContext;

  template <typename T, BranchActionType B>
  class OccurrenceTraits;

  template <>
  class OccurrenceTraits<EventPrincipal, BranchActionStreamBegin> {
  public:
    using MyPrincipal = EventPrincipal;
    using TransitionInfoType = EventTransitionInfo;
    using Context = StreamContext;
    static BranchType constexpr branchType_ = InEvent;
    static bool constexpr begin_ = true;
    static bool constexpr isEvent_ = true;
    static Transition constexpr transition_ = Transition::Event;

    static void setStreamContext(StreamContext& streamContext, MyPrincipal const& principal) {
      streamContext.setTransition(StreamContext::Transition::kEvent);
      streamContext.setEventID(principal.id());
      streamContext.setTimestamp(principal.time());
    }

    static void preScheduleSignal(ActivityRegistry* a, StreamContext const* streamContext) {
      a->preEventSignal_(*streamContext);
    }
    static void postScheduleSignal(ActivityRegistry* a, StreamContext const* streamContext) {
      a->postEventSignal_(*streamContext);
    }
    static void prePathSignal(ActivityRegistry* a, PathContext const* pathContext) {
      a->prePathEventSignal_(*pathContext->streamContext(), *pathContext);
    }
    static void postPathSignal(ActivityRegistry* a, HLTPathStatus const& status, PathContext const* pathContext) {
      a->postPathEventSignal_(*pathContext->streamContext(), *pathContext, status);
    }

    static const char* transitionName() { return "Event"; }
  };

  template <>
  class OccurrenceTraits<RunPrincipal, BranchActionGlobalBegin> {
  public:
    using MyPrincipal = RunPrincipal;
    using TransitionInfoType = RunTransitionInfo;
    using Context = GlobalContext;
    static BranchType constexpr branchType_ = InRun;
    static bool constexpr begin_ = true;
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
      a->preGlobalBeginRunSignal_(*globalContext);
    }
    static void postScheduleSignal(ActivityRegistry* a, GlobalContext const* globalContext) {
      a->postGlobalBeginRunSignal_(*globalContext);
    }
    static void prePathSignal(ActivityRegistry*, PathContext const*) {}
    static void postPathSignal(ActivityRegistry*, HLTPathStatus const&, PathContext const*) {}
    static void preModuleSignal(ActivityRegistry* a,
                                GlobalContext const* globalContext,
                                ModuleCallingContext const* moduleCallingContext) {
      a->preModuleGlobalBeginRunSignal_(*globalContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry* a,
                                 GlobalContext const* globalContext,
                                 ModuleCallingContext const* moduleCallingContext) {
      a->postModuleGlobalBeginRunSignal_(*globalContext, *moduleCallingContext);
    }
    static const char* transitionName() { return "global begin Run"; }
  };

  template <>
  class OccurrenceTraits<RunPrincipal, BranchActionStreamBegin> {
  public:
    using MyPrincipal = RunPrincipal;
    using TransitionInfoType = RunTransitionInfo;
    using Context = StreamContext;
    static BranchType constexpr branchType_ = InRun;
    static bool constexpr begin_ = true;
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
      a->preStreamBeginRunSignal_(*streamContext);
    }
    static void postScheduleSignal(ActivityRegistry* a, StreamContext const* streamContext) {
      a->postStreamBeginRunSignal_(*streamContext);
    }
    static void prePathSignal(ActivityRegistry*, PathContext const*) {}
    static void postPathSignal(ActivityRegistry*, HLTPathStatus const&, PathContext const*) {}
    static void preModuleSignal(ActivityRegistry* a,
                                StreamContext const* streamContext,
                                ModuleCallingContext const* moduleCallingContext) {
      a->preModuleStreamBeginRunSignal_(*streamContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry* a,
                                 StreamContext const* streamContext,
                                 ModuleCallingContext const* moduleCallingContext) {
      a->postModuleStreamBeginRunSignal_(*streamContext, *moduleCallingContext);
    }
    static const char* transitionName() { return "stream begin Run"; }
  };

  template <>
  class OccurrenceTraits<RunPrincipal, BranchActionStreamEnd> {
  public:
    using MyPrincipal = RunPrincipal;
    using TransitionInfoType = RunTransitionInfo;
    using Context = StreamContext;
    static BranchType constexpr branchType_ = InRun;
    static bool constexpr begin_ = false;
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
      a->preStreamEndRunSignal_(*streamContext);
    }
    static void postScheduleSignal(ActivityRegistry* a, StreamContext const* streamContext) {
      a->postStreamEndRunSignal_(*streamContext);
    }
    static void prePathSignal(ActivityRegistry*, PathContext const*) {}
    static void postPathSignal(ActivityRegistry*, HLTPathStatus const&, PathContext const*) {}
    static void preModuleSignal(ActivityRegistry* a,
                                StreamContext const* streamContext,
                                ModuleCallingContext const* moduleCallingContext) {
      a->preModuleStreamEndRunSignal_(*streamContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry* a,
                                 StreamContext const* streamContext,
                                 ModuleCallingContext const* moduleCallingContext) {
      a->postModuleStreamEndRunSignal_(*streamContext, *moduleCallingContext);
    }
    static const char* transitionName() { return "stream end Run"; }
  };

  template <>
  class OccurrenceTraits<RunPrincipal, BranchActionGlobalEnd> {
  public:
    using MyPrincipal = RunPrincipal;
    using TransitionInfoType = RunTransitionInfo;
    using Context = GlobalContext;
    static BranchType constexpr branchType_ = InRun;
    static bool constexpr begin_ = false;
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
      a->preGlobalEndRunSignal_(*globalContext);
    }
    static void postScheduleSignal(ActivityRegistry* a, GlobalContext const* globalContext) {
      a->postGlobalEndRunSignal_(*globalContext);
    }
    static void prePathSignal(ActivityRegistry*, PathContext const*) {}
    static void postPathSignal(ActivityRegistry*, HLTPathStatus const&, PathContext const*) {}
    static void preModuleSignal(ActivityRegistry* a,
                                GlobalContext const* globalContext,
                                ModuleCallingContext const* moduleCallingContext) {
      a->preModuleGlobalEndRunSignal_(*globalContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry* a,
                                 GlobalContext const* globalContext,
                                 ModuleCallingContext const* moduleCallingContext) {
      a->postModuleGlobalEndRunSignal_(*globalContext, *moduleCallingContext);
    }
    static const char* transitionName() { return "global end Run"; }
  };

  template <>
  class OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalBegin> {
  public:
    using MyPrincipal = LuminosityBlockPrincipal;
    using TransitionInfoType = LumiTransitionInfo;
    using Context = GlobalContext;
    static BranchType constexpr branchType_ = InLumi;
    static bool constexpr begin_ = true;
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
      a->preGlobalBeginLumiSignal_(*globalContext);
    }
    static void postScheduleSignal(ActivityRegistry* a, GlobalContext const* globalContext) {
      a->postGlobalBeginLumiSignal_(*globalContext);
    }
    static void prePathSignal(ActivityRegistry*, PathContext const*) {}
    static void postPathSignal(ActivityRegistry*, HLTPathStatus const&, PathContext const*) {}
    static void preModuleSignal(ActivityRegistry* a,
                                GlobalContext const* globalContext,
                                ModuleCallingContext const* moduleCallingContext) {
      a->preModuleGlobalBeginLumiSignal_(*globalContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry* a,
                                 GlobalContext const* globalContext,
                                 ModuleCallingContext const* moduleCallingContext) {
      a->postModuleGlobalBeginLumiSignal_(*globalContext, *moduleCallingContext);
    }
    static const char* transitionName() { return "global begin LuminosityBlock"; }
  };

  template <>
  class OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamBegin> {
  public:
    using MyPrincipal = LuminosityBlockPrincipal;
    using TransitionInfoType = LumiTransitionInfo;
    using Context = StreamContext;
    static BranchType constexpr branchType_ = InLumi;
    static bool constexpr begin_ = true;
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
      a->preStreamBeginLumiSignal_(*streamContext);
    }
    static void postScheduleSignal(ActivityRegistry* a, StreamContext const* streamContext) {
      a->postStreamBeginLumiSignal_(*streamContext);
    }
    static void prePathSignal(ActivityRegistry*, PathContext const*) {}
    static void postPathSignal(ActivityRegistry*, HLTPathStatus const&, PathContext const*) {}
    static void preModuleSignal(ActivityRegistry* a,
                                StreamContext const* streamContext,
                                ModuleCallingContext const* moduleCallingContext) {
      a->preModuleStreamBeginLumiSignal_(*streamContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry* a,
                                 StreamContext const* streamContext,
                                 ModuleCallingContext const* moduleCallingContext) {
      a->postModuleStreamBeginLumiSignal_(*streamContext, *moduleCallingContext);
    }
    static const char* transitionName() { return "stream begin LuminosityBlock"; }
  };

  template <>
  class OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamEnd> {
  public:
    using MyPrincipal = LuminosityBlockPrincipal;
    using TransitionInfoType = LumiTransitionInfo;
    using Context = StreamContext;
    static BranchType constexpr branchType_ = InLumi;
    static bool constexpr begin_ = false;
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
      a->preStreamEndLumiSignal_(*streamContext);
    }
    static void postScheduleSignal(ActivityRegistry* a, StreamContext const* streamContext) {
      a->postStreamEndLumiSignal_(*streamContext);
    }
    static void prePathSignal(ActivityRegistry*, PathContext const*) {}
    static void postPathSignal(ActivityRegistry*, HLTPathStatus const&, PathContext const*) {}
    static void preModuleSignal(ActivityRegistry* a,
                                StreamContext const* streamContext,
                                ModuleCallingContext const* moduleCallingContext) {
      a->preModuleStreamEndLumiSignal_(*streamContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry* a,
                                 StreamContext const* streamContext,
                                 ModuleCallingContext const* moduleCallingContext) {
      a->postModuleStreamEndLumiSignal_(*streamContext, *moduleCallingContext);
    }
    static const char* transitionName() { return "end stream LuminosityBlock"; }
  };

  template <>
  class OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalEnd> {
  public:
    using MyPrincipal = LuminosityBlockPrincipal;
    using TransitionInfoType = LumiTransitionInfo;
    using Context = GlobalContext;
    static BranchType constexpr branchType_ = InLumi;
    static bool constexpr begin_ = false;
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
      a->preGlobalEndLumiSignal_(*globalContext);
    }
    static void postScheduleSignal(ActivityRegistry* a, GlobalContext const* globalContext) {
      a->postGlobalEndLumiSignal_(*globalContext);
    }
    static void prePathSignal(ActivityRegistry*, PathContext const*) {}
    static void postPathSignal(ActivityRegistry*, HLTPathStatus const&, PathContext const*) {}
    static void preModuleSignal(ActivityRegistry* a,
                                GlobalContext const* globalContext,
                                ModuleCallingContext const* moduleCallingContext) {
      a->preModuleGlobalEndLumiSignal_(*globalContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry* a,
                                 GlobalContext const* globalContext,
                                 ModuleCallingContext const* moduleCallingContext) {
      a->postModuleGlobalEndLumiSignal_(*globalContext, *moduleCallingContext);
    }
    static const char* transitionName() { return "end global LuminosityBlock"; }
  };

  template <>
  class OccurrenceTraits<ProcessBlockPrincipal, BranchActionGlobalBegin> {
  public:
    using MyPrincipal = ProcessBlockPrincipal;
    using TransitionInfoType = ProcessBlockTransitionInfo;
    using Context = GlobalContext;
    static BranchType constexpr branchType_ = InProcess;
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
      a->preBeginProcessBlockSignal_(*globalContext);
    }
    static void postScheduleSignal(ActivityRegistry* a, GlobalContext const* globalContext) {
      a->postBeginProcessBlockSignal_(*globalContext);
    }
    static void preModuleSignal(ActivityRegistry* a,
                                GlobalContext const* globalContext,
                                ModuleCallingContext const* moduleCallingContext) {
      a->preModuleBeginProcessBlockSignal_(*globalContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry* a,
                                 GlobalContext const* globalContext,
                                 ModuleCallingContext const* moduleCallingContext) {
      a->postModuleBeginProcessBlockSignal_(*globalContext, *moduleCallingContext);
    }
    static const char* transitionName() { return "begin ProcessBlock"; }
  };

  template <>
  class OccurrenceTraits<ProcessBlockPrincipal, BranchActionProcessBlockInput> {
  public:
    using MyPrincipal = ProcessBlockPrincipal;
    using TransitionInfoType = ProcessBlockTransitionInfo;
    using Context = GlobalContext;
    static BranchType constexpr branchType_ = InProcess;
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
      a->preAccessInputProcessBlockSignal_(*globalContext);
    }
    static void postScheduleSignal(ActivityRegistry* a, GlobalContext const* globalContext) {
      a->postAccessInputProcessBlockSignal_(*globalContext);
    }
    static void preModuleSignal(ActivityRegistry* a,
                                GlobalContext const* globalContext,
                                ModuleCallingContext const* moduleCallingContext) {
      a->preModuleAccessInputProcessBlockSignal_(*globalContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry* a,
                                 GlobalContext const* globalContext,
                                 ModuleCallingContext const* moduleCallingContext) {
      a->postModuleAccessInputProcessBlockSignal_(*globalContext, *moduleCallingContext);
    }
    static const char* transitionName() { return "access input ProcessBlock"; }
  };

  template <>
  class OccurrenceTraits<ProcessBlockPrincipal, BranchActionGlobalEnd> {
  public:
    using MyPrincipal = ProcessBlockPrincipal;
    using TransitionInfoType = ProcessBlockTransitionInfo;
    using Context = GlobalContext;
    static BranchType constexpr branchType_ = InProcess;
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
      a->preEndProcessBlockSignal_(*globalContext);
    }
    static void postScheduleSignal(ActivityRegistry* a, GlobalContext const* globalContext) {
      a->postEndProcessBlockSignal_(*globalContext);
    }
    static void preModuleSignal(ActivityRegistry* a,
                                GlobalContext const* globalContext,
                                ModuleCallingContext const* moduleCallingContext) {
      a->preModuleEndProcessBlockSignal_(*globalContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry* a,
                                 GlobalContext const* globalContext,
                                 ModuleCallingContext const* moduleCallingContext) {
      a->postModuleEndProcessBlockSignal_(*globalContext, *moduleCallingContext);
    }
    static const char* transitionName() { return "end ProcessBlock"; }
  };

}  // namespace edm
#endif

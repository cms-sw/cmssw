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
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"

#include<string>

namespace edm {

  class ProcessContext;

  template <typename T, BranchActionType B> class OccurrenceTraits;

  template <>
  class OccurrenceTraits<EventPrincipal, BranchActionStreamBegin> {
  public:
    typedef EventPrincipal MyPrincipal;
    typedef StreamContext Context;
    static BranchType const branchType_ = InEvent;
    static bool const begin_ = true;
    static bool const isEvent_ = true;

    static void setStreamContext(StreamContext& streamContext, MyPrincipal const& principal) {
      streamContext.setTransition(StreamContext::Transition::kEvent);
      streamContext.setEventID(principal.id());
      streamContext.setTimestamp(principal.time());
    }

    static void preScheduleSignal(ActivityRegistry *a, StreamContext const* streamContext) {
      a->preEventSignal_(*streamContext);
    }
    static void postScheduleSignal(ActivityRegistry *a, StreamContext const* streamContext) {
      a->postEventSignal_(*streamContext);
    }
    static void prePathSignal(ActivityRegistry *a, PathContext const* pathContext) {
      a->prePathEventSignal_(*pathContext->streamContext(), *pathContext);
    }
    static void postPathSignal(ActivityRegistry *a, HLTPathStatus const& status, PathContext const* pathContext) {
      a->postPathEventSignal_(*pathContext->streamContext(), *pathContext, status);
    }
    static void preModuleSignal(ActivityRegistry *a, StreamContext const* streamContext, ModuleCallingContext const*  moduleCallingContext) {
      a->preModuleEventSignal_(*streamContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry *a, StreamContext const* streamContext, ModuleCallingContext const*  moduleCallingContext) {
      a->postModuleEventSignal_(*streamContext, *moduleCallingContext);
    }
  };

  template <>
  class OccurrenceTraits<RunPrincipal, BranchActionGlobalBegin> {
  public:
    typedef RunPrincipal MyPrincipal;
    typedef GlobalContext Context;
    static BranchType const branchType_ = InRun;
    static bool const begin_ = true;
    static bool const isEvent_ = false;

    static GlobalContext makeGlobalContext(MyPrincipal const& principal, ProcessContext const* processContext) {
      return GlobalContext(GlobalContext::Transition::kBeginRun,
                           LuminosityBlockID(principal.run(), 0),
                           principal.index(),
                           LuminosityBlockIndex::invalidLuminosityBlockIndex(),
                           principal.beginTime(),
                           processContext);
    }

    static void preScheduleSignal(ActivityRegistry *a, GlobalContext const* globalContext) {
      a->preGlobalBeginRunSignal_(*globalContext);
    }
    static void postScheduleSignal(ActivityRegistry *a, GlobalContext const* globalContext) {
      a->postGlobalBeginRunSignal_(*globalContext);
    }
    static void prePathSignal(ActivityRegistry *, PathContext const* ) {
    }
    static void postPathSignal(ActivityRegistry *, HLTPathStatus const& , PathContext const* ) {
    }
    static void preModuleSignal(ActivityRegistry *a, GlobalContext const* globalContext, ModuleCallingContext const*  moduleCallingContext) {
      a->preModuleGlobalBeginRunSignal_(*globalContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry *a, GlobalContext const* globalContext, ModuleCallingContext const*  moduleCallingContext) {
      a->postModuleGlobalBeginRunSignal_(*globalContext, *moduleCallingContext);
    }
  };

  template <>
  class OccurrenceTraits<RunPrincipal, BranchActionStreamBegin> {
  public:
    typedef RunPrincipal MyPrincipal;
    typedef StreamContext Context;
    static BranchType const branchType_ = InRun;
    static bool const begin_ = true;
    static bool const isEvent_ = false;

    static void setStreamContext(StreamContext& streamContext, MyPrincipal const& principal) {
      streamContext.setTransition(StreamContext::Transition::kBeginRun);
      streamContext.setEventID(EventID(principal.run(), 0, 0));
      streamContext.setRunIndex(principal.index());
      streamContext.setLuminosityBlockIndex(LuminosityBlockIndex::invalidLuminosityBlockIndex());
      streamContext.setTimestamp(principal.beginTime());
    }

    static void preScheduleSignal(ActivityRegistry *a, StreamContext const* streamContext) {
      a->preStreamBeginRunSignal_(*streamContext);
    }
    static void postScheduleSignal(ActivityRegistry *a, StreamContext const* streamContext) {
      a->postStreamBeginRunSignal_(*streamContext);
    }
    static void prePathSignal(ActivityRegistry *, PathContext const* ) {
    }
    static void postPathSignal(ActivityRegistry *, HLTPathStatus const& , PathContext const*) {
    }
    static void preModuleSignal(ActivityRegistry *a, StreamContext const* streamContext, ModuleCallingContext const*  moduleCallingContext) {
      a->preModuleStreamBeginRunSignal_(*streamContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry *a, StreamContext const* streamContext, ModuleCallingContext const*  moduleCallingContext) {
      a->postModuleStreamBeginRunSignal_(*streamContext, *moduleCallingContext);
    }
  };

  template <>
  class OccurrenceTraits<RunPrincipal, BranchActionStreamEnd> {
  public:
    typedef RunPrincipal MyPrincipal;
    typedef StreamContext Context;
    static BranchType const branchType_ = InRun;
    static bool const begin_ = false;
    static bool const isEvent_ = false;

    static void setStreamContext(StreamContext& streamContext, MyPrincipal const& principal) {
      streamContext.setTransition(StreamContext::Transition::kEndRun);
      streamContext.setEventID(EventID(principal.run(), 0, 0));
      streamContext.setRunIndex(principal.index());
      streamContext.setLuminosityBlockIndex(LuminosityBlockIndex::invalidLuminosityBlockIndex());
      streamContext.setTimestamp(principal.endTime());
    }

    static void preScheduleSignal(ActivityRegistry *a, StreamContext const* streamContext) {
      a->preStreamEndRunSignal_(*streamContext);
    }
    static void postScheduleSignal(ActivityRegistry *a, StreamContext const* streamContext) {
      a->postStreamEndRunSignal_(*streamContext);
    }
    static void prePathSignal(ActivityRegistry *, PathContext const* ) {
    }
    static void postPathSignal(ActivityRegistry *, HLTPathStatus const& , PathContext const* ) {
    }
    static void preModuleSignal(ActivityRegistry *a, StreamContext const* streamContext, ModuleCallingContext const*  moduleCallingContext) {
      a->preModuleStreamEndRunSignal_(*streamContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry *a, StreamContext const* streamContext, ModuleCallingContext const*  moduleCallingContext) {
      a->postModuleStreamEndRunSignal_(*streamContext, *moduleCallingContext);
    }
  };

  template <>
  class OccurrenceTraits<RunPrincipal, BranchActionGlobalEnd> {
  public:
    typedef RunPrincipal MyPrincipal;
    typedef GlobalContext Context;
    static BranchType const branchType_ = InRun;
    static bool const begin_ = false;
    static bool const isEvent_ = false;

    static GlobalContext makeGlobalContext(MyPrincipal const& principal, ProcessContext const* processContext) {
      return GlobalContext(GlobalContext::Transition::kEndRun,
                           LuminosityBlockID(principal.run(), 0),
                           principal.index(),
                           LuminosityBlockIndex::invalidLuminosityBlockIndex(),
                           principal.endTime(),
                           processContext);
    }

    static void preScheduleSignal(ActivityRegistry *a, GlobalContext const* globalContext) {
      a->preGlobalEndRunSignal_(*globalContext);
    }
    static void postScheduleSignal(ActivityRegistry *a, GlobalContext const* globalContext) {
      a->postGlobalEndRunSignal_(*globalContext);
    }
    static void prePathSignal(ActivityRegistry *, PathContext const*) {
    }
    static void postPathSignal(ActivityRegistry *, HLTPathStatus const& , PathContext const* ) {
    }
    static void preModuleSignal(ActivityRegistry *a, GlobalContext const* globalContext, ModuleCallingContext const*  moduleCallingContext) {
      a->preModuleGlobalEndRunSignal_(*globalContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry *a, GlobalContext const* globalContext, ModuleCallingContext const*  moduleCallingContext) {
      a->postModuleGlobalEndRunSignal_(*globalContext, *moduleCallingContext);
    }
  };

  template <>
  class OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalBegin> {
  public:
    typedef LuminosityBlockPrincipal MyPrincipal;
    typedef GlobalContext Context;
    static BranchType const branchType_ = InLumi;
    static bool const begin_ = true;
    static bool const isEvent_ = false;

    static GlobalContext makeGlobalContext(MyPrincipal const& principal, ProcessContext const* processContext) {
      return GlobalContext(GlobalContext::Transition::kBeginLuminosityBlock,
                           principal.id(),
                           principal.runPrincipal().index(),
                           principal.index(),
                           principal.beginTime(),
                           processContext);
    }

    static void preScheduleSignal(ActivityRegistry *a, GlobalContext const* globalContext) {
      a->preGlobalBeginLumiSignal_(*globalContext);
    }
    static void postScheduleSignal(ActivityRegistry *a, GlobalContext const* globalContext) {
      a->postGlobalBeginLumiSignal_(*globalContext);
    }
    static void prePathSignal(ActivityRegistry *, PathContext const*) {
    }
    static void postPathSignal(ActivityRegistry *, HLTPathStatus const&, PathContext const*) {
    }
    static void preModuleSignal(ActivityRegistry *a, GlobalContext const* globalContext, ModuleCallingContext const*  moduleCallingContext) {
      a->preModuleGlobalBeginLumiSignal_(*globalContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry *a, GlobalContext const* globalContext, ModuleCallingContext const*  moduleCallingContext) {
      a->postModuleGlobalBeginLumiSignal_(*globalContext, *moduleCallingContext);
    }
  };

  template <>
  class OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamBegin> {
  public:
    typedef LuminosityBlockPrincipal MyPrincipal;
    typedef StreamContext Context;
    static BranchType const branchType_ = InLumi;
    static bool const begin_ = true;
    static bool const isEvent_ = false;

    static void setStreamContext(StreamContext& streamContext, MyPrincipal const& principal) {
      streamContext.setTransition(StreamContext::Transition::kBeginLuminosityBlock);
      streamContext.setEventID(EventID(principal.run(), principal.luminosityBlock(), 0));
      streamContext.setRunIndex(principal.runPrincipal().index());
      streamContext.setLuminosityBlockIndex(principal.index());
      streamContext.setTimestamp(principal.beginTime());
    }

    static void preScheduleSignal(ActivityRegistry *a, StreamContext const* streamContext) {
      a->preStreamBeginLumiSignal_(*streamContext);
    }
    static void postScheduleSignal(ActivityRegistry *a, StreamContext const* streamContext) {
      a->postStreamBeginLumiSignal_(*streamContext);
    }
    static void prePathSignal(ActivityRegistry *, PathContext const*) {
    }
    static void postPathSignal(ActivityRegistry *, HLTPathStatus const&, PathContext const*) {
    }
    static void preModuleSignal(ActivityRegistry *a, StreamContext const* streamContext, ModuleCallingContext const*  moduleCallingContext) {
      a->preModuleStreamBeginLumiSignal_(*streamContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry *a, StreamContext const* streamContext, ModuleCallingContext const*  moduleCallingContext) {
      a->postModuleStreamBeginLumiSignal_(*streamContext, *moduleCallingContext);
    }
  };

  template <>
  class OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamEnd> {
  public:
    typedef LuminosityBlockPrincipal MyPrincipal;
    typedef StreamContext Context;
    static BranchType const branchType_ = InLumi;
    static bool const begin_ = false;
    static bool const isEvent_ = false;

    static StreamContext const* context(StreamContext const* s, GlobalContext const*) { return s; }

    static void setStreamContext(StreamContext& streamContext, MyPrincipal const& principal) {
      streamContext.setTransition(StreamContext::Transition::kEndLuminosityBlock);
      streamContext.setEventID(EventID(principal.run(), principal.luminosityBlock(), 0));
      streamContext.setRunIndex(principal.runPrincipal().index());
      streamContext.setLuminosityBlockIndex(principal.index());
      streamContext.setTimestamp(principal.endTime());
    }

    static void preScheduleSignal(ActivityRegistry *a, StreamContext const* streamContext) {
      a->preStreamEndLumiSignal_(*streamContext);
    }
    static void postScheduleSignal(ActivityRegistry *a, StreamContext const* streamContext) {
      a->postStreamEndLumiSignal_(*streamContext);
    }
    static void prePathSignal(ActivityRegistry *, PathContext const* ) {
    }
    static void postPathSignal(ActivityRegistry *, HLTPathStatus const&, PathContext const*) {
    }
    static void preModuleSignal(ActivityRegistry *a, StreamContext const* streamContext, ModuleCallingContext const*  moduleCallingContext) {
      a->preModuleStreamEndLumiSignal_(*streamContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry *a, StreamContext const* streamContext, ModuleCallingContext const*  moduleCallingContext) {
      a->postModuleStreamEndLumiSignal_(*streamContext, *moduleCallingContext);
    }
  };

  template <>
  class OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalEnd> {
  public:
    typedef LuminosityBlockPrincipal MyPrincipal;
    typedef GlobalContext Context;
    static BranchType const branchType_ = InLumi;
    static bool const begin_ = false;
    static bool const isEvent_ = false;

    static GlobalContext makeGlobalContext(MyPrincipal const& principal, ProcessContext const* processContext) {
      return GlobalContext(GlobalContext::Transition::kEndLuminosityBlock,
                           principal.id(),
                           principal.runPrincipal().index(),
                           principal.index(),
                           principal.beginTime(),
                           processContext);
    }

    static void preScheduleSignal(ActivityRegistry *a, GlobalContext const* globalContext) {
      a->preGlobalEndLumiSignal_(*globalContext);
    }
    static void postScheduleSignal(ActivityRegistry *a, GlobalContext const* globalContext) {
      a->postGlobalEndLumiSignal_(*globalContext);
    }
    static void prePathSignal(ActivityRegistry *, PathContext const*) {
    }
    static void postPathSignal(ActivityRegistry *, HLTPathStatus const& , PathContext const* ) {
    }
    static void preModuleSignal(ActivityRegistry *a, GlobalContext const* globalContext, ModuleCallingContext const*  moduleCallingContext) {
      a->preModuleGlobalEndLumiSignal_(*globalContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry *a, GlobalContext const* globalContext, ModuleCallingContext const*  moduleCallingContext) {
      a->postModuleGlobalEndLumiSignal_(*globalContext, *moduleCallingContext);
    }
  };
}
#endif

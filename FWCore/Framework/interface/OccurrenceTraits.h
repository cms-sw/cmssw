#ifndef FWCore_Framework_OccurrenceTraits_h
#define FWCore_Framework_OccurrenceTraits_h

/*----------------------------------------------------------------------
  
OccurrenceTraits: 

----------------------------------------------------------------------*/

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

#include<string>

namespace edm {
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

    static void preScheduleSignal(ActivityRegistry *a, EventPrincipal const* ep, StreamContext const* streamContext) {
      a->preProcessEventSignal_(ep->id(), ep->time()); 
      a->preEventSignal_(*streamContext); 
    }
    static void postScheduleSignal(ActivityRegistry *a, EventPrincipal* ep, EventSetup const* es, StreamContext const* streamContext) {
      edm::ModuleDescription modDesc("postScheduledSignal", "");
      ParentContext parentContext(streamContext);
      ModuleCallingContext moduleCallingContext(&modDesc, ModuleCallingContext::State::kRunning, parentContext);
      Event ev(*ep, ModuleDescription(), &moduleCallingContext);
      a->postProcessEventSignal_(ev, *es);
      a->postEventSignal_(*streamContext, ev, *es);
    }
    static void prePathSignal(ActivityRegistry *a, std::string const& s, PathContext const* pathContext) {
      a->preProcessPathSignal_(s);
      a->prePathEventSignal_(*pathContext->streamContext(), *pathContext); 
    }
    static void postPathSignal(ActivityRegistry *a, std::string const& s, HLTPathStatus const& status, PathContext const* pathContext) {
      a->postProcessPathSignal_(s, status); 
      a->postPathEventSignal_(*pathContext->streamContext(), *pathContext, status); 
    }
    static void preModuleSignal(ActivityRegistry *a, ModuleDescription const* md, StreamContext const* streamContext, ModuleCallingContext const*  moduleCallingContext) {
      a->preModuleSignal_(*md); 
      a->preModuleEventSignal_(*streamContext, *moduleCallingContext); 
    }
    static void postModuleSignal(ActivityRegistry *a, ModuleDescription const* md, StreamContext const* streamContext, ModuleCallingContext const*  moduleCallingContext) {
      a->postModuleSignal_(*md); 
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

    static void preScheduleSignal(ActivityRegistry *a, RunPrincipal const* ep, GlobalContext const* globalContext) {
      a->preBeginRunSignal_(ep->id(), ep->beginTime());
      a->preGlobalBeginRunSignal_(*globalContext);
    }
    static void postScheduleSignal(ActivityRegistry *a, RunPrincipal* ep, EventSetup const* es, GlobalContext const* globalContext) {
      // Next 4 lines not needed when we go to threaded interface
      edm::ModuleDescription modDesc("postScheduledSignal", "");
      ParentContext parentContext(globalContext);
      ModuleCallingContext moduleCallingContext(&modDesc, ModuleCallingContext::State::kRunning, parentContext);
      Run run(*ep, ModuleDescription(), &moduleCallingContext);
      a->postBeginRunSignal_(run, *es);
      a->postGlobalBeginRunSignal_(*globalContext);
    }
    static void prePathSignal(ActivityRegistry *a, std::string const& s, PathContext const* pathContext) {
      a->prePathBeginRunSignal_(s); 
    }
    static void postPathSignal(ActivityRegistry *a, std::string const& s, HLTPathStatus const& status, PathContext const* pathContext) {
      a->postPathBeginRunSignal_(s, status); 
    }
    static void preModuleSignal(ActivityRegistry *a, ModuleDescription const* md, GlobalContext const* globalContext, ModuleCallingContext const*  moduleCallingContext) {
      a->preModuleBeginRunSignal_(*md); 
      a->preModuleGlobalBeginRunSignal_(*globalContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry *a, ModuleDescription const* md, GlobalContext const* globalContext, ModuleCallingContext const*  moduleCallingContext) {
      a->postModuleBeginRunSignal_(*md); 
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

    static void preScheduleSignal(ActivityRegistry *a, RunPrincipal const* ep, StreamContext const* streamContext) {
      a->preStreamBeginRunSignal_(*streamContext);
    }
    static void postScheduleSignal(ActivityRegistry *a, RunPrincipal* ep, EventSetup const* es, StreamContext const* streamContext) {
      a->postStreamBeginRunSignal_(*streamContext);
    }
    static void prePathSignal(ActivityRegistry *a, std::string const& s, PathContext const* pathContext) {
    }
    static void postPathSignal(ActivityRegistry *a, std::string const& s, HLTPathStatus const& status, PathContext const* pathContext) {
    }
    static void preModuleSignal(ActivityRegistry *a, ModuleDescription const* md, StreamContext const* streamContext, ModuleCallingContext const*  moduleCallingContext) {
      a->preModuleStreamBeginRunSignal_(*streamContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry *a, ModuleDescription const* md, StreamContext const* streamContext, ModuleCallingContext const*  moduleCallingContext) {
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

    static void preScheduleSignal(ActivityRegistry *a, RunPrincipal const* ep, StreamContext const* streamContext) {
      a->preStreamEndRunSignal_(*streamContext);
    }
    static void postScheduleSignal(ActivityRegistry *a, RunPrincipal* ep, EventSetup const* es, StreamContext const* streamContext) {
      edm::ModuleDescription modDesc("postScheduledSignal", "");
      ParentContext parentContext(streamContext);
      ModuleCallingContext moduleCallingContext(&modDesc, ModuleCallingContext::State::kRunning, parentContext);
      Run run(*ep, ModuleDescription(), &moduleCallingContext);
      a->postStreamEndRunSignal_(*streamContext, run, *es);
    }
    static void prePathSignal(ActivityRegistry *a, std::string const& s, PathContext const* pathContext) {
    }
    static void postPathSignal(ActivityRegistry *a, std::string const& s, HLTPathStatus const& status, PathContext const* pathContext) {
    }
    static void preModuleSignal(ActivityRegistry *a, ModuleDescription const* md, StreamContext const* streamContext, ModuleCallingContext const*  moduleCallingContext) {
      a->preModuleStreamEndRunSignal_(*streamContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry *a, ModuleDescription const* md, StreamContext const* streamContext, ModuleCallingContext const*  moduleCallingContext) {
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

    static void preScheduleSignal(ActivityRegistry *a, RunPrincipal const* ep, GlobalContext const* globalContext) {
      a->preEndRunSignal_(ep->id(), ep->endTime());
      a->preGlobalEndRunSignal_(*globalContext);
    }
    static void postScheduleSignal(ActivityRegistry *a, RunPrincipal* ep, EventSetup const* es, GlobalContext const* globalContext) {
      edm::ModuleDescription modDesc("postScheduledSignal", "");
      ParentContext parentContext(globalContext);
      ModuleCallingContext moduleCallingContext(&modDesc, ModuleCallingContext::State::kRunning, parentContext);
      Run run(*ep, ModuleDescription(), &moduleCallingContext);
      a->postEndRunSignal_(run, *es);
      a->postGlobalEndRunSignal_(*globalContext, run, *es);
    }
    static void prePathSignal(ActivityRegistry *a, std::string const& s, PathContext const* pathContext) {
      a->prePathEndRunSignal_(s);
    }
    static void postPathSignal(ActivityRegistry *a, std::string const& s, HLTPathStatus const& status, PathContext const* pathContext) {
      a->postPathEndRunSignal_(s, status);
    }
    static void preModuleSignal(ActivityRegistry *a, ModuleDescription const* md, GlobalContext const* globalContext, ModuleCallingContext const*  moduleCallingContext) {
      a->preModuleEndRunSignal_(*md);
      a->preModuleGlobalEndRunSignal_(*globalContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry *a, ModuleDescription const* md, GlobalContext const* globalContext, ModuleCallingContext const*  moduleCallingContext) {
      a->postModuleEndRunSignal_(*md);
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

    static void preScheduleSignal(ActivityRegistry *a, LuminosityBlockPrincipal const* ep, GlobalContext const* globalContext) {
      a->preBeginLumiSignal_(ep->id(), ep->beginTime());
      a->preGlobalBeginLumiSignal_(*globalContext);
    }
    static void postScheduleSignal(ActivityRegistry *a, LuminosityBlockPrincipal* ep, EventSetup const* es, GlobalContext const* globalContext) {
      // Next 4 lines not needed when we go to threaded interface
      edm::ModuleDescription modDesc("postScheduledSignal", "");
      ParentContext parentContext(globalContext);
      ModuleCallingContext moduleCallingContext(&modDesc, ModuleCallingContext::State::kRunning, parentContext);
      LuminosityBlock lumi(*ep, ModuleDescription(), &moduleCallingContext);
      a->postBeginLumiSignal_(lumi, *es);
      a->postGlobalBeginLumiSignal_(*globalContext);
    }
    static void prePathSignal(ActivityRegistry *a, std::string const& s, PathContext const* pathContext) {
      a->prePathBeginLumiSignal_(s);
    }
    static void postPathSignal(ActivityRegistry *a, std::string const& s, HLTPathStatus const& status, PathContext const* pathContext) {
      a->postPathBeginLumiSignal_(s, status);
    }
    static void preModuleSignal(ActivityRegistry *a, ModuleDescription const* md, GlobalContext const* globalContext, ModuleCallingContext const*  moduleCallingContext) {
      a->preModuleBeginLumiSignal_(*md);
      a->preModuleGlobalBeginLumiSignal_(*globalContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry *a, ModuleDescription const* md, GlobalContext const* globalContext, ModuleCallingContext const*  moduleCallingContext) {
      a->postModuleBeginLumiSignal_(*md);
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

    static void preScheduleSignal(ActivityRegistry *a, LuminosityBlockPrincipal const* ep, StreamContext const* streamContext) {
      a->preStreamBeginLumiSignal_(*streamContext);
    }
    static void postScheduleSignal(ActivityRegistry *a, LuminosityBlockPrincipal* ep, EventSetup const* es, StreamContext const* streamContext) {
      a->postStreamBeginLumiSignal_(*streamContext);
    }
    static void prePathSignal(ActivityRegistry *a, std::string const& s, PathContext const* pathContext) {
    }
    static void postPathSignal(ActivityRegistry *a, std::string const& s, HLTPathStatus const& status, PathContext const* pathContext) {
    }
    static void preModuleSignal(ActivityRegistry *a, ModuleDescription const* md, StreamContext const* streamContext, ModuleCallingContext const*  moduleCallingContext) {
      a->preModuleStreamBeginLumiSignal_(*streamContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry *a, ModuleDescription const* md, StreamContext const* streamContext, ModuleCallingContext const*  moduleCallingContext) {
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

    static StreamContext const* context(StreamContext const* s, GlobalContext const* g) { return s; }

    static void setStreamContext(StreamContext& streamContext, MyPrincipal const& principal) {
      streamContext.setTransition(StreamContext::Transition::kEndLuminosityBlock);
      streamContext.setEventID(EventID(principal.run(), principal.luminosityBlock(), 0));
      streamContext.setRunIndex(principal.runPrincipal().index());
      streamContext.setLuminosityBlockIndex(principal.index());
      streamContext.setTimestamp(principal.endTime());
    }

    static void preScheduleSignal(ActivityRegistry *a, LuminosityBlockPrincipal const* ep, StreamContext const* streamContext) {
      a->preStreamEndLumiSignal_(*streamContext);
    }
    static void postScheduleSignal(ActivityRegistry *a, LuminosityBlockPrincipal* ep, EventSetup const* es, StreamContext const* streamContext) {
      edm::ModuleDescription modDesc("postScheduledSignal", "");
      ParentContext parentContext(streamContext);
      ModuleCallingContext moduleCallingContext(&modDesc, ModuleCallingContext::State::kRunning, parentContext);
      LuminosityBlock lumi(*ep, ModuleDescription(), &moduleCallingContext);
      a->postStreamEndLumiSignal_(*streamContext, lumi, *es);
    }
    static void prePathSignal(ActivityRegistry *a, std::string const& s, PathContext const* pathContext) {
    }
    static void postPathSignal(ActivityRegistry *a, std::string const& s, HLTPathStatus const& status, PathContext const* pathContext) {
    }
    static void preModuleSignal(ActivityRegistry *a, ModuleDescription const* md, StreamContext const* streamContext, ModuleCallingContext const*  moduleCallingContext) {
      a->preModuleStreamEndLumiSignal_(*streamContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry *a, ModuleDescription const* md, StreamContext const* streamContext, ModuleCallingContext const*  moduleCallingContext) {
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

    static void preScheduleSignal(ActivityRegistry *a, LuminosityBlockPrincipal const* ep, GlobalContext const* globalContext) {
      a->preEndLumiSignal_(ep->id(), ep->beginTime()); 
      a->preGlobalEndLumiSignal_(*globalContext); 
    }
    static void postScheduleSignal(ActivityRegistry *a, LuminosityBlockPrincipal* ep, EventSetup const* es, GlobalContext const* globalContext) {
      edm::ModuleDescription modDesc("postScheduledSignal", "");
      ParentContext parentContext(globalContext);
      ModuleCallingContext moduleCallingContext(&modDesc, ModuleCallingContext::State::kRunning, parentContext);
      LuminosityBlock lumi(*ep, ModuleDescription(), &moduleCallingContext);
      a->postEndLumiSignal_(lumi, *es);
      a->postGlobalEndLumiSignal_(*globalContext, lumi, *es); 
    }
    static void prePathSignal(ActivityRegistry *a, std::string const& s, PathContext const* pathContext) {
      a->prePathEndLumiSignal_(s); 
    }
    static void postPathSignal(ActivityRegistry *a, std::string const& s, HLTPathStatus const& status, PathContext const* pathContext) {
      a->postPathEndLumiSignal_(s, status); 
    }
    static void preModuleSignal(ActivityRegistry *a, ModuleDescription const* md, GlobalContext const* globalContext, ModuleCallingContext const*  moduleCallingContext) {
      a->preModuleEndLumiSignal_(*md); 
      a->preModuleGlobalEndLumiSignal_(*globalContext, *moduleCallingContext);
    }
    static void postModuleSignal(ActivityRegistry *a, ModuleDescription const* md, GlobalContext const* globalContext, ModuleCallingContext const*  moduleCallingContext) {
      a->postModuleEndLumiSignal_(*md); 
      a->postModuleGlobalEndLumiSignal_(*globalContext, *moduleCallingContext);
    }
  };
}
#endif

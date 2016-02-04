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

#include<string>

namespace edm {
  template <typename T, BranchActionType B> class OccurrenceTraits;

  template <>
  class OccurrenceTraits<EventPrincipal, BranchActionBegin> {
  public:
    typedef EventPrincipal MyPrincipal;
    static BranchType const branchType_ = InEvent;
    static bool const begin_ = true;
    static bool const isEvent_ = true;
    static void preScheduleSignal(ActivityRegistry *a, EventPrincipal const* ep) {
      a->preProcessEventSignal_(ep->id(), ep->time()); 
    }
    static void postScheduleSignal(ActivityRegistry *a, EventPrincipal* ep, EventSetup const* es) {
      Event ev(*ep, ModuleDescription());
      a->postProcessEventSignal_(ev, *es);
    }
    static void prePathSignal(ActivityRegistry *a, std::string const& s) {
      a->preProcessPathSignal_(s); 
    }
    static void postPathSignal(ActivityRegistry *a, std::string const& s, HLTPathStatus const& status) {
      a->postProcessPathSignal_(s, status); 
    }
    static void preModuleSignal(ActivityRegistry *a, ModuleDescription const* md) {
      a->preModuleSignal_(*md); 
    }
    static void postModuleSignal(ActivityRegistry *a, ModuleDescription const* md) {
      a->postModuleSignal_(*md); 
    }
  };

  template <>
  class OccurrenceTraits<RunPrincipal, BranchActionBegin> {
  public:
    typedef RunPrincipal MyPrincipal;
    static BranchType const branchType_ = InRun;
    static bool const begin_ = true;
    static bool const isEvent_ = false;
    static void preScheduleSignal(ActivityRegistry *a, RunPrincipal const* ep) {
      a->preBeginRunSignal_(ep->id(), ep->beginTime()); 
    }
    static void postScheduleSignal(ActivityRegistry *a, RunPrincipal* ep, EventSetup const* es) {
      Run run(*ep, ModuleDescription());
      a->postBeginRunSignal_(run, *es);
    }
    static void prePathSignal(ActivityRegistry *a, std::string const& s) {
      a->prePathBeginRunSignal_(s); 
    }
    static void postPathSignal(ActivityRegistry *a, std::string const& s, HLTPathStatus const& status) {
      a->postPathBeginRunSignal_(s, status); 
    }
    static void preModuleSignal(ActivityRegistry *a, ModuleDescription const* md) {
      a->preModuleBeginRunSignal_(*md); 
    }
    static void postModuleSignal(ActivityRegistry *a, ModuleDescription const* md) {
      a->postModuleBeginRunSignal_(*md); 
    }
  };

  template <>
  class OccurrenceTraits<RunPrincipal, BranchActionEnd> {
  public:
    typedef RunPrincipal MyPrincipal;
    static BranchType const branchType_ = InRun;
    static bool const begin_ = false;
    static bool const isEvent_ = false;
    static void preScheduleSignal(ActivityRegistry *a, RunPrincipal const* ep) {
      a->preEndRunSignal_(ep->id(), ep->endTime()); 
    }
    static void postScheduleSignal(ActivityRegistry *a, RunPrincipal* ep, EventSetup const* es) {
      Run run(*ep, ModuleDescription());
      a->postEndRunSignal_(run, *es);
    }
    static void prePathSignal(ActivityRegistry *a, std::string const& s) {
      a->prePathEndRunSignal_(s); 
    }
    static void postPathSignal(ActivityRegistry *a, std::string const& s, HLTPathStatus const& status) {
      a->postPathEndRunSignal_(s, status); 
    }
    static void preModuleSignal(ActivityRegistry *a, ModuleDescription const* md) {
      a->preModuleEndRunSignal_(*md); 
    }
    static void postModuleSignal(ActivityRegistry *a, ModuleDescription const* md) {
      a->postModuleEndRunSignal_(*md); 
    }
  };

  template <>
  class OccurrenceTraits<LuminosityBlockPrincipal, BranchActionBegin> {
  public:
    typedef LuminosityBlockPrincipal MyPrincipal;
    static BranchType const branchType_ = InLumi;
    static bool const begin_ = true;
    static bool const isEvent_ = false;
    static void preScheduleSignal(ActivityRegistry *a, LuminosityBlockPrincipal const* ep) {
      a->preBeginLumiSignal_(ep->id(), ep->beginTime()); 
    }
    static void postScheduleSignal(ActivityRegistry *a, LuminosityBlockPrincipal* ep, EventSetup const* es) {
      LuminosityBlock lumi(*ep, ModuleDescription());
      a->postBeginLumiSignal_(lumi, *es);
    }
    static void prePathSignal(ActivityRegistry *a, std::string const& s) {
      a->prePathBeginLumiSignal_(s); 
    }
    static void postPathSignal(ActivityRegistry *a, std::string const& s, HLTPathStatus const& status) {
      a->postPathBeginLumiSignal_(s, status); 
    }
    static void preModuleSignal(ActivityRegistry *a, ModuleDescription const* md) {
      a->preModuleBeginLumiSignal_(*md); 
    }
    static void postModuleSignal(ActivityRegistry *a, ModuleDescription const* md) {
      a->postModuleBeginLumiSignal_(*md); 
    }
  };

  template <>
  class OccurrenceTraits<LuminosityBlockPrincipal, BranchActionEnd> {
  public:
    typedef LuminosityBlockPrincipal MyPrincipal;
    static BranchType const branchType_ = InLumi;
    static bool const begin_ = false;
    static bool const isEvent_ = false;
    static void preScheduleSignal(ActivityRegistry *a, LuminosityBlockPrincipal const* ep) {
      a->preEndLumiSignal_(ep->id(), ep->beginTime()); 
    }
    static void postScheduleSignal(ActivityRegistry *a, LuminosityBlockPrincipal* ep, EventSetup const* es) {
      LuminosityBlock lumi(*ep, ModuleDescription());
      a->postEndLumiSignal_(lumi, *es);
    }
    static void prePathSignal(ActivityRegistry *a, std::string const& s) {
      a->prePathEndLumiSignal_(s); 
    }
    static void postPathSignal(ActivityRegistry *a, std::string const& s, HLTPathStatus const& status) {
      a->postPathEndLumiSignal_(s, status); 
    }
    static void preModuleSignal(ActivityRegistry *a, ModuleDescription const* md) {
      a->preModuleEndLumiSignal_(*md); 
    }
    static void postModuleSignal(ActivityRegistry *a, ModuleDescription const* md) {
      a->postModuleEndLumiSignal_(*md); 
    }
  };
}
#endif

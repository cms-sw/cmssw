#ifndef EDM_FWD_H
#define EDM_FWD_H

/*----------------------------------------------------------------------
  
Forward declarations of types in the EDM.

$Id: CoreFrameworkfwd.h,v 1.7 2005/05/12 21:47:04 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm
{

  class BranchKey;
  class EventSetup;
  class EDAnalyzer;
  class EDFilter;
  class EDProducer;
  class EDProduct;
  class Event;
  class EventAux;
  class EventPrincipal;
  class EventProvenance;
  class EventRegistry;
  class InputService;
  class LuminositySection;
  class ModuleDescription;
  class ModuleDescriptionSelector;
  class OutputModule;
  class ParameterSet;
  class ProcessNameSelector;
  class Provenance;
  class PS_ID;
  class RefBase;
  class Retriever;
  class Run;
  class RunHandler;
  class Selector;
  class UntrackedParameterSet;

  template <class T> class EDCollection;
  template <class T> class Handle;
  template <class T> class Ref;
}

// The following are trivial enough so that the real headers can be included.
#include "FWCore/EDProduct/interface/CollisionID.h"
#include "FWCore/CoreFramework/interface/ConditionsID.h"
#include "FWCore/EDProduct/interface/EDP_ID.h"
#include "FWCore/CoreFramework/interface/PassID.h"
#include "FWCore/CoreFramework/interface/VersionNumber.h"

#endif

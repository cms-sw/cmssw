#ifndef Framework_Frameworkfwd_h
#define Framework_Frameworkfwd_h

/*----------------------------------------------------------------------
  
Forward declarations of types in the EDM.

$Id: Frameworkfwd.h,v 1.32 2006/12/18 17:40:50 paterno Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/EDProductfwd.h"

namespace edm {
  class BasicHandle;
  class BranchDescription;
  class BranchKey;
  class ConfigurableInputSource;
  class CurrentProcessingContext;
  class DelayedReader;
  class DataBlockImpl;
  class EDAnalyzer;
  class EDFilter;
  class EDLooper;
  class EDLooperHelper;
  class EDProducer;
  class Event;
  class EventPrincipal;
  class EventAux;
  class EventSetup;
  class ExternalInputSource;
  class GeneratedInputSource;
  class EDInputSource;
  class Group;
  class InputSource;
  class InputSourceDescription;
  class LuminosityBlock;
  class LuminosityBlockPrincipal;
  class ModuleDescription;
  class ModuleDescriptionSelector;
  class OutputModule;
  class ParameterSet;
  class ProcessNameSelector;
  class ProductRegistry;
  class ProductRegistryHelper;
  class Provenance;
  class RawInputSource;
  class Run;
  class RunPrincipal;
  class Schedule;
  class Selector;
  class SelectorBase;
  class TypeID;
  class UnsheduledHandler;
  class VectorInputSource;

  struct EventSummary;
  struct PathSummary;
  struct TriggerReport;

  template <typename T> class Handle;
  template <typename T> class View;
}

// The following are trivial enough so that the real headers can be included.
#include "DataFormats/Common/interface/ConditionsID.h"
#include "DataFormats/Common/interface/PassID.h"
#include "DataFormats/Common/interface/ReleaseVersion.h"
#include "FWCore/Framework/interface/BranchActionType.h"

#endif

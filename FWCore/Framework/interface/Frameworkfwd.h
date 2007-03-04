#ifndef Framework_Frameworkfwd_h
#define Framework_Frameworkfwd_h

/*----------------------------------------------------------------------
  
Forward declarations of types in the EDM.

$Id: Frameworkfwd.h,v 1.34 2007/01/10 05:58:01 wmtan Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/EDProductfwd.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"

namespace edm {
  class ConfigurableInputSource;
  class CurrentProcessingContext;
  class DelayedReader;
  class DataBlockImpl;
  class DataViewImpl;
  class EDAnalyzer;
  class EDFilter;
  class EDLooper;
  class EDLooperHelper;
  class EDProducer;
  class Event;
  class EventPrincipal;
  class EventSetup;
  class ExternalInputSource;
  class GeneratedInputSource;
  class EDInputSource;
  class Group;
  class InputSource;
  class InputSourceDescription;
  class LuminosityBlock;
  class LuminosityBlockPrincipal;
  class ModuleDescriptionSelector;
  class OutputModule;
  class ParameterSet;
  class ProcessNameSelector;
  class ProductRegistryHelper;
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
  template <typename T> class View;
}

// The following are trivial enough so that the real headers can be included.
#include "FWCore/Framework/interface/BranchActionType.h"

#endif

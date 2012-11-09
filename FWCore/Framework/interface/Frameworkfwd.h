#ifndef Framework_Frameworkfwd_h
#define Framework_Frameworkfwd_h

/*----------------------------------------------------------------------
  
Forward declarations of types in the EDM.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/EDProductfwd.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"

namespace edm {
  class CurrentProcessingContext;
  class PrincipalGetAdapter;
  class DelayedReader;
  class EDAnalyzer;
  class EDFilter;
  class EDLooper;
  class EDProducer;
  class Event;
  class EventPrincipal;
  class EventSetup;
  class FileBlock;
  class Group;
  class InputSource;
  struct InputSourceDescription;
  class LuminosityBlock;
  class LuminosityBlockPrincipal;
  class OutputModule;
  class OutputModuleDescription;
  class OutputWorker;
  class ParameterSet;
  class Principal;
  class PrincipalCache;
  class PrincipalGetAdapter;
  class ProcessNameSelector;
  class ProductRegistryHelper;
  class Run;
  class RunPrincipal;
  class Schedule;
  class TypeID;
  class UnscheduledHandler;
  class ViewBase;

  struct EventSummary;
  struct PathSummary;
  struct TriggerReport;
  template <typename T> class View;
  template <typename T> class WorkerT;
}

#endif

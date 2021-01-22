#ifndef Framework_Frameworkfwd_h
#define Framework_Frameworkfwd_h

/*----------------------------------------------------------------------
  
Forward declarations of types in the EDM.

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/EDProductfwd.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"

namespace edm {
  class PrincipalGetAdapter;
  class ConfigurationDescriptions;
  class ConsumesCollector;
  class DelayedReader;
  class EDAnalyzer;
  class EDFilter;
  class EDLooper;
  class EDProducer;
  class Event;
  class EventForOutput;
  class EventPrincipal;
  class EventSetup;
  class EventSetupImpl;
  class EventTransitionInfo;
  class FileBlock;
  class InputSource;
  struct InputSourceDescription;
  class LuminosityBlock;
  class LuminosityBlockForOutput;
  class LuminosityBlockPrincipal;
  class LumiTransitionInfo;
  class OutputModule;
  struct OutputModuleDescription;
  class ParameterSet;
  class ParameterSetDescription;
  class Principal;
  class PrincipalCache;
  class PrincipalGetAdapter;
  class ProcessBlock;
  class ProcessBlockPrincipal;
  class ProcessBlockTransitionInfo;
  class ProcessNameSelector;
  class ProductRegistryHelper;
  class Run;
  class RunForOutput;
  class RunPrincipal;
  class RunTransitionInfo;
  class Schedule;
  class StreamID;
  class TypeID;
  class ViewBase;

  struct EventSummary;
  struct PathSummary;
  struct TriggerReport;
  template <typename T>
  class View;
  template <typename T>
  class WorkerT;
}  // namespace edm

#endif

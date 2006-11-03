#ifndef Framework_Frameworkfwd_h
#define Framework_Frameworkfwd_h

/*----------------------------------------------------------------------
  
Forward declarations of types in the EDM.

$Id: Frameworkfwd.h,v 1.29 2006/10/31 23:54:01 wmtan Exp $

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
  class EDProducer;
  class Event;
  class EventAux;
  class EventSetup;
  class ExternalInputSource;
  class GeneratedInputSource;
  class EDInputSource;
  class Group;
  class InputSource;
  class InputSourceDescription;
  class LuminosityBlock;
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
}

// The following are trivial enough so that the real headers can be included.
#include "DataFormats/Common/interface/ConditionsID.h"
#include "DataFormats/Common/interface/PassID.h"
#include "DataFormats/Common/interface/ReleaseVersion.h"
#include "FWCore/Framework/interface/EventPrincipalFwd.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipalFwd.h"
#include "FWCore/Framework/interface/RunPrincipalFwd.h"

#endif

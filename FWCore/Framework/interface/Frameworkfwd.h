#ifndef Framework_Frameworkfwd_h
#define Framework_Frameworkfwd_h

/*----------------------------------------------------------------------
  
Forward declarations of types in the EDM.

$Id: Frameworkfwd.h,v 1.15 2006/01/10 22:49:30 wmtan Exp $

----------------------------------------------------------------------*/

#include "FWCore/EDProduct/interface/EDProductfwd.h"

namespace edm {
  class BasicHandle;
  class BranchDescription;
  class BranchKey;
  class ConfigurableInputSource;
  class DelayedReader;
  class EDAnalyzer;
  class EDFilter;
  class EDProducer;
  class Event;
  class EventAux;
  class EventPrincipal;
  class EventProvenance;
  class EventSetup;
  class ExternalInputSource;
  class GeneratedInputSource;
  class GenericInputSource;
  class Group;
  class InputSource;
  class InputSourceDescription;
  class LuminositySection;
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
  class RunHandler;
  class Selector;
  class TypeID;
  class VectorInputSource;

  template <typename T> class Handle;
}

// The following are trivial enough so that the real headers can be included.
#include "FWCore/Framework/interface/ConditionsID.h"
#include "FWCore/Framework/interface/PassID.h"
#include "FWCore/Framework/interface/VersionNumber.h"

#endif

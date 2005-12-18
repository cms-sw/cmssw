#ifndef Framework_Frameworkfwd_h
#define Framework_Frameworkfwd_h

/*----------------------------------------------------------------------
  
Forward declarations of types in the EDM.

$Id: Frameworkfwd.h,v 1.10 2005/10/03 19:04:04 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm
{
  class BasicHandle;
  class BranchKey;
  class DelayedReader;
  class EDAnalyzer;
  class EDFilter;
  class EDProducer;
  class EDProduct;
  class Event;
  class EventAux;
  class EventPrincipal;
  class EventProvenance;
  class EventRegistry;
  class EventSetup;
  class Group;
  class InputSource;
  class InputSourceDescription;
  class LuminositySection;
  class ModuleDescription;
  class ModuleDescriptionSelector;
  class OutputModule;
  class ParameterSet;
  class ProcessNameSelector;
  class BranchDescription;
  class ProductID;
  class ProductRegistry;
  class Provenance;
  class RandomAccessInputSource;
  class RefBase;
  class RefVectorBase;
  class Run;
  class RunHandler;
  class SecondaryInputSource;
  class Selector;

  template <class T> class Wrapper;
  template <class T> class Handle;
  template <class T> class Ref;
  template <class T> class RefVector;
  template <class T> class RefVectorIterator;
}

// The following are trivial enough so that the real headers can be included.
#include "FWCore/EDProduct/interface/EventID.h"
#include "FWCore/Framework/interface/ConditionsID.h"
#include "FWCore/Framework/interface/PassID.h"
#include "FWCore/Framework/interface/VersionNumber.h"

#endif

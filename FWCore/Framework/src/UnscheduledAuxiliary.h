#ifndef FWCore_Framework_UnscheduledAuxiliary_h
#define FWCore_Framework_UnscheduledAuxiliary_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     UnscheduledAuxiliary
//
/**\class UnscheduledAuxiliary UnscheduledAuxiliary.h "UnscheduledAuxiliary.h"

 Description: Holds auxiliary information needed for unscheduled calls to EDProducers

 Usage:
    Used internally by the framework

*/
//
// Original Author:  Chris Jones
//         Created:  Tue, 12 Apr 2016 20:49:46 GMT
//

// system include files

// user include files
#include "FWCore/Utilities/interface/Signal.h"

// forward declarations

namespace edm {
  class EventSetupImpl;
  class ModuleCallingContext;
  class StreamContext;

  class UnscheduledAuxiliary {
  public:
    UnscheduledAuxiliary() : m_eventSetup(nullptr) {}

    // ---------- const member functions ---------------------
    EventSetupImpl const* eventSetup() const { return m_eventSetup; }

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------
    void setEventSetup(EventSetupImpl const* iSetup) { m_eventSetup = iSetup; }

    signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> preModuleDelayedGetSignal_;
    signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> postModuleDelayedGetSignal_;

  private:
    // ---------- member data --------------------------------
    EventSetupImpl const* m_eventSetup;
  };
}  // namespace edm

#endif

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
#include "FWCore/Framework/src/TransitionInfoTypes.h"
#include "FWCore/Utilities/interface/Signal.h"

// forward declarations

namespace edm {
  class ModuleCallingContext;
  class StreamContext;

  class UnscheduledAuxiliary {
  public:
    // ---------- const member functions ---------------------
    EventTransitionInfo const& eventTransitionInfo() const { return m_eventTransitionInfo; }

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------
    void setEventTransitionInfo(EventTransitionInfo const& info) { m_eventTransitionInfo = info; }

    signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> preModuleDelayedGetSignal_;
    signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> postModuleDelayedGetSignal_;

  private:
    // ---------- member data --------------------------------
    EventTransitionInfo m_eventTransitionInfo;
  };
}  // namespace edm

#endif

#ifndef FWCore_Framework_EventAcquireSignalsSentry_h
#define FWCore_Framework_EventAcquireSignalsSentry_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     EventAcquireSignalsSentry
//
/**\class edm::EventAcquireSignalsSentry EventAcquireSignalsSentry.h "EventAcquireSignalsSentry.h"

 Description: Guarantees that the pre/post module EventAcquire signals are sent

 Usage:
    <usage>

*/
//
// Original Author:  W. David Dagenhart
//         Created:  Mon, 20 October 2017
//

// system include files
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"

// user include files

// forward declarations
namespace edm {
  class EventAcquireSignalsSentry {
  public:
    EventAcquireSignalsSentry(ActivityRegistry* iReg, ModuleCallingContext const* iContext)
        : m_reg(iReg), m_context(iContext) {
      iReg->preModuleEventAcquireSignal_(*(iContext->getStreamContext()), *iContext);
    }

    ~EventAcquireSignalsSentry() { m_reg->postModuleEventAcquireSignal_(*(m_context->getStreamContext()), *m_context); }

  private:
    // ---------- member data --------------------------------
    ActivityRegistry* m_reg;  // We do not use propagate_const because the registry itself is mutable.
    ModuleCallingContext const* m_context;
  };
}  // namespace edm

#endif

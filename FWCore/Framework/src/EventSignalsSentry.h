#ifndef FWCore_Framework_EventSignalsSentry_h
#define FWCore_Framework_EventSignalsSentry_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     EventSignalsSentry
// 
/**\class edm::EventSignalsSentry EventSignalsSentry.h "EventSignalsSentry.h"

 Description: Guarantees that the pre/post module Event signals are sent

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon, 12 May 2014 19:18:21 GMT
//

// system include files
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"

// user include files

// forward declarations
namespace edm {
  class EventSignalsSentry {

  public:
    EventSignalsSentry(ActivityRegistry* iReg,
                       ModuleCallingContext const * iContext) :
      m_reg(iReg),
      m_context(iContext)
    { iReg->preModuleEventSignal_( *(iContext->getStreamContext()), *iContext);}
    
    ~EventSignalsSentry() {
      m_reg->postModuleEventSignal_( *(m_context->getStreamContext()), *m_context);
    }
    
  private:
    // ---------- member data --------------------------------
    ActivityRegistry* m_reg;
    ModuleCallingContext const* m_context;
  };
}

#endif

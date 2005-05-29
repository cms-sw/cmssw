/*----------------------------------------------------------------------
$Id: InputService.cc,v 1.5 2005/05/20 16:55:46 paterno Exp $
----------------------------------------------------------------------*/
#include <cassert>

#include "FWCore/CoreFramework/interface/InputService.h"
#include "FWCore/CoreFramework/interface/EventPrincipal.h"
#include "FWCore/CoreFramework/interface/EventRegistry.h"

namespace edm
{
  InputService::InputService(const std::string& process) :
    process_(process)
  { 
    assert( !process.empty() );
  }

  InputService::~InputService() 
  { }

  std::auto_ptr<EventPrincipal>
  InputService::readEvent()
  {
    // Do we need any error handling (e.g. exception translation)
    // here?
    std::auto_ptr<EventPrincipal> ep( this->read() );
    if ( ep.get() ) 
      {
	EventRegistry::instance()->addEvent(ep->ID(), ep.get());
	ep->addToProcessHistory(process_);
      }
    return ep;
  }
}

/*----------------------------------------------------------------------
$Id: InputService.cc,v 1.2 2005/06/07 21:05:56 wmtan Exp $
----------------------------------------------------------------------*/
#include <cassert>

#include "FWCore/CoreFramework/interface/InputService.h"
#include "FWCore/CoreFramework/interface/EventPrincipal.h"

namespace edm
{
  InputService::InputService(const std::string& process) :
    process_(process)
  { 
    assert(!process.empty());
  }

  InputService::~InputService() 
  { }

  std::auto_ptr<EventPrincipal>
  InputService::readEvent()
  {
    // Do we need any error handling (e.g. exception translation)
    // here?
    std::auto_ptr<EventPrincipal> ep(this->read());
    if (ep.get()) 
      {
	ep->addToProcessHistory(process_);
      }
    return ep;
  }
}

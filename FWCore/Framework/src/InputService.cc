/*----------------------------------------------------------------------
$Id: InputService.cc,v 1.3 2005/06/23 19:59:48 wmtan Exp $
----------------------------------------------------------------------*/
#include <cassert>

#include "FWCore/Framework/interface/InputService.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

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

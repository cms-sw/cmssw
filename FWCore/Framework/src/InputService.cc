/*----------------------------------------------------------------------
$Id: InputService.cc,v 1.1 2005/05/29 02:29:54 wmtan Exp $
----------------------------------------------------------------------*/
#include <cassert>

#include "FWCore/CoreFramework/interface/InputService.h"
#include "FWCore/CoreFramework/interface/EventPrincipal.h"

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
	ep->addToProcessHistory(process_);
      }
    return ep;
  }
}

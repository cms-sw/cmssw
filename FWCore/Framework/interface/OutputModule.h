#ifndef EDM_OUTPUTMODULE_H
#define EDM_OUTPUTMODULE_H

/*----------------------------------------------------------------------
  
OutputModule: The base class of all "modules" that write Events to an
output stream.

$Id: OutputModule.h,v 1.3 2005/04/21 04:21:38 jbk Exp $

----------------------------------------------------------------------*/

#include "FWCore/CoreFramework/interface/EventPrincipal.h"

namespace edm
{
  class OutputModule
  {
  public:
    typedef OutputModule module_type;

    virtual ~OutputModule();
    virtual void write(const EventPrincipal& e) = 0;
  };
}

#endif

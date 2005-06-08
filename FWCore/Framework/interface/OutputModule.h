#ifndef EDM_OUTPUTMODULE_H
#define EDM_OUTPUTMODULE_H

/*----------------------------------------------------------------------
  
OutputModule: The base class of all "modules" that write Events to an
output stream.

$Id: OutputModule.h,v 1.1 2005/05/29 02:29:53 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm
{
  class EventPrincipal;
  class OutputModule
  {
  public:
    typedef OutputModule module_type;

    virtual ~OutputModule();
    virtual void write(const EventPrincipal& e) = 0;
  };
}

#endif

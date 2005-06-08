#ifndef EDM_OUTPUTMODULE_H
#define EDM_OUTPUTMODULE_H

/*----------------------------------------------------------------------
  
OutputModule: The base class of all "modules" that write Events to an
output stream.

$Id: OutputModule.h,v 1.2 2005/06/08 18:38:02 wmtan Exp $

----------------------------------------------------------------------*/

#include "FWCore/CoreFramework/interface/GroupSelector.h"

namespace edm
{
  class ParameterSet;
  class EventPrincipal;
  class OutputModule
  {
  public:
    typedef OutputModule module_type;

    explicit OutputModule(ParameterSet const& pset) : groupSelector_(pset) {}
    virtual ~OutputModule();
    virtual void write(const EventPrincipal& e) = 0;
    bool selected(std::string const& label) const {return groupSelector_.selected(label);}
  private:
    GroupSelector groupSelector_;
  };
}

#endif

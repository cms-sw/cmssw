#ifndef EDM_OUTPUTMODULE_H
#define EDM_OUTPUTMODULE_H

/*----------------------------------------------------------------------
  
OutputModule: The base class of all "modules" that write Events to an
output stream.

$Id: OutputModule.h,v 1.4 2005/06/23 05:23:10 wmtan Exp $

----------------------------------------------------------------------*/

#include "FWCore/CoreFramework/interface/GroupSelector.h"

namespace edm
{
  class ParameterSet;
  class EventPrincipal;
  class OutputModule
  {
  public:
    typedef OutputModule ModuleType;

    explicit OutputModule(ParameterSet const& pset) : groupSelector_(pset) {}
    virtual ~OutputModule();
    virtual void write(const EventPrincipal& e) = 0;
    bool selected(Provenance const& prov) const {return groupSelector_.selected(prov);}
  private:
    GroupSelector groupSelector_;
  };
}

#endif

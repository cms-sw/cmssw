#ifndef EDM_OUTPUTMODULE_H
#define EDM_OUTPUTMODULE_H

/*----------------------------------------------------------------------
  
OutputModule: The base class of all "modules" that write Events to an
output stream.

$Id: OutputModule.h,v 1.5 2005/06/24 23:32:13 wmtan Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/GroupSelector.h"

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

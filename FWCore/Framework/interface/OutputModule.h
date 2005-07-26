#ifndef EDM_OUTPUTMODULE_H
#define EDM_OUTPUTMODULE_H

/*----------------------------------------------------------------------
  
OutputModule: The base class of all "modules" that write Events to an
output stream.

$Id: OutputModule.h,v 1.7 2005/07/23 07:32:57 wmtan Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/GroupSelector.h"

namespace edm {
  class ParameterSet;
  class EventPrincipal;
  class ProductRegistry;
  class OutputModule {
  public:
    typedef OutputModule ModuleType;

    explicit OutputModule(ParameterSet const& pset) : groupSelector_(pset) {}
    virtual ~OutputModule();
    virtual void write(EventPrincipal const& e) = 0;
    bool selected(ProductDescription const& desc) const {return groupSelector_.selected(desc);}
    ProductRegistry const& productRegistry() const {return *preg_;}
    void setProductRegistry(ProductRegistry const* reg_) {preg_ = reg_;}
  private:
    GroupSelector groupSelector_;
    ProductRegistry const* preg_;
  };
}

#endif

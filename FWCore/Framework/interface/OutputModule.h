#ifndef EDM_OUTPUTMODULE_H
#define EDM_OUTPUTMODULE_H

/*----------------------------------------------------------------------
  
OutputModule: The base class of all "modules" that write Events to an
output stream.

$Id: OutputModule.h,v 1.11 2005/07/31 14:25:09 wmtan Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/GroupSelector.h"
#include "FWCore/Framework/interface/ProductDescription.h"
#include <vector>

namespace edm {
  class ParameterSet;
  class EventPrincipal;
  class EventSetup;
  class ProductDescription;
  class ProductRegistry;
  class OutputModule {
  public:
    typedef OutputModule ModuleType;
    typedef std::vector<ProductDescription const *> Selections;

    explicit OutputModule(ParameterSet const& pset, ProductRegistry const& reg);
    virtual ~OutputModule();
    virtual void beginJob(EventSetup const&);
    virtual void endJob();
    virtual void write(EventPrincipal const& e) = 0;
    bool selected(ProductDescription const& desc) const {return groupSelector_.selected(desc);}
  protected:
    ProductRegistry const* const preg_;
    Selections descVec_;
  private:
    GroupSelector groupSelector_;
  };
}

#endif

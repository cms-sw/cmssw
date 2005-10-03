#ifndef Framework_OutputModule_h
#define Framework_OutputModule_h

/*----------------------------------------------------------------------
  
OutputModule: The base class of all "modules" that write Events to an
output stream.

$Id: OutputModule.h,v 1.13 2005/09/01 05:12:04 wmtan Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/GroupSelector.h"
#include "FWCore/Framework/interface/BranchDescription.h"
#include <vector>

namespace edm {
  class ParameterSet;
  class EventPrincipal;
  class EventSetup;
  class BranchDescription;
  class ProductRegistry;
  class OutputModule {
  public:
    typedef OutputModule ModuleType;
    typedef std::vector<BranchDescription const *> Selections;

    explicit OutputModule(ParameterSet const& pset, ProductRegistry const& reg);
    virtual ~OutputModule();
    virtual void beginJob(EventSetup const&);
    virtual void endJob();
    virtual void write(EventPrincipal const& e) = 0;
    bool selected(BranchDescription const& desc) const {return groupSelector_.selected(desc);}
  protected:
    ProductRegistry const* const preg_;
    Selections descVec_;
  private:
    GroupSelector groupSelector_;
  };
}

#endif

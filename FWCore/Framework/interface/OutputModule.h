#ifndef Framework_OutputModule_h
#define Framework_OutputModule_h

/*----------------------------------------------------------------------
  
OutputModule: The base class of all "modules" that write Events to an
output stream.

$Id: OutputModule.h,v 1.15 2005/10/12 02:31:50 wmtan Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/GroupSelector.h"
#include "FWCore/Framework/interface/BranchDescription.h"
#include <vector>

namespace edm {
  class ParameterSet;
  class EventPrincipal;
  class EventSetup;
  class BranchDescription;
  class OutputModule {
  public:
    typedef OutputModule ModuleType;
    typedef std::vector<BranchDescription const *> Selections;

    explicit OutputModule(ParameterSet const& pset);
    virtual ~OutputModule();
    virtual void beginJob(EventSetup const&);
    virtual void endJob();
    virtual void write(EventPrincipal const& e) = 0;
    bool selected(BranchDescription const& desc) const {return groupSelector_.selected(desc);}

    unsigned long nextID() const;
  private:
    unsigned long nextID_;
  protected:
    Selections descVec_;
  private:
    GroupSelector groupSelector_;
  };
}

#endif

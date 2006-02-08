#ifndef Framework_OutputModule_h
#define Framework_OutputModule_h

/*----------------------------------------------------------------------
  
OutputModule: The base class of all "modules" that write Events to an
output stream.

$Id: OutputModule.h,v 1.18 2006/02/03 06:00:15 jbk Exp $

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/GroupSelector.h"
#include "FWCore/Framework/interface/EventSelector.h"
#include "DataFormats/Common/interface/BranchDescription.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "DataFormats/Common/interface/Provenance.h"
#include "FWCore/Framework/interface/Selector.h"

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
    void writeEvent(EventPrincipal const& e, ModuleDescription const&);
    bool selected(BranchDescription const& desc) const;

    unsigned long nextID() const;
  private:
    unsigned long nextID_;

    // TODO: Make this data member private, and give OutputModule an
    // interface (protected?) that supplies client code with the
    // needed functionality *without* giving away implementation
    // details ... don't just return a reference to descVec_, because
    // we are looking to have the flexibility to change the
    // implementation of descVec_ without modifying clients. When this
    // change is made, we'll have a one-time-only task of modifying
    // clients (classes derived from OutputModule) to use the
    // newly-introduced interface.
    // TODO: Consider using shared pointers here?

    // These are pointers to the BranchDescription objects describing
    // the branches we are to write. We do not own the
    // BranchDescriptions to which we point.
  protected:
    Selections descVec_;

  private:
    class ResultsSelector : public edm::Selector
    {
    public:
      explicit ResultsSelector(const std::string& proc_name):
	name_(proc_name) {}
      
      virtual bool doMatch(const edm::Provenance& p) const {
	return p.product.module.processName_==name_;
      }
    private:
      std::string name_;
    };

    virtual void write(EventPrincipal const& e) = 0;
    bool wantEvent(EventPrincipal const& e, ModuleDescription const&);

    std::string process_name_;
    GroupSelector groupSelector_;
    EventSelector eventSelector_;
    ResultsSelector selectResult_;
  };
}

#endif

#ifndef Framework_OutputModule_h
#define Framework_OutputModule_h

/*----------------------------------------------------------------------
  
OutputModule: The base class of all "modules" that write Events to an
output stream.

$Id: OutputModule.h,v 1.28 2006/10/31 23:54:01 wmtan Exp $

----------------------------------------------------------------------*/

#include <vector>

#include "DataFormats/Common/interface/BranchDescription.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "DataFormats/Common/interface/Provenance.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSelector.h"
#include "FWCore/Framework/interface/GroupSelector.h"
#include "FWCore/Framework/interface/Selector.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/Handle.h"

namespace edm {
   
  typedef edm::Handle<edm::TriggerResults> Trig;
  std::vector<std::string> const& getAllTriggerNames();

  class OutputModule {
  public:
    typedef OutputModule ModuleType;
    typedef std::vector<BranchDescription const *> Selections;

    explicit OutputModule(ParameterSet const& pset);
    virtual ~OutputModule();
    void doBeginJob(EventSetup const&);
    void doEndJob();
    void writeEvent(EventPrincipal const& e, ModuleDescription const& d,
		    CurrentProcessingContext const* c);
    void doBeginRun(RunPrincipal const& e, ModuleDescription const& d,
		    CurrentProcessingContext const* c);
    void doEndRun(RunPrincipal const& e, ModuleDescription const& d,
		    CurrentProcessingContext const* c);
    void doBeginLuminosityBlock(LuminosityBlockPrincipal const& e, ModuleDescription const& d,
		    CurrentProcessingContext const* c);
    void doEndLuminosityBlock(LuminosityBlockPrincipal const& e, ModuleDescription const& d,
		    CurrentProcessingContext const* c);
    bool selected(BranchDescription const& desc) const;

    unsigned long nextID() const;
    void selectProducts();

    const Trig& getTrigMask(EventPrincipal const& ep) const;

  protected:
    // The returned pointer will be null unless the this is currently
    // executing its event loop function ('write').
    CurrentProcessingContext const* currentContext() const;

  private:
    unsigned long             nextID_;
    // TODO: Make these data members private, and give OutputModule
    // an interface (protected?) that supplies client code with the
    // needed functionality *without* giving away implementation
    // details ... don't just return a reference to descVec_, because
    // we are looking to have the flexibility to change the
    // implementation of descVec_ without modifying clients. When this
    // change is made, we'll have a one-time-only task of modifying
    // clients (classes derived from OutputModule) to use the
    // newly-introduced interface.
    // ditto for droppedVec_.
    // TODO: Consider using shared pointers here?

    // descVec_ are pointers to the BranchDescription objects describing
    // the branches we are to write.
    // droppedVec_ are pointers to the BranchDescription objects describing
    // the branches we are NOT to write.
    // 
    // We do not own the BranchDescriptions to which we point.
  protected:
    Selections descVec_;
    Selections droppedVec_;

  private:
//     class ResultsSelector : public edm::Selector
//     {
//     public:
//       explicit ResultsSelector(const std::string& proc_name):
// 	name_(proc_name) {}
      
//       virtual bool doMatch(const edm::ProvenanceAccess& p) const {
// 	return p.provenance().processName()==name_;
//       }
//     private:
//       std::string name_;
//     };

    virtual void write(EventPrincipal const& e) = 0;
    bool wantEvent(EventPrincipal const& e);
    virtual void beginJob(EventSetup const&){}
    virtual void endJob(){}
    virtual void beginRun(RunPrincipal const& r){}
    virtual void endRun(RunPrincipal const& r){}
    //virtual void endRun(RunPrincipal const& r) = 0; // QQQ
    virtual void beginLuminosityBlock(LuminosityBlockPrincipal const& lb){}
    //virtual void endLuminosityBlock(LuminosityBlockPrincipal const& lb) = 0; // QQQ
    virtual void endLuminosityBlock(LuminosityBlockPrincipal const& lb){}

    std::string process_name_;
    GroupSelector groupSelector_;
    EventSelector eventSelector_;
    //ResultsSelector selectResult_;
    ProcessNameSelector selectResult_;

    // We do not own the pointed-to CurrentProcessingContext.
    CurrentProcessingContext const* current_context_;

    //This var will store Trigger Bit mask
    mutable Trig prod_;

    //Store the current Module Desc
    //  *** This should be superfluous, because current_context_->moduleDescription()
    // returns a pointer to the current ModuleDescription.
    ModuleDescription const* current_md_;  

  };
}

#endif

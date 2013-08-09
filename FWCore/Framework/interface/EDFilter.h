#ifndef FWCore_Framework_EDFilter_h
#define FWCore_Framework_EDFilter_h

/*----------------------------------------------------------------------
  
EDFilter: The base class of all "modules" used to control the flow of
processing in a processing path.
Filters can also insert products into the event.
These products should be informational products about the filter decision.


----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include <string>
#include <vector>

namespace edm {

  class ModuleCallingContext;

  class EDFilter : public ProducerBase, public EDConsumerBase {
  public:
    template <typename T> friend class WorkerT;
    typedef EDFilter ModuleType;
    typedef WorkerT<EDFilter> WorkerType;
    
     EDFilter() : ProducerBase() , moduleDescription_(), current_context_(nullptr), 
     previousParentage_(), previousParentageId_() {
    }
    virtual ~EDFilter();

    static void fillDescriptions(ConfigurationDescriptions& descriptions);
    static void prevalidate(ConfigurationDescriptions& );

    static const std::string& baseType();

    // Warning: the returned moduleDescription will be invalid during construction
    ModuleDescription const& moduleDescription() const { return moduleDescription_; }

  protected:
    // The returned pointer will be null unless the this is currently
    // executing its event loop function ('filter').
    CurrentProcessingContext const* currentContext() const;

  private:    
    bool doEvent(EventPrincipal& ep, EventSetup const& c,
                 CurrentProcessingContext const* cpc,
                 ModuleCallingContext const* mcc);
    void doBeginJob();
    void doEndJob();    
    void doBeginRun(RunPrincipal& rp, EventSetup const& c,
                    CurrentProcessingContext const* cpc,
                    ModuleCallingContext const* mcc);
    void doEndRun(RunPrincipal& rp, EventSetup const& c,
                  CurrentProcessingContext const* cpc,
                  ModuleCallingContext const* mcc);
    void doBeginLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                                CurrentProcessingContext const* cpc,
                                ModuleCallingContext const* mcc);
    void doEndLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                              CurrentProcessingContext const* cpc,
                              ModuleCallingContext const* mcc);
    void doRespondToOpenInputFile(FileBlock const& fb);
    void doRespondToCloseInputFile(FileBlock const& fb);
    void doPreForkReleaseResources();
    void doPostForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren);

    void registerProductsAndCallbacks(EDFilter* module, ProductRegistry* reg) {
      registerProducts(module, reg, moduleDescription_);
    }

    std::string workerType() const {return "WorkerT<EDFilter>";}

    virtual bool filter(Event&, EventSetup const&) = 0;
    virtual void beginJob(){}
    virtual void endJob(){}

    virtual void beginRun(Run const& iR, EventSetup const& iE){ }
    virtual void endRun(Run const& iR, EventSetup const& iE){}
    virtual void beginLuminosityBlock(LuminosityBlock const& iL, EventSetup const& iE){}
    virtual void endLuminosityBlock(LuminosityBlock const& iL, EventSetup const& iE){}
    virtual void respondToOpenInputFile(FileBlock const&) {}
    virtual void respondToCloseInputFile(FileBlock const&) {}
    virtual void preForkReleaseResources() {}
    virtual void postForkReacquireResources(unsigned int /*iChildIndex*/, unsigned int /*iNumberOfChildren*/) {}
     
    void setModuleDescription(ModuleDescription const& md) {
      moduleDescription_ = md;
    }
    ModuleDescription moduleDescription_;
    CurrentProcessingContext const* current_context_;
    std::vector<BranchID> previousParentage_;
    ParentageID previousParentageId_;
  };
}

#endif

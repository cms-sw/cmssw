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
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include <string>
#include <vector>
#include <array>

namespace edm {
  namespace maker {
    template <typename T>
    class ModuleHolderT;
  }

  class ModuleCallingContext;
  class PreallocationConfiguration;
  class ActivityRegistry;
  class ProductRegistry;
  class ThinnedAssociationsHelper;
  class WaitingTask;

  class EDFilter : public ProducerBase, public EDConsumerBase {
  public:
    template <typename T>
    friend class maker::ModuleHolderT;
    template <typename T>
    friend class WorkerT;
    typedef EDFilter ModuleType;

    EDFilter();
    ~EDFilter() override;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);
    static void prevalidate(ConfigurationDescriptions&);

    static const std::string& baseType();

    // Warning: the returned moduleDescription will be invalid during construction
    ModuleDescription const& moduleDescription() const { return moduleDescription_; }

    static bool wantsGlobalRuns() { return true; }
    static bool wantsGlobalLuminosityBlocks() { return true; }
    static bool wantsStreamRuns() { return false; }
    static bool wantsStreamLuminosityBlocks() { return false; };

    SerialTaskQueue* globalRunsQueue() { return &runQueue_; }
    SerialTaskQueue* globalLuminosityBlocksQueue() { return &luminosityBlockQueue_; }

  private:
    bool doEvent(EventPrincipal const& ep,
                 EventSetupImpl const& c,
                 ActivityRegistry* act,
                 ModuleCallingContext const* mcc);
    //Needed by WorkerT but not supported
    void preActionBeforeRunEventAsync(WaitingTask* iTask,
                                      ModuleCallingContext const& iModuleCallingContext,
                                      Principal const& iPrincipal) const {}

    void doPreallocate(PreallocationConfiguration const&) {}
    void doBeginJob();
    void doEndJob();
    void doBeginRun(RunPrincipal const& rp, EventSetupImpl const& c, ModuleCallingContext const* mcc);
    void doEndRun(RunPrincipal const& rp, EventSetupImpl const& c, ModuleCallingContext const* mcc);
    void doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                EventSetupImpl const& c,
                                ModuleCallingContext const* mcc);
    void doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                              EventSetupImpl const& c,
                              ModuleCallingContext const* mcc);
    void doRespondToOpenInputFile(FileBlock const& fb);
    void doRespondToCloseInputFile(FileBlock const& fb);
    void doRegisterThinnedAssociations(ProductRegistry const&, ThinnedAssociationsHelper&) {}

    void registerProductsAndCallbacks(EDFilter* module, ProductRegistry* reg) {
      registerProducts(module, reg, moduleDescription_);
    }

    std::string workerType() const { return "WorkerT<EDFilter>"; }

    SharedResourcesAcquirer& sharedResourcesAcquirer() { return resourceAcquirer_; }

    virtual bool filter(Event&, EventSetup const&) = 0;
    virtual void beginJob() {}
    virtual void endJob() {}

    virtual void beginRun(Run const&, EventSetup const&) {}
    virtual void endRun(Run const&, EventSetup const&) {}
    virtual void beginLuminosityBlock(LuminosityBlock const&, EventSetup const&) {}
    virtual void endLuminosityBlock(LuminosityBlock const&, EventSetup const&) {}
    virtual void respondToOpenInputFile(FileBlock const&) {}
    virtual void respondToCloseInputFile(FileBlock const&) {}

    bool hasAcquire() const { return false; }
    bool hasAccumulator() const { return false; }

    void setModuleDescription(ModuleDescription const& md) { moduleDescription_ = md; }
    ModuleDescription moduleDescription_;
    std::vector<BranchID> previousParentage_;
    SharedResourcesAcquirer resourceAcquirer_;
    SerialTaskQueue runQueue_;
    SerialTaskQueue luminosityBlockQueue_;
    ParentageID previousParentageId_;
  };
}  // namespace edm

#endif

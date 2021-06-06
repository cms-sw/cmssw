#ifndef FWCore_Framework_EDProducer_h
#define FWCore_Framework_EDProducer_h

/*----------------------------------------------------------------------

EDProducer: The base class of "modules" whose main purpose is to insert new
EDProducts into an Event.


----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"

#include <string>
#include <vector>

namespace edm {

  class ModuleCallingContext;
  class PreallocationConfiguration;
  class ActivityRegistry;
  class ThinnedAssociationsHelper;

  namespace maker {
    template <typename T>
    class ModuleHolderT;
  }

  class EDProducer : public ProducerBase, public EDConsumerBase {
  public:
    template <typename T>
    friend class maker::ModuleHolderT;
    template <typename T>
    friend class WorkerT;
    typedef EDProducer ModuleType;

    EDProducer();
    ~EDProducer() override;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);
    static void prevalidate(ConfigurationDescriptions& descriptions);
    static const std::string& baseType();

    // Warning: the returned moduleDescription will be invalid during construction
    ModuleDescription const& moduleDescription() const { return moduleDescription_; }

    static bool wantsProcessBlocks() { return false; }
    static bool wantsInputProcessBlocks() { return false; }
    static bool wantsGlobalRuns() { return true; }
    static bool wantsGlobalLuminosityBlocks() { return true; }
    static bool wantsStreamRuns() { return false; }
    static bool wantsStreamLuminosityBlocks() { return false; };

    SerialTaskQueue* globalRunsQueue() { return &runQueue_; }
    SerialTaskQueue* globalLuminosityBlocksQueue() { return &luminosityBlockQueue_; }

  private:
    bool doEvent(EventTransitionInfo const&, ActivityRegistry*, ModuleCallingContext const*);
    //Needed by WorkerT but not supported
    void preActionBeforeRunEventAsync(WaitingTaskHolder, ModuleCallingContext const&, Principal const&) const {}

    void doPreallocate(PreallocationConfiguration const&) {}
    void doBeginJob();
    void doEndJob();
    void doBeginProcessBlock(ProcessBlockPrincipal const&, ModuleCallingContext const*) {}
    void doAccessInputProcessBlock(ProcessBlockPrincipal const&, ModuleCallingContext const*) {}
    void doEndProcessBlock(ProcessBlockPrincipal const&, ModuleCallingContext const*) {}
    void doBeginRun(RunTransitionInfo const&, ModuleCallingContext const*);
    void doEndRun(RunTransitionInfo const&, ModuleCallingContext const*);
    void doBeginLuminosityBlock(LumiTransitionInfo const&, ModuleCallingContext const*);
    void doEndLuminosityBlock(LumiTransitionInfo const&, ModuleCallingContext const*);
    void doRespondToOpenInputFile(FileBlock const& fb);
    void doRespondToCloseInputFile(FileBlock const& fb);
    void doRegisterThinnedAssociations(ProductRegistry const&, ThinnedAssociationsHelper&) {}
    void registerProductsAndCallbacks(EDProducer* module, ProductRegistry* reg) {
      registerProducts(module, reg, moduleDescription_);
    }

    std::string workerType() const { return "WorkerT<EDProducer>"; }

    SharedResourcesAcquirer& sharedResourcesAcquirer() { return resourceAcquirer_; }

    virtual void produce(Event&, EventSetup const&) = 0;
    virtual void beginJob() {}
    virtual void endJob() {}

    virtual void beginRun(Run const& /* iR */, EventSetup const& /* iE */) {}
    virtual void endRun(Run const& /* iR */, EventSetup const& /* iE */) {}
    virtual void beginLuminosityBlock(LuminosityBlock const& /* iL */, EventSetup const& /* iE */) {}
    virtual void endLuminosityBlock(LuminosityBlock const& /* iL */, EventSetup const& /* iE */) {}
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

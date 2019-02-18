#ifndef FWCore_Framework_EDAnalyzer_h
#define FWCore_Framework_EDAnalyzer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "FWCore/Concurrency/interface/SerialTaskQueue.h"

#include <string>

// EDAnalyzer is the base class for all analyzer "modules".

namespace edm {

  class ModuleCallingContext;
  class PreallocationConfiguration;
  class ActivityRegistry;
  class ProductRegistry;
  class ThinnedAssociationsHelper;
  class WaitingTask;

  namespace maker {
    template<typename T> class ModuleHolderT;
  }

  class EDAnalyzer : public EDConsumerBase {
  public:
    template <typename T> friend class maker::ModuleHolderT;
    template <typename T> friend class WorkerT;
    typedef EDAnalyzer ModuleType;

    EDAnalyzer();
    ~EDAnalyzer() override;
    
    std::string workerType() const {return "WorkerT<EDAnalyzer>";}

    static void fillDescriptions(ConfigurationDescriptions& descriptions);
    static const std::string& baseType();
    static   void prevalidate(ConfigurationDescriptions& );

    // Warning: the returned moduleDescription will be invalid during construction
    ModuleDescription const& moduleDescription() const { return moduleDescription_; }
    
    static bool wantsGlobalRuns() {return true;}
    static bool wantsGlobalLuminosityBlocks() {return true;}
    static bool wantsStreamRuns() {return false;}
    static bool wantsStreamLuminosityBlocks() {return false;};

    void callWhenNewProductsRegistered(std::function<void(BranchDescription const&)> const& func);

    SerialTaskQueue* globalRunsQueue() { return &runQueue_;}
    SerialTaskQueue* globalLuminosityBlocksQueue() { return &luminosityBlockQueue_;}
  private:
    bool doEvent(EventPrincipal const& ep, EventSetupImpl const&  c,
                 ActivityRegistry* act,
                 ModuleCallingContext const* mcc);
    //Needed by Worker but not something supported
    void preActionBeforeRunEventAsync(WaitingTask* iTask, ModuleCallingContext const& iModuleCallingContext, Principal const& iPrincipal) const {}

    void doPreallocate(PreallocationConfiguration const&) {}
    void doBeginJob();
    void doEndJob();
    bool doBeginRun(RunPrincipal const& rp, EventSetupImpl const&  c,
                    ModuleCallingContext const* mcc);
    bool doEndRun(RunPrincipal const& rp, EventSetupImpl const&  c,
                  ModuleCallingContext const* mcc);
    bool doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetupImpl const&  c,
                                ModuleCallingContext const* mcc);
    bool doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetupImpl const&  c,
                              ModuleCallingContext const* mcc);
    void doRespondToOpenInputFile(FileBlock const& fb);
    void doRespondToCloseInputFile(FileBlock const& fb);
    void doRegisterThinnedAssociations(ProductRegistry const&,
                                       ThinnedAssociationsHelper&) { }

    void registerProductsAndCallbacks(EDAnalyzer const*, ProductRegistry* reg);
    
    SharedResourcesAcquirer& sharedResourcesAcquirer() {
      return resourceAcquirer_;
    }

    virtual void analyze(Event const&, EventSetup const&) = 0;
    virtual void beginJob(){}
    virtual void endJob(){}
    virtual void beginRun(Run const&, EventSetup const&){}
    virtual void endRun(Run const&, EventSetup const&){}
    virtual void beginLuminosityBlock(LuminosityBlock const&, EventSetup const&){}
    virtual void endLuminosityBlock(LuminosityBlock const&, EventSetup const&){}
    virtual void respondToOpenInputFile(FileBlock const&) {}
    virtual void respondToCloseInputFile(FileBlock const&) {}

    bool hasAcquire() const { return false; }
    bool hasAccumulator() const { return false; }

    void setModuleDescription(ModuleDescription const& md) {
      moduleDescription_ = md;
    }
    ModuleDescription moduleDescription_;
    SharedResourcesAcquirer resourceAcquirer_;

    SerialTaskQueue runQueue_;
    SerialTaskQueue luminosityBlockQueue_;
    
    std::function<void(BranchDescription const&)> callWhenNewProductsRegistered_;
  };
}

#endif

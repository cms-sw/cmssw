#ifndef FWCore_Framework_WorkerT_h
#define FWCore_Framework_WorkerT_h

/*----------------------------------------------------------------------

WorkerT: Code common to all workers.

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/WorkerParams.h"
#include "FWCore/ServiceRegistry/interface/ConsumesInfo.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace edm {

  class ModuleCallingContext;
  class ModuleDescription;
  class ProductResolverIndexAndSkipBit;
  class ProductRegistry;
  class ThinnedAssociationsHelper;

  template<typename T>
  class WorkerT : public Worker {
  public:
    typedef T ModuleType;
    typedef WorkerT<T> WorkerType;
    WorkerT(std::shared_ptr<T>,
            ModuleDescription const&,
            ExceptionToActionTable const* actions);

    virtual ~WorkerT();

    void setModule( std::shared_ptr<T> iModule) {
      module_ = iModule;
      resetModuleDescription(&(module_->moduleDescription()));
    }
    
    virtual Types moduleType() const override;

    virtual void updateLookup(BranchType iBranchType,
                              ProductResolverIndexHelper const&) override;


    template<typename D>
    void callWorkerBeginStream(D, StreamID);
    template<typename D>
    void callWorkerEndStream(D, StreamID);
    template<typename D>
    void callWorkerStreamBegin(D, StreamID id, RunPrincipal const& rp,
                               EventSetup const& c,
                               ModuleCallingContext const* mcc);
    template<typename D>
    void callWorkerStreamEnd(D, StreamID id, RunPrincipal const& rp,
                             EventSetup const& c,
                             ModuleCallingContext const* mcc);
    template<typename D>
    void callWorkerStreamBegin(D, StreamID id, LuminosityBlockPrincipal const& rp,
                               EventSetup const& c,
                               ModuleCallingContext const* mcc);
    template<typename D>
    void callWorkerStreamEnd(D, StreamID id, LuminosityBlockPrincipal const& rp,
                             EventSetup const& c,
                             ModuleCallingContext const* mcc);
    
  protected:
    T& module() {return *module_;}
    T const& module() const {return *module_;}

  private:
    virtual bool implDo(EventPrincipal const& ep, EventSetup const& c,
                        ModuleCallingContext const* mcc) override;
    virtual bool implDoPrePrefetchSelection(StreamID id,
                                            EventPrincipal const& ep,
                                            ModuleCallingContext const* mcc) override;
    virtual bool implDoBegin(RunPrincipal const& rp, EventSetup const& c,
                             ModuleCallingContext const* mcc) override;
    virtual bool implDoStreamBegin(StreamID id, RunPrincipal const& rp, EventSetup const& c,
                                   ModuleCallingContext const* mcc) override;
    virtual bool implDoStreamEnd(StreamID id, RunPrincipal const& rp, EventSetup const& c,
                                 ModuleCallingContext const* mcc) override;
    virtual bool implDoEnd(RunPrincipal const& rp, EventSetup const& c,
                           ModuleCallingContext const* mcc) override;
    virtual bool implDoBegin(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                             ModuleCallingContext const* mcc) override;
    virtual bool implDoStreamBegin(StreamID id, LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                                   ModuleCallingContext const* mcc) override;
    virtual bool implDoStreamEnd(StreamID id, LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                                 ModuleCallingContext const* mcc) override;
    virtual bool implDoEnd(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                           ModuleCallingContext const* mcc) override;
    virtual void implBeginJob() override;
    virtual void implEndJob() override;
    virtual void implBeginStream(StreamID) override;
    virtual void implEndStream(StreamID) override;
    virtual void implRespondToOpenInputFile(FileBlock const& fb) override;
    virtual void implRespondToCloseInputFile(FileBlock const& fb) override;
    virtual void implPreForkReleaseResources() override;
    virtual void implPostForkReacquireResources(unsigned int iChildIndex, 
                                               unsigned int iNumberOfChildren) override;
    virtual void implRegisterThinnedAssociations(ProductRegistry const&, ThinnedAssociationsHelper&) override;
    virtual std::string workerType() const override;

    virtual void modulesDependentUpon(std::vector<const char*>& oModuleLabels, bool iPrint) const override {
      module_->modulesDependentUpon(module_->moduleDescription().processName(),module_->moduleDescription().moduleLabel(), iPrint, oModuleLabels);
    }

    virtual void modulesWhoseProductsAreConsumed(std::vector<ModuleDescription const*>& modules,
                                                 ProductRegistry const& preg,
                                                 std::map<std::string, ModuleDescription const*> const& labelsToDesc) const override {
      module_->modulesWhoseProductsAreConsumed(modules, preg, labelsToDesc, module_->moduleDescription().processName());
    }

    virtual std::vector<ConsumesInfo> consumesInfo() const override {
      return module_->consumesInfo();
    }

    virtual void itemsToGet(BranchType branchType, std::vector<ProductResolverIndexAndSkipBit>& indexes) const override {
      module_->itemsToGet(branchType, indexes);
    }

    virtual void itemsMayGet(BranchType branchType, std::vector<ProductResolverIndexAndSkipBit>& indexes) const override {
      module_->itemsMayGet(branchType, indexes);
    }

    virtual std::vector<ProductResolverIndexAndSkipBit> const& itemsToGetFromEvent() const override { return module_->itemsToGetFromEvent(); }

    edm::propagate_const<std::shared_ptr<T>> module_;
  };

}

#endif

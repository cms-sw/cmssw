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

    ~WorkerT() override;

    void setModule( std::shared_ptr<T> iModule) {
      module_ = iModule;
      resetModuleDescription(&(module_->moduleDescription()));
    }
    
    Types moduleType() const override;

    void updateLookup(BranchType iBranchType,
                              ProductResolverIndexHelper const&) override;
    void resolvePutIndicies(BranchType iBranchType,
                                    std::unordered_multimap<std::string, edm::ProductResolverIndex> const& iIndicies) override;

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
    bool implDo(EventPrincipal const& ep, EventSetup const& c,
                        ModuleCallingContext const* mcc) override;
    bool implDoPrePrefetchSelection(StreamID id,
                                            EventPrincipal const& ep,
                                            ModuleCallingContext const* mcc) override;
    bool implDoBegin(RunPrincipal const& rp, EventSetup const& c,
                             ModuleCallingContext const* mcc) override;
    bool implDoStreamBegin(StreamID id, RunPrincipal const& rp, EventSetup const& c,
                                   ModuleCallingContext const* mcc) override;
    bool implDoStreamEnd(StreamID id, RunPrincipal const& rp, EventSetup const& c,
                                 ModuleCallingContext const* mcc) override;
    bool implDoEnd(RunPrincipal const& rp, EventSetup const& c,
                           ModuleCallingContext const* mcc) override;
    bool implDoBegin(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                             ModuleCallingContext const* mcc) override;
    bool implDoStreamBegin(StreamID id, LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                                   ModuleCallingContext const* mcc) override;
    bool implDoStreamEnd(StreamID id, LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                                 ModuleCallingContext const* mcc) override;
    bool implDoEnd(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                           ModuleCallingContext const* mcc) override;
    void implBeginJob() override;
    void implEndJob() override;
    void implBeginStream(StreamID) override;
    void implEndStream(StreamID) override;
    void implRespondToOpenInputFile(FileBlock const& fb) override;
    void implRespondToCloseInputFile(FileBlock const& fb) override;
    void implRegisterThinnedAssociations(ProductRegistry const&, ThinnedAssociationsHelper&) override;
    std::string workerType() const override;
    TaskQueueAdaptor serializeRunModule() override;


    void modulesWhoseProductsAreConsumed(std::vector<ModuleDescription const*>& modules,
                                                 ProductRegistry const& preg,
                                                 std::map<std::string, ModuleDescription const*> const& labelsToDesc) const override {
      module_->modulesWhoseProductsAreConsumed(modules, preg, labelsToDesc, module_->moduleDescription().processName());
    }

    void convertCurrentProcessAlias(std::string const& processName) override {
      module_->convertCurrentProcessAlias(processName);
    }

    std::vector<ConsumesInfo> consumesInfo() const override {
      return module_->consumesInfo();
    }

    void itemsToGet(BranchType branchType, std::vector<ProductResolverIndexAndSkipBit>& indexes) const override {
      module_->itemsToGet(branchType, indexes);
    }

    void itemsMayGet(BranchType branchType, std::vector<ProductResolverIndexAndSkipBit>& indexes) const override {
      module_->itemsMayGet(branchType, indexes);
    }

    std::vector<ProductResolverIndexAndSkipBit> const& itemsToGetFrom(BranchType iType) const final { return module_->itemsToGetFrom(iType); }
    
    std::vector<ProductResolverIndex> const& itemsShouldPutInEvent() const override;

    void preActionBeforeRunEventAsync(WaitingTask* iTask, ModuleCallingContext const& iModuleCallingContext, Principal const& iPrincipal) const override {
      module_->preActionBeforeRunEventAsync(iTask,iModuleCallingContext,iPrincipal);
    }

    
    edm::propagate_const<std::shared_ptr<T>> module_;
  };

}

#endif

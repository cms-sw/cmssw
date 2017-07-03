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

#include <string>
#include <vector>

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

  class EDProducer : public ProducerBase, public EDConsumerBase {
  public:
    template <typename T> friend class maker::ModuleHolderT;
    template <typename T> friend class WorkerT;
    typedef EDProducer ModuleType;

    EDProducer ();
    ~EDProducer() override;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);
    static void prevalidate(ConfigurationDescriptions& descriptions);
    static const std::string& baseType();

    // Warning: the returned moduleDescription will be invalid during construction
    ModuleDescription const& moduleDescription() const { return moduleDescription_; }

  private:
    bool doEvent(EventPrincipal const& ep, EventSetup const& c,
                 ActivityRegistry* act,
                 ModuleCallingContext const* mcc);
    //Needed by WorkerT but not supported
    void preActionBeforeRunEventAsync(WaitingTask* iTask, ModuleCallingContext const& iModuleCallingContext, Principal const& iPrincipal) const {}

    void doPreallocate(PreallocationConfiguration const&) {}
    void doBeginJob();
    void doEndJob();
    void doBeginRun(RunPrincipal const& rp, EventSetup const& c,
                    ModuleCallingContext const* mcc);
    void doEndRun(RunPrincipal const& rp, EventSetup const& c,
                  ModuleCallingContext const* mcc);
    void doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                                ModuleCallingContext const* mcc);
    void doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                              ModuleCallingContext const* mcc);
    void doRespondToOpenInputFile(FileBlock const& fb);
    void doRespondToCloseInputFile(FileBlock const& fb);
    void doRegisterThinnedAssociations(ProductRegistry const&,
                                       ThinnedAssociationsHelper&) { }
    void registerProductsAndCallbacks(EDProducer* module, ProductRegistry* reg) {
      registerProducts(module, reg, moduleDescription_);
    }

    std::string workerType() const {return "WorkerT<EDProducer>";}
    
    SharedResourcesAcquirer& sharedResourcesAcquirer() {
      return resourceAcquirer_;
    }

    virtual void produce(Event&, EventSetup const&) = 0;
    virtual void beginJob() {}
    virtual void endJob(){}

    virtual void beginRun(Run const& /* iR */, EventSetup const& /* iE */){}
    virtual void endRun(Run const& /* iR */, EventSetup const& /* iE */){}
    virtual void beginLuminosityBlock(LuminosityBlock const& /* iL */, EventSetup const& /* iE */){}
    virtual void endLuminosityBlock(LuminosityBlock const& /* iL */, EventSetup const& /* iE */){}
    virtual void respondToOpenInputFile(FileBlock const&) {}
    virtual void respondToCloseInputFile(FileBlock const&) {}

    void setModuleDescription(ModuleDescription const& md) {
      moduleDescription_ = md;
    }
    ModuleDescription moduleDescription_;
    std::vector<BranchID> previousParentage_;
    SharedResourcesAcquirer resourceAcquirer_;
    ParentageID previousParentageId_;
  };
}

#endif

#ifndef FWCore_Framework_one_EDFilterBase_h
#define FWCore_Framework_one_EDFilterBase_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     one::EDFilterBase
// 
/**\class one::EDFilterBase EDFilterBase.h "FWCore/Framework/interface/one/EDFilterBase.h"

 Description: Base class for edm::one::EDFilter<>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 02 May 2013 21:21:21 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

// forward declarations
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

  namespace one {

    class EDFilterBase : public ProducerBase, public EDConsumerBase
    {
      
    public:
      template <typename T> friend class edm::maker::ModuleHolderT;
      template <typename T> friend class edm::WorkerT;
      typedef EDFilterBase ModuleType;

      
      EDFilterBase();
      ~EDFilterBase() override;
      
      static void fillDescriptions(ConfigurationDescriptions& descriptions);
      static void prevalidate(ConfigurationDescriptions& descriptions);
      static const std::string& baseType();

      // Warning: the returned moduleDescription will be invalid during construction
      ModuleDescription const& moduleDescription() const { return moduleDescription_; }

      virtual bool wantsGlobalRuns() const =0;
      virtual bool wantsGlobalLuminosityBlocks() const =0;
      bool wantsStreamRuns() const {return false;}
      bool wantsStreamLuminosityBlocks() const {return false;};

      virtual SerialTaskQueue* globalRunsQueue();
      virtual SerialTaskQueue* globalLuminosityBlocksQueue();

    private:
      bool doEvent(EventPrincipal const& ep, EventSetup const& c,
                   ActivityRegistry*,
                   ModuleCallingContext const*);
      //For now this is a placeholder
      /*virtual*/ void preActionBeforeRunEventAsync(WaitingTask* iTask, ModuleCallingContext const& iModuleCallingContext, Principal const& iPrincipal) const {}

      void doPreallocate(PreallocationConfiguration const&);
      virtual void preallocLumis(unsigned int);
      void doBeginJob();
      void doEndJob();
      
      void doBeginRun(RunPrincipal const& rp, EventSetup const& c,
                      ModuleCallingContext const*);
      void doEndRun(RunPrincipal const& rp, EventSetup const& c,
                    ModuleCallingContext const*);
      void doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                                  ModuleCallingContext const*);
      void doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp, EventSetup const& c,
                                ModuleCallingContext const*);
      
      //For now, the following are just dummy implemenations with no ability for users to override
      void doRespondToOpenInputFile(FileBlock const& fb);
      void doRespondToCloseInputFile(FileBlock const& fb);
      void doRegisterThinnedAssociations(ProductRegistry const&,
                                         ThinnedAssociationsHelper&) { }

      void registerProductsAndCallbacks(EDFilterBase* module, ProductRegistry* reg) {
        registerProducts(module, reg, moduleDescription_);
      }
      std::string workerType() const {return "WorkerT<EDFilter>";}
      
      SharedResourcesAcquirer& sharedResourcesAcquirer() {
        return resourcesAcquirer_;
      }
      
      virtual bool filter(Event&, EventSetup const&) = 0;
      virtual void beginJob() {}
      virtual void endJob(){}

      virtual void preallocThreads(unsigned int) {}

      virtual void doBeginRun_(Run const& rp, EventSetup const& c);
      virtual void doEndRun_(Run const& rp, EventSetup const& c);
      virtual void doBeginLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c);
      virtual void doEndLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c);

      virtual void doBeginRunProduce_(Run& rp, EventSetup const& c);
      virtual void doEndRunProduce_(Run& rp, EventSetup const& c);
      virtual void doBeginLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c);
      virtual void doEndLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c);

      bool hasAcquire() const { return false; }
      bool hasAccumulator() const { return false; }

      virtual SharedResourcesAcquirer createAcquirer();

      void setModuleDescription(ModuleDescription const& md) {
        moduleDescription_ = md;
      }
      ModuleDescription moduleDescription_;
      std::vector<BranchID> previousParentage_;
      ParentageID previousParentageId_;

      SharedResourcesAcquirer resourcesAcquirer_;
    };
    
  }
}


#endif

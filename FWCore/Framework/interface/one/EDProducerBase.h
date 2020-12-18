#ifndef FWCore_Framework_one_EDProducerBase_h
#define FWCore_Framework_one_EDProducerBase_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     one::EDProducerBase
//
/**\class one::EDProducerBase EDProducerBase.h "FWCore/Framework/interface/one/EDProducerBase.h"

 Description: Base class for edm::one::EDProducer<>

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
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"

// forward declarations
namespace edm {

  class ModuleCallingContext;
  class PreallocationConfiguration;
  class ActivityRegistry;
  class ThinnedAssociationsHelper;

  namespace maker {
    template <typename T>
    class ModuleHolderT;
  }

  namespace one {

    class EDProducerBase : public ProducerBase, public EDConsumerBase {
    public:
      template <typename T>
      friend class edm::maker::ModuleHolderT;
      template <typename T>
      friend class edm::WorkerT;
      typedef EDProducerBase ModuleType;

      EDProducerBase();
      ~EDProducerBase() override;

      static void fillDescriptions(ConfigurationDescriptions& descriptions);
      static void prevalidate(ConfigurationDescriptions& descriptions);
      static const std::string& baseType();

      // Warning: the returned moduleDescription will be invalid during construction
      ModuleDescription const& moduleDescription() const { return moduleDescription_; }

      virtual bool wantsProcessBlocks() const = 0;
      virtual bool wantsInputProcessBlocks() const = 0;
      virtual bool wantsGlobalRuns() const = 0;
      virtual bool wantsGlobalLuminosityBlocks() const = 0;
      bool wantsStreamRuns() const { return false; }
      bool wantsStreamLuminosityBlocks() const { return false; };

      virtual SerialTaskQueue* globalRunsQueue();
      virtual SerialTaskQueue* globalLuminosityBlocksQueue();

    private:
      bool doEvent(EventTransitionInfo const&, ActivityRegistry*, ModuleCallingContext const*);
      //For now this is a placeholder
      /*virtual*/ void preActionBeforeRunEventAsync(WaitingTaskHolder,
                                                    ModuleCallingContext const&,
                                                    Principal const&) const {}

      void doPreallocate(PreallocationConfiguration const&);
      virtual void preallocLumis(unsigned int);
      void doBeginJob();
      void doEndJob();

      void doBeginProcessBlock(ProcessBlockPrincipal const&, ModuleCallingContext const*);
      void doAccessInputProcessBlock(ProcessBlockPrincipal const&, ModuleCallingContext const*);
      void doEndProcessBlock(ProcessBlockPrincipal const&, ModuleCallingContext const*);
      void doBeginRun(RunTransitionInfo const&, ModuleCallingContext const*);
      void doEndRun(RunTransitionInfo const&, ModuleCallingContext const*);
      void doBeginLuminosityBlock(LumiTransitionInfo const&, ModuleCallingContext const*);
      void doEndLuminosityBlock(LumiTransitionInfo const&, ModuleCallingContext const*);

      //For now, the following are just dummy implemenations with no ability for users to override
      void doRespondToOpenInputFile(FileBlock const& fb);
      void doRespondToCloseInputFile(FileBlock const& fb);
      void doRegisterThinnedAssociations(ProductRegistry const&, ThinnedAssociationsHelper&) {}

      void registerProductsAndCallbacks(EDProducerBase* module, ProductRegistry* reg) {
        registerProducts(module, reg, moduleDescription_);
      }
      std::string workerType() const { return "WorkerT<EDProducer>"; }

      SharedResourcesAcquirer& sharedResourcesAcquirer() { return resourcesAcquirer_; }

      virtual void produce(Event&, EventSetup const&) = 0;
      virtual void beginJob() {}
      virtual void endJob() {}

      virtual void preallocThreads(unsigned int) {}

      virtual void doBeginProcessBlock_(ProcessBlock const&);
      virtual void doAccessInputProcessBlock_(ProcessBlock const&);
      virtual void doEndProcessBlock_(ProcessBlock const&);
      virtual void doBeginRun_(Run const& rp, EventSetup const& c);
      virtual void doEndRun_(Run const& rp, EventSetup const& c);
      virtual void doBeginLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c);
      virtual void doEndLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c);

      virtual void doBeginProcessBlockProduce_(ProcessBlock&);
      virtual void doEndProcessBlockProduce_(ProcessBlock&);
      virtual void doBeginRunProduce_(Run& rp, EventSetup const& c);
      virtual void doEndRunProduce_(Run& rp, EventSetup const& c);
      virtual void doBeginLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c);
      virtual void doEndLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c);

      virtual bool hasAccumulator() const { return false; }

      bool hasAcquire() const { return false; }

      virtual SharedResourcesAcquirer createAcquirer();

      void setModuleDescription(ModuleDescription const& md) { moduleDescription_ = md; }
      ModuleDescription moduleDescription_;
      std::vector<BranchID> previousParentage_;
      ParentageID previousParentageId_;

      SharedResourcesAcquirer resourcesAcquirer_;
    };

  }  // namespace one
}  // namespace edm

#endif

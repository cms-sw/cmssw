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
  class EventForTransformer;
  class ServiceWeakToken;

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

      virtual bool wantsProcessBlocks() const noexcept = 0;
      virtual bool wantsInputProcessBlocks() const noexcept = 0;
      virtual bool wantsGlobalRuns() const noexcept = 0;
      virtual bool wantsGlobalLuminosityBlocks() const noexcept = 0;
      bool wantsStreamRuns() const noexcept { return false; }
      bool wantsStreamLuminosityBlocks() const noexcept { return false; };

      virtual SerialTaskQueue* globalRunsQueue();
      virtual SerialTaskQueue* globalLuminosityBlocksQueue();

    private:
      bool doEvent(EventTransitionInfo const&, ActivityRegistry*, ModuleCallingContext const*);
      //For now this is a placeholder
      /*virtual*/ void preActionBeforeRunEventAsync(WaitingTaskHolder,
                                                    ModuleCallingContext const&,
                                                    Principal const&) const noexcept {}

      void doTransformAsync(WaitingTaskHolder iTask,
                            size_t iTransformIndex,
                            EventPrincipal const& iEvent,
                            ActivityRegistry*,
                            ModuleCallingContext,
                            ServiceWeakToken const&) noexcept;
      void doPreallocate(PreallocationConfiguration const&);
      virtual void preallocRuns(unsigned int);
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

      void doRespondToOpenInputFile(FileBlock const&) {}
      void doRespondToCloseInputFile(FileBlock const&) {}
      void doRespondToCloseOutputFile() { clearInputProcessBlockCaches(); }
      void doRegisterThinnedAssociations(ProductRegistry const&, ThinnedAssociationsHelper&) {}

      void registerProductsAndCallbacks(EDProducerBase* module, SignallingProductRegistry* reg) {
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

      virtual size_t transformIndex_(edm::ProductDescription const& iBranch) const noexcept;
      virtual ProductResolverIndex transformPrefetch_(std::size_t iIndex) const noexcept;
      virtual void transformAsync_(WaitingTaskHolder iTask,
                                   std::size_t iIndex,
                                   edm::EventForTransformer& iEvent,
                                   edm::ActivityRegistry* iAct,
                                   ServiceWeakToken const& iToken) const noexcept;

      virtual void clearInputProcessBlockCaches();
      virtual bool hasAccumulator() const noexcept { return false; }

      bool hasAcquire() const noexcept { return false; }

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

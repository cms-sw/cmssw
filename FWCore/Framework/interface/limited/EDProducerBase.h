#ifndef FWCore_Framework_limited_EDProducerBase_h
#define FWCore_Framework_limited_EDProducerBase_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     EDProducerBase
//
/**\class EDProducerBase EDProducerBase.h "EDProducerBase.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 18 Jul 2013 11:51:14 GMT
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Concurrency/interface/LimitedTaskQueue.h"

// forward declarations

namespace edm {
  class ModuleCallingContext;
  class PreallocationConfiguration;
  class StreamID;
  class GlobalSchedule;
  class ActivityRegistry;
  class ProductRegistry;
  class ThinnedAssociationsHelper;
  class WaitingTask;

  namespace maker {
    template <typename T>
    class ModuleHolderT;
  }

  namespace limited {

    class EDProducerBase : public ProducerBase, public EDConsumerBase {
    public:
      template <typename T>
      friend class edm::maker::ModuleHolderT;
      template <typename T>
      friend class edm::WorkerT;
      typedef EDProducerBase ModuleType;

      friend class edm::GlobalSchedule;

      EDProducerBase(ParameterSet const& pset);
      ~EDProducerBase() override;

      static void fillDescriptions(ConfigurationDescriptions& descriptions);
      static void prevalidate(ConfigurationDescriptions& descriptions);
      static const std::string& baseType();

      // Warning: the returned moduleDescription will be invalid during construction
      ModuleDescription const& moduleDescription() const { return moduleDescription_; }

      virtual bool wantsGlobalRuns() const = 0;
      virtual bool wantsGlobalLuminosityBlocks() const = 0;
      virtual bool wantsStreamRuns() const = 0;
      virtual bool wantsStreamLuminosityBlocks() const = 0;

      unsigned int concurrencyLimit() const { return queue_.concurrencyLimit(); }

      LimitedTaskQueue& queue() { return queue_; }

    private:
      bool doEvent(EventPrincipal const& ep, EventSetupImpl const& c, ActivityRegistry*, ModuleCallingContext const*);
      void doPreallocate(PreallocationConfiguration const&);
      void doBeginJob();
      void doEndJob();

      void doBeginStream(StreamID id);
      void doEndStream(StreamID id);
      void doStreamBeginRun(StreamID id, RunPrincipal const& ep, EventSetupImpl const& c, ModuleCallingContext const*);
      void doStreamEndRun(StreamID id, RunPrincipal const& ep, EventSetupImpl const& c, ModuleCallingContext const*);
      void doStreamBeginLuminosityBlock(StreamID id,
                                        LuminosityBlockPrincipal const& ep,
                                        EventSetupImpl const& c,
                                        ModuleCallingContext const*);
      void doStreamEndLuminosityBlock(StreamID id,
                                      LuminosityBlockPrincipal const& ep,
                                      EventSetupImpl const& c,
                                      ModuleCallingContext const*);

      void doBeginRun(RunPrincipal const& rp, EventSetupImpl const& c, ModuleCallingContext const*);
      void doEndRun(RunPrincipal const& rp, EventSetupImpl const& c, ModuleCallingContext const*);
      void doBeginLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                  EventSetupImpl const& c,
                                  ModuleCallingContext const*);
      void doEndLuminosityBlock(LuminosityBlockPrincipal const& lbp,
                                EventSetupImpl const& c,
                                ModuleCallingContext const*);

      //For now, the following are just dummy implemenations with no ability for users to override
      void doRespondToOpenInputFile(FileBlock const& fb);
      void doRespondToCloseInputFile(FileBlock const& fb);
      void doRegisterThinnedAssociations(ProductRegistry const&, ThinnedAssociationsHelper&) {}

      void registerProductsAndCallbacks(EDProducerBase* module, ProductRegistry* reg) {
        registerProducts(module, reg, moduleDescription_);
      }
      std::string workerType() const { return "WorkerT<EDProducer>"; }

      virtual void produce(StreamID, Event&, EventSetup const&) const = 0;
      //For now this is a placeholder
      /*virtual*/ void preActionBeforeRunEventAsync(WaitingTask* iTask,
                                                    ModuleCallingContext const& iModuleCallingContext,
                                                    Principal const& iPrincipal) const {}

      virtual void beginJob() {}
      virtual void endJob() {}

      virtual void preallocStreams(unsigned int);
      virtual void preallocLumis(unsigned int);
      virtual void preallocLumisSummary(unsigned int);
      virtual void preallocate(PreallocationConfiguration const&);
      virtual void doBeginStream_(StreamID id);
      virtual void doEndStream_(StreamID id);
      virtual void doStreamBeginRun_(StreamID id, Run const& rp, EventSetup const& c);
      virtual void doStreamEndRun_(StreamID id, Run const& rp, EventSetup const& c);
      virtual void doStreamEndRunSummary_(StreamID id, Run const& rp, EventSetup const& c);
      virtual void doStreamBeginLuminosityBlock_(StreamID id, LuminosityBlock const& lbp, EventSetup const& c);
      virtual void doStreamEndLuminosityBlock_(StreamID id, LuminosityBlock const& lbp, EventSetup const& c);
      virtual void doStreamEndLuminosityBlockSummary_(StreamID id, LuminosityBlock const& lbp, EventSetup const& c);

      virtual void doBeginRun_(Run const& rp, EventSetup const& c);
      virtual void doBeginRunSummary_(Run const& rp, EventSetup const& c);
      virtual void doEndRunSummary_(Run const& rp, EventSetup const& c);
      virtual void doEndRun_(Run const& rp, EventSetup const& c);
      virtual void doBeginLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c);
      virtual void doBeginLuminosityBlockSummary_(LuminosityBlock const& rp, EventSetup const& c);
      virtual void doEndLuminosityBlockSummary_(LuminosityBlock const& lb, EventSetup const& c);
      virtual void doEndLuminosityBlock_(LuminosityBlock const& lb, EventSetup const& c);

      virtual void doBeginRunProduce_(Run& rp, EventSetup const& c);
      virtual void doEndRunProduce_(Run& rp, EventSetup const& c);
      virtual void doBeginLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c);
      virtual void doEndLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c);

      virtual bool hasAccumulator() const { return false; }

      bool hasAcquire() const { return false; }

      void setModuleDescription(ModuleDescription const& md) { moduleDescription_ = md; }
      ModuleDescription moduleDescription_;
      std::unique_ptr<std::vector<BranchID>[]> previousParentages_;
      std::unique_ptr<ParentageID[]> previousParentageIds_;
      LimitedTaskQueue queue_;
    };

  }  // namespace limited
}  // namespace edm

#endif

#ifndef FWCore_Framework_limited_EDAnalyzerBase_h
#define FWCore_Framework_limited_EDAnalyzerBase_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     EDAnalyzerBase
//
/**\class limited::EDAnalyzerBase EDAnalyzerBase.h "EDAnalyzerBase.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 18 Jul 2013 11:51:14 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Concurrency/interface/LimitedTaskQueue.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"

// forward declarations

namespace edm {
  class ModuleCallingContext;
  class PreallocationConfiguration;
  class StreamID;
  class ActivityRegistry;
  class ThinnedAssociationsHelper;

  namespace maker {
    template <typename T>
    class ModuleHolderT;
  }

  namespace limited {

    class EDAnalyzerBase : public EDConsumerBase {
    public:
      template <typename T>
      friend class edm::WorkerT;
      template <typename T>
      friend class edm::maker::ModuleHolderT;
      typedef EDAnalyzerBase ModuleType;

      EDAnalyzerBase(ParameterSet const& pset);
      ~EDAnalyzerBase() override;

      static void fillDescriptions(ConfigurationDescriptions& descriptions);
      static void prevalidate(ConfigurationDescriptions& descriptions);
      static const std::string& baseType();

      // Warning: the returned moduleDescription will be invalid during construction
      ModuleDescription const& moduleDescription() const { return moduleDescription_; }

      virtual bool wantsProcessBlocks() const = 0;
      virtual bool wantsInputProcessBlocks() const = 0;
      virtual bool wantsGlobalRuns() const = 0;
      virtual bool wantsGlobalLuminosityBlocks() const = 0;
      virtual bool wantsStreamRuns() const = 0;
      virtual bool wantsStreamLuminosityBlocks() const = 0;

      void callWhenNewProductsRegistered(std::function<void(BranchDescription const&)> const& func) {
        callWhenNewProductsRegistered_ = func;
      }

      unsigned int concurrencyLimit() const { return queue_.concurrencyLimit(); }

      LimitedTaskQueue& queue() { return queue_; }

    private:
      bool doEvent(EventTransitionInfo const&, ActivityRegistry*, ModuleCallingContext const*);
      //For now this is a placeholder
      /*virtual*/ void preActionBeforeRunEventAsync(WaitingTaskHolder iTask,
                                                    ModuleCallingContext const& iModuleCallingContext,
                                                    Principal const& iPrincipal) const {}

      void doPreallocate(PreallocationConfiguration const&);
      void doBeginJob();
      void doEndJob();

      void doBeginStream(StreamID id);
      void doEndStream(StreamID id);
      void doStreamBeginRun(StreamID, RunTransitionInfo const&, ModuleCallingContext const*);
      void doStreamEndRun(StreamID, RunTransitionInfo const&, ModuleCallingContext const*);
      void doStreamBeginLuminosityBlock(StreamID, LumiTransitionInfo const&, ModuleCallingContext const*);
      void doStreamEndLuminosityBlock(StreamID, LumiTransitionInfo const&, ModuleCallingContext const*);

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

      void registerProductsAndCallbacks(EDAnalyzerBase* module, ProductRegistry* reg);
      std::string workerType() const { return "WorkerT<EDAnalyzer>"; }

      virtual void analyze(StreamID, Event const&, EventSetup const&) const = 0;
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

      virtual void doBeginProcessBlock_(ProcessBlock const&);
      virtual void doAccessInputProcessBlock_(ProcessBlock const&);
      virtual void doEndProcessBlock_(ProcessBlock const&);
      virtual void doBeginRun_(Run const& rp, EventSetup const& c);
      virtual void doBeginRunSummary_(Run const& rp, EventSetup const& c);
      virtual void doEndRunSummary_(Run const& rp, EventSetup const& c);
      virtual void doEndRun_(Run const& rp, EventSetup const& c);
      virtual void doBeginLuminosityBlock_(LuminosityBlock const& lbp, EventSetup const& c);
      virtual void doBeginLuminosityBlockSummary_(LuminosityBlock const& rp, EventSetup const& c);
      virtual void doEndLuminosityBlockSummary_(LuminosityBlock const& lb, EventSetup const& c);
      virtual void doEndLuminosityBlock_(LuminosityBlock const& lb, EventSetup const& c);

      virtual void clearInputProcessBlockCaches();

      bool hasAcquire() const { return false; }
      bool hasAccumulator() const { return false; }

      void setModuleDescription(ModuleDescription const& md) { moduleDescription_ = md; }
      ModuleDescription moduleDescription_;

      std::function<void(BranchDescription const&)> callWhenNewProductsRegistered_;

      LimitedTaskQueue queue_;
    };

  }  // namespace limited
}  // namespace edm

#endif

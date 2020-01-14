/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/ModuleContextSentry.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/FunctorTask.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"
#include "FWCore/Utilities/interface/make_sentry.h"

#include "FWCore/Framework/src/OutputModuleCommunicatorT.h"

#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/interface/global/OutputModuleBase.h"
#include "FWCore/Framework/interface/one/OutputModuleBase.h"
#include "FWCore/Framework/interface/limited/OutputModuleBase.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

namespace {
  template <typename F>
  void async(edm::OutputModule& iMod, F&& iFunc) {
    iMod.sharedResourcesAcquirer().serialQueueChain().push(std::move(iFunc));
  }

  template <typename F>
  void async(edm::one::OutputModuleBase& iMod, F&& iFunc) {
    iMod.sharedResourcesAcquirer().serialQueueChain().push(std::move(iFunc));
  }

  template <typename F>
  void async(edm::limited::OutputModuleBase& iMod, F&& iFunc) {
    iMod.queue().push(std::move(iFunc));
  }

  template <typename F>
  void async(edm::global::OutputModuleBase&, F iFunc) {
    auto t = edm::make_functor_task(tbb::task::allocate_root(), iFunc);
    tbb::task::spawn(*t);
  }
}  // namespace

namespace edm {

  template <typename T>
  void OutputModuleCommunicatorT<T>::closeFile() {
    module().doCloseFile();
  }

  template <typename T>
  bool OutputModuleCommunicatorT<T>::shouldWeCloseFile() const {
    return module().shouldWeCloseFile();
  }

  template <typename T>
  void OutputModuleCommunicatorT<T>::openFile(edm::FileBlock const& fb) {
    module().doOpenFile(fb);
  }

  template <typename T>
  void OutputModuleCommunicatorT<T>::writeRunAsync(WaitingTaskHolder iTask,
                                                   edm::RunPrincipal const& rp,
                                                   ProcessContext const* processContext,
                                                   ActivityRegistry* activityRegistry,
                                                   MergeableRunProductMetadata const* mergeableRunProductMetadata) {
    auto token = ServiceRegistry::instance().presentToken();
    GlobalContext globalContext(GlobalContext::Transition::kWriteRun,
                                LuminosityBlockID(rp.run(), 0),
                                rp.index(),
                                LuminosityBlockIndex::invalidLuminosityBlockIndex(),
                                rp.endTime(),
                                processContext);
    auto t = [& mod = module(),
              &rp,
              globalContext,
              token,
              desc = &description(),
              activityRegistry,
              mergeableRunProductMetadata,
              iTask]() mutable {
      std::exception_ptr ex;
      // Caught exception is propagated via WaitingTaskHolder
      CMS_SA_ALLOW try {
        ServiceRegistry::Operate op(token);
        ParentContext parentContext(&globalContext);
        ModuleCallingContext mcc(desc);
        ModuleContextSentry moduleContextSentry(&mcc, parentContext);
        activityRegistry->preModuleWriteRunSignal_(globalContext, mcc);
        auto sentry(make_sentry(activityRegistry, [&globalContext, &mcc](ActivityRegistry* ar) {
          ar->postModuleWriteRunSignal_(globalContext, mcc);
        }));
        mod.doWriteRun(rp, &mcc, mergeableRunProductMetadata);
      } catch (...) {
        ex = std::current_exception();
      }
      iTask.doneWaiting(ex);
    };
    async(module(), std::move(t));
  }

  template <typename T>
  void OutputModuleCommunicatorT<T>::writeLumiAsync(WaitingTaskHolder iTask,
                                                    edm::LuminosityBlockPrincipal const& lbp,
                                                    ProcessContext const* processContext,
                                                    ActivityRegistry* activityRegistry) {
    auto token = ServiceRegistry::instance().presentToken();
    GlobalContext globalContext(GlobalContext::Transition::kWriteLuminosityBlock,
                                lbp.id(),
                                lbp.runPrincipal().index(),
                                lbp.index(),
                                lbp.beginTime(),
                                processContext);
    auto t = [& mod = module(), &lbp, activityRegistry, token, globalContext, desc = &description(), iTask]() mutable {
      std::exception_ptr ex;
      // Caught exception is propagated via WaitingTaskHolder
      CMS_SA_ALLOW try {
        ServiceRegistry::Operate op(token);

        ParentContext parentContext(&globalContext);
        ModuleCallingContext mcc(desc);
        ModuleContextSentry moduleContextSentry(&mcc, parentContext);
        activityRegistry->preModuleWriteLumiSignal_(globalContext, mcc);
        auto sentry(make_sentry(activityRegistry, [&globalContext, &mcc](ActivityRegistry* ar) {
          ar->postModuleWriteLumiSignal_(globalContext, mcc);
        }));
        mod.doWriteLuminosityBlock(lbp, &mcc);
      } catch (...) {
        ex = std::current_exception();
      }
      iTask.doneWaiting(ex);
    };
    async(module(), std::move(t));
  }

  template <typename T>
  bool OutputModuleCommunicatorT<T>::wantAllEvents() const {
    return module().wantAllEvents();
  }

  template <typename T>
  bool OutputModuleCommunicatorT<T>::limitReached() const {
    return module().limitReached();
  }

  template <typename T>
  void OutputModuleCommunicatorT<T>::configure(OutputModuleDescription const& desc) {
    module().configure(desc);
  }

  template <typename T>
  edm::SelectedProductsForBranchType const& OutputModuleCommunicatorT<T>::keptProducts() const {
    return module().keptProducts();
  }

  template <typename T>
  void OutputModuleCommunicatorT<T>::selectProducts(edm::ProductRegistry const& preg,
                                                    ThinnedAssociationsHelper const& helper) {
    module().selectProducts(preg, helper);
  }

  template <typename T>
  void OutputModuleCommunicatorT<T>::setEventSelectionInfo(
      std::map<std::string, std::vector<std::pair<std::string, int>>> const& outputModulePathPositions,
      bool anyProductProduced) {
    module().setEventSelectionInfo(outputModulePathPositions, anyProductProduced);
  }

  template <typename T>
  ModuleDescription const& OutputModuleCommunicatorT<T>::description() const {
    return module().description();
  }

  namespace impl {
    std::unique_ptr<edm::OutputModuleCommunicator> createCommunicatorIfNeeded(void*) {
      return std::unique_ptr<edm::OutputModuleCommunicator>{};
    }
    std::unique_ptr<edm::OutputModuleCommunicator> createCommunicatorIfNeeded(::edm::OutputModule* iMod) {
      return std::make_unique<OutputModuleCommunicatorT<edm::OutputModule>>(iMod);
    }
    std::unique_ptr<edm::OutputModuleCommunicator> createCommunicatorIfNeeded(::edm::global::OutputModuleBase* iMod) {
      return std::make_unique<OutputModuleCommunicatorT<edm::global::OutputModuleBase>>(iMod);
    }
    std::unique_ptr<edm::OutputModuleCommunicator> createCommunicatorIfNeeded(::edm::one::OutputModuleBase* iMod) {
      return std::make_unique<OutputModuleCommunicatorT<edm::one::OutputModuleBase>>(iMod);
    }
    std::unique_ptr<edm::OutputModuleCommunicator> createCommunicatorIfNeeded(::edm::limited::OutputModuleBase* iMod) {
      return std::make_unique<OutputModuleCommunicatorT<edm::limited::OutputModuleBase>>(iMod);
    }
  }  // namespace impl
}  // namespace edm

namespace edm {
  template class OutputModuleCommunicatorT<OutputModule>;
  template class OutputModuleCommunicatorT<one::OutputModuleBase>;
  template class OutputModuleCommunicatorT<global::OutputModuleBase>;
  template class OutputModuleCommunicatorT<limited::OutputModuleBase>;
}  // namespace edm

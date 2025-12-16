#include "FWCore/Framework/interface/ModuleRegistryUtilities.h"
#include "FWCore/Framework/interface/ModuleRegistry.h"
#include "FWCore/Framework/interface/maker/ModuleSignalSentry.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/make_sentry.h"
#include "FWCore/Utilities/interface/ExceptionCollector.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"

namespace edm {
  void finishModulesInitialization(ModuleRegistry& iModuleRegistry,
                                   ProductRegistry const& iProductRegistry,
                                   eventsetup::ESRecordsToProductResolverIndices const& iESIndices,
                                   ProcessBlockHelperBase const& processBlockHelperBase,
                                   std::string const& processName) {
    auto const processBlockLookup = iProductRegistry.productLookup(InProcess);
    auto const runLookup = iProductRegistry.productLookup(InRun);
    auto const lumiLookup = iProductRegistry.productLookup(InLumi);
    auto const eventLookup = iProductRegistry.productLookup(InEvent);

    auto processBlockModuleToIndicies = processBlockLookup->indiciesForModulesInProcess(processName);
    auto runModuleToIndicies = runLookup->indiciesForModulesInProcess(processName);
    auto lumiModuleToIndicies = lumiLookup->indiciesForModulesInProcess(processName);
    auto eventModuleToIndicies = eventLookup->indiciesForModulesInProcess(processName);

    iModuleRegistry.forAllModuleHolders([&](auto& iHolder) {
      iHolder->updateLookup(InProcess, *processBlockLookup);
      iHolder->updateLookup(InRun, *runLookup);
      iHolder->updateLookup(InLumi, *lumiLookup);
      iHolder->updateLookup(InEvent, *eventLookup);
      iHolder->updateLookup(iESIndices);
      iHolder->resolvePutIndicies(InProcess, processBlockModuleToIndicies);
      iHolder->resolvePutIndicies(InRun, runModuleToIndicies);
      iHolder->resolvePutIndicies(InLumi, lumiModuleToIndicies);
      iHolder->resolvePutIndicies(InEvent, eventModuleToIndicies);
      iHolder->selectInputProcessBlocks(iProductRegistry, processBlockHelperBase);
    });
  }

  namespace {
    template <typename F>
    void captureExceptionAndContinue(std::exception_ptr& iExcept, ModuleCallingContext const& iContext, F&& iF) {
      try {
        convertException::wrap(std::forward<F&&>(iF));
      } catch (cms::Exception& newException) {
        if (not iExcept) {
          edm::exceptionContext(newException, iContext);
          iExcept = std::current_exception();
        }
      }
    }

    template <typename F>
    void captureExceptionsAndContinue(ExceptionCollector& iCollector, ModuleCallingContext const& iContext, F&& iF) {
      try {
        convertException::wrap(std::forward<F&&>(iF));
      } catch (cms::Exception& ex) {
        edm::exceptionContext(ex, iContext);
        iCollector.addException(ex);
      }
    }
    template <typename F>
    void captureExceptionsAndContinue(ExceptionCollector& iCollector,
                                      std::mutex& iMutex,
                                      ModuleCallingContext const& iContext,
                                      F&& iF) {
      try {
        convertException::wrap(std::forward<F&&>(iF));
      } catch (cms::Exception& ex) {
        edm::exceptionContext(ex, iContext);
        std::lock_guard guard(iMutex);
        iCollector.addException(ex);
      }
    }

    class ModuleBeginJobTraits {
    public:
      using Context = GlobalContext;
      static void preModuleSignal(ActivityRegistry* activityRegistry,
                                  GlobalContext const*,
                                  ModuleCallingContext const* moduleCallingContext) {
        activityRegistry->preModuleBeginJobSignal_.emit(*moduleCallingContext->moduleDescription());
      }
      static void postModuleSignal(ActivityRegistry* activityRegistry,
                                   GlobalContext const*,
                                   ModuleCallingContext const* moduleCallingContext) {
        activityRegistry->postModuleBeginJobSignal_.emit(*moduleCallingContext->moduleDescription());
      }
    };

    class ModuleEndJobTraits {
    public:
      using Context = GlobalContext;
      static void preModuleSignal(ActivityRegistry* activityRegistry,
                                  GlobalContext const*,
                                  ModuleCallingContext const* moduleCallingContext) {
        activityRegistry->preModuleEndJobSignal_.emit(*moduleCallingContext->moduleDescription());
      }
      static void postModuleSignal(ActivityRegistry* activityRegistry,
                                   GlobalContext const*,
                                   ModuleCallingContext const* moduleCallingContext) {
        activityRegistry->postModuleEndJobSignal_.emit(*moduleCallingContext->moduleDescription());
      }
    };

    class ModuleBeginStreamTraits {
    public:
      using Context = StreamContext;
      static void preModuleSignal(ActivityRegistry* activityRegistry,
                                  StreamContext const* streamContext,
                                  ModuleCallingContext const* moduleCallingContext) {
        activityRegistry->preModuleBeginStreamSignal_.emit(*streamContext, *moduleCallingContext);
      }
      static void postModuleSignal(ActivityRegistry* activityRegistry,
                                   StreamContext const* streamContext,
                                   ModuleCallingContext const* moduleCallingContext) {
        activityRegistry->postModuleBeginStreamSignal_.emit(*streamContext, *moduleCallingContext);
      }
    };

    class ModuleEndStreamTraits {
    public:
      using Context = StreamContext;
      static void preModuleSignal(ActivityRegistry* activityRegistry,
                                  StreamContext const* streamContext,
                                  ModuleCallingContext const* moduleCallingContext) {
        activityRegistry->preModuleEndStreamSignal_.emit(*streamContext, *moduleCallingContext);
      }
      static void postModuleSignal(ActivityRegistry* activityRegistry,
                                   StreamContext const* streamContext,
                                   ModuleCallingContext const* moduleCallingContext) {
        activityRegistry->postModuleEndStreamSignal_.emit(*streamContext, *moduleCallingContext);
      }
    };

  }  // namespace
  void runBeginJobForModules(GlobalContext const& iGlobalContext,
                             ModuleRegistry& iModuleRegistry,
                             edm::ActivityRegistry& iActivityRegistry,
                             std::vector<unsigned int>& beginJobFailedForModule) noexcept(false) {
    beginJobFailedForModule.clear();

    ParentContext pc(&iGlobalContext);

    std::exception_ptr exceptionPtr;
    iModuleRegistry.forAllModuleHolders([&](auto& holder) {
      ModuleCallingContext mcc(&holder->moduleDescription());
      //Also sets a thread local
      ModuleContextSentry mccSentry(&mcc, pc);
      captureExceptionAndContinue(exceptionPtr, mcc, [&]() {
        ModuleSignalSentry<ModuleBeginJobTraits> signalSentry(&iActivityRegistry, &iGlobalContext, &mcc);
        auto failedSentry = make_sentry(&beginJobFailedForModule,
                                        [&](auto* failed) { failed->push_back(holder->moduleDescription().id()); });
        signalSentry.preModuleSignal();
        holder->beginJob();
        signalSentry.postModuleSignal();
        failedSentry.release();
      });
    });
    if (exceptionPtr) {
      std::rethrow_exception(exceptionPtr);
    }
  }

  void runEndJobForModules(GlobalContext const& iGlobalContext,
                           ModuleRegistry& iModuleRegistry,
                           ActivityRegistry& iActivityRegistry,
                           ExceptionCollector& collector,
                           std::vector<unsigned int> const& beginJobFailedForModule) noexcept {
    ParentContext pc(&iGlobalContext);
    const bool beginJobFailed = not beginJobFailedForModule.empty();
    iModuleRegistry.forAllModuleHolders([&](auto& holder) {
      if (beginJobFailed and
          std::count(
              beginJobFailedForModule.begin(), beginJobFailedForModule.end(), holder->moduleDescription().id())) {
        // If beginJob was never called, we should not call endJob.
        return;
      }
      ModuleCallingContext mcc(&holder->moduleDescription());
      //Also sets a thread local
      ModuleContextSentry mccSentry(&mcc, pc);
      captureExceptionsAndContinue(collector, mcc, [&]() {
        ModuleSignalSentry<ModuleEndJobTraits> signalSentry(&iActivityRegistry, &iGlobalContext, &mcc);
        signalSentry.preModuleSignal();
        holder->endJob();
        signalSentry.postModuleSignal();
      });
    });
  }

  void runBeginStreamForModules(StreamContext const& iStreamContext,
                                ModuleRegistry& iModuleRegistry,
                                edm::ActivityRegistry& iActivityRegistry,
                                std::vector<unsigned int>& beginStreamFailedForModule) noexcept(false) {
    beginStreamFailedForModule.clear();

    std::exception_ptr exceptionPtr;
    ParentContext pc(&iStreamContext);
    iModuleRegistry.forAllModuleHolders([&](auto& holder) {
      ModuleCallingContext mcc(&holder->moduleDescription());
      //Also sets a thread local
      ModuleContextSentry mccSentry(&mcc, pc);
      captureExceptionAndContinue(exceptionPtr, mcc, [&]() {
        auto failedSentry = make_sentry(&beginStreamFailedForModule,
                                        [&](auto* failed) { failed->push_back(holder->moduleDescription().id()); });
        ModuleSignalSentry<ModuleBeginStreamTraits> signalSentry(&iActivityRegistry, &iStreamContext, &mcc);
        signalSentry.preModuleSignal();
        holder->beginStream(iStreamContext.streamID());
        signalSentry.postModuleSignal();
        failedSentry.release();
      });
    });
    if (exceptionPtr) {
      std::rethrow_exception(exceptionPtr);
    }
  }

  void runEndStreamForModules(StreamContext const& iStreamContext,
                              ModuleRegistry& iModuleRegistry,
                              ActivityRegistry& iActivityRegistry,
                              ExceptionCollector& collector,
                              std::mutex& collectorMutex,
                              std::vector<unsigned int> const& beginStreamFailedForModule) noexcept {
    ParentContext pc(&iStreamContext);
    bool const beginStreamFailed = not beginStreamFailedForModule.empty();
    iModuleRegistry.forAllModuleHolders([&](auto& holder) {
      if (beginStreamFailed and
          std::count(
              beginStreamFailedForModule.begin(), beginStreamFailedForModule.end(), holder->moduleDescription().id())) {
        // If beginStream was never called, we should not call endStream.
        return;
      }
      ModuleCallingContext mcc(&holder->moduleDescription());
      //Also sets a thread local
      ModuleContextSentry mccSentry(&mcc, pc);
      captureExceptionsAndContinue(collector, collectorMutex, mcc, [&]() {
        ModuleSignalSentry<ModuleEndStreamTraits> signalSentry(&iActivityRegistry, &iStreamContext, &mcc);
        signalSentry.preModuleSignal();
        holder->endStream(iStreamContext.streamID());
        signalSentry.postModuleSignal();
      });
    });
  }

}  // namespace edm

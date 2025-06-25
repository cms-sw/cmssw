#include "FWCore/Framework/interface/ModuleRegistryUtilities.h"
#include "FWCore/Framework/interface/ModuleRegistry.h"
#include "FWCore/Common/interface/ProcessBlockHelperBase.h"
#include "FWCore/Framework/interface/ESRecordsToProductResolverIndices.h"
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
      try {
        convertException::wrap([&]() {
          // if exception happens, before or during beginJob, add to beginJobFailedForModule
          auto failedSentry = make_sentry(&beginJobFailedForModule,
                                          [&](auto* failed) { failed->push_back(holder->moduleDescription().id()); });
          auto sentry = make_sentry(&holder->moduleDescription(), [&](auto const* description) {
            iActivityRegistry.postModuleBeginJobSignal_(*description);
          });
          iActivityRegistry.preModuleBeginJobSignal_(holder->moduleDescription());
          holder->beginJob();
          failedSentry.release();
        });
      } catch (cms::Exception& ex) {
        if (!exceptionPtr) {
          edm::exceptionContext(ex, mcc);
          exceptionPtr = std::current_exception();
        }
      }
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
      try {
        convertException::wrap([&]() {
          auto sentry = make_sentry(&holder->moduleDescription(), [&](auto const* description) {
            iActivityRegistry.postModuleEndJobSignal_(*description);
          });
          iActivityRegistry.preModuleEndJobSignal_(holder->moduleDescription());
          holder->endJob();
        });
      } catch (cms::Exception& ex) {
        edm::exceptionContext(ex, mcc);
        collector.addException(ex);
      }
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
      try {
        convertException::wrap([&]() {
          auto sentry = make_sentry(&mcc, [&](auto const* context) {
            iActivityRegistry.postModuleBeginStreamSignal_(iStreamContext, *context);
          });
          // if exception happens, before or during beginStream, add to beginStreamFailedForModule
          auto failedSentry = make_sentry(&beginStreamFailedForModule,
                                          [&](auto* failed) { failed->push_back(holder->moduleDescription().id()); });
          iActivityRegistry.preModuleBeginStreamSignal_(iStreamContext, mcc);
          holder->beginStream(iStreamContext.streamID());
          failedSentry.release();
        });
      } catch (cms::Exception& ex) {
        if (!exceptionPtr) {
          edm::exceptionContext(ex, mcc);
          exceptionPtr = std::current_exception();
        }
      }
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
      try {
        convertException::wrap([&]() {
          auto sentry = make_sentry(
              &mcc, [&](auto const* mc) { iActivityRegistry.postModuleEndStreamSignal_(iStreamContext, *mc); });
          iActivityRegistry.preModuleEndStreamSignal_(iStreamContext, mcc);
          holder->endStream(iStreamContext.streamID());
        });
      } catch (cms::Exception& ex) {
        edm::exceptionContext(ex, mcc);
        std::lock_guard<std::mutex> collectorLock(collectorMutex);
        collector.addException(ex);
      }
    });
  }

}  // namespace edm
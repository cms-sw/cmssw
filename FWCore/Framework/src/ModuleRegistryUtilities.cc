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
                             std::vector<bool>& beginJobCalledForModule) noexcept(false) {
    beginJobCalledForModule.clear();
    beginJobCalledForModule.resize(iModuleRegistry.maxModuleID() + 1, false);

    ParentContext pc(&iGlobalContext);

    std::exception_ptr exceptionPtr;
    iModuleRegistry.forAllModuleHolders([&](auto& holder) {
      ModuleCallingContext mcc(&holder->moduleDescription());
      //Also sets a thread local
      ModuleContextSentry mccSentry(&mcc, pc);
      try {
        convertException::wrap([&]() {
          auto sentry = make_sentry(&holder->moduleDescription(), [&](auto const* description) {
            iActivityRegistry.postModuleBeginJobSignal_(*description);
          });
          iActivityRegistry.preModuleBeginJobSignal_(holder->moduleDescription());
          holder->beginJob();
          beginJobCalledForModule[holder->moduleDescription().id()] = true;
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
    beginJobCalledForModule.clear();
  }

  void runEndJobForModules(GlobalContext const& iGlobalContext,
                           ModuleRegistry& iModuleRegistry,
                           ActivityRegistry& iActivityRegistry,
                           ExceptionCollector& collector,
                           std::vector<bool> const& beginJobCalledForModule) noexcept {
    assert(beginJobCalledForModule.empty() or beginJobCalledForModule.size() == iModuleRegistry.maxModuleID() + 1);
    ParentContext pc(&iGlobalContext);
    iModuleRegistry.forAllModuleHolders([&](auto& holder) {
      ModuleCallingContext mcc(&holder->moduleDescription());
      //Also sets a thread local
      ModuleContextSentry mccSentry(&mcc, pc);
      try {
        convertException::wrap([&]() {
          if (not beginJobCalledForModule.empty() and !beginJobCalledForModule[holder->moduleDescription().id()]) {
            // If beginJob was never called, we should not call endJob.
            return;
          }
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
                                std::vector<bool>& beginStreamCalledForModule) noexcept(false) {
    beginStreamCalledForModule.clear();
    beginStreamCalledForModule.resize(iModuleRegistry.maxModuleID() + 1, false);

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
          iActivityRegistry.preModuleBeginStreamSignal_(iStreamContext, mcc);
          holder->beginStream(iStreamContext.streamID());
          beginStreamCalledForModule[holder->moduleDescription().id()] = true;
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
    beginStreamCalledForModule.clear();
  }

  void runEndStreamForModules(StreamContext const& iStreamContext,
                              ModuleRegistry& iModuleRegistry,
                              ActivityRegistry& iActivityRegistry,
                              ExceptionCollector& collector,
                              std::mutex& collectorMutex,
                              std::vector<bool> const& beginStreamCalledForModule) noexcept {
    assert(beginStreamCalledForModule.empty() or
           beginStreamCalledForModule.size() == iModuleRegistry.maxModuleID() + 1);
    ParentContext pc(&iStreamContext);
    iModuleRegistry.forAllModuleHolders([&](auto& holder) {
      ModuleCallingContext mcc(&holder->moduleDescription());
      //Also sets a thread local
      ModuleContextSentry mccSentry(&mcc, pc);
      try {
        convertException::wrap([&]() {
          if (not beginStreamCalledForModule.empty() and
              !beginStreamCalledForModule[holder->moduleDescription().id()]) {
            // If beginStream was never called, we should not call endStream.
            return;
          }
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
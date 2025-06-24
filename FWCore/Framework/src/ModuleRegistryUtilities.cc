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

  void runBeginJobForModules(ModuleRegistry& iModuleRegistry,
                             edm::ActivityRegistry& iActivityRegistry,
                             std::vector<bool>& beginJobCalledForModule) noexcept(false) {
    unsigned int largestIndex = 0;

    iModuleRegistry.forAllModuleHolders([&largestIndex](auto& holder) {
      auto id = holder->moduleDescription().id();
      if (id > largestIndex) {
        largestIndex = id;
      }
    });

    beginJobCalledForModule.clear();
    beginJobCalledForModule.resize(largestIndex + 1, false);

    std::exception_ptr exceptionPtr;
    iModuleRegistry.forAllModuleHolders([&](auto& holder) {
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
          std::ostringstream ost;
          ost << "Calling method for module " << holder->moduleDescription().moduleName() << "/'"
              << holder->moduleDescription().moduleLabel() << "'";
          ex.addContext(ost.str());
          exceptionPtr = std::current_exception();
        }
      }
    });
    if (exceptionPtr) {
      std::rethrow_exception(exceptionPtr);
    }
    beginJobCalledForModule.clear();
  }

  void runEndJobForModules(ModuleRegistry& iModuleRegistry,
                           ActivityRegistry& iActivityRegistry,
                           ExceptionCollector& collector,
                           std::vector<bool> const& beginJobCalledForModule,
                           const char* context) noexcept {
    iModuleRegistry.forAllModuleHolders([&](auto& holder) {
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
        std::ostringstream ost;
        ost << "Calling method for module " << holder->moduleDescription().moduleName() << "/'"
            << holder->moduleDescription().moduleLabel() << "'";
        ex.addContext(ost.str());
        ex.addContext(context);
        collector.addException(ex);
      }
    });
  }

}  // namespace edm
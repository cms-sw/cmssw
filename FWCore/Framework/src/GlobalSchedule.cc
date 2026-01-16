#include "FWCore/Framework/interface/GlobalSchedule.h"
#include "FWCore/Framework/interface/maker/ModuleMaker.h"
#include "FWCore/Framework/src/TriggerResultInserter.h"
#include "FWCore/Framework/src/PathStatusInserter.h"
#include "FWCore/Framework/src/EndPathStatusInserter.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"
#include "FWCore/Framework/interface/ModuleRegistry.h"
#include "FWCore/Framework/interface/ModuleRegistryUtilities.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ExceptionCollector.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/make_sentry.h"

#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <map>
#include <sstream>

namespace edm {
  GlobalSchedule::GlobalSchedule(
      std::shared_ptr<TriggerResultInserter> inserter,
      std::vector<edm::propagate_const<std::shared_ptr<PathStatusInserter>>>& pathStatusInserters,
      std::vector<edm::propagate_const<std::shared_ptr<EndPathStatusInserter>>>& endPathStatusInserters,
      std::shared_ptr<ModuleRegistry> modReg,
      std::vector<edm::ModuleDescription const*> const& iModulesToUse,
      PreallocationConfiguration const& prealloc,
      ExceptionToActionTable const& actions,
      std::shared_ptr<ActivityRegistry> areg,
      ProcessContext const* processContext)
      : actReg_(areg),
        processContext_(processContext),
        numberOfConcurrentLumis_(prealloc.numberOfLuminosityBlocks()),
        numberOfConcurrentRuns_(prealloc.numberOfRuns()) {
    unsigned int nManagers =
        prealloc.numberOfLuminosityBlocks() + prealloc.numberOfRuns() + numberOfConcurrentProcessBlocks_;
    workerManagers_.reserve(nManagers);
    for (unsigned int i = 0; i < nManagers; ++i) {
      workerManagers_.emplace_back(modReg, areg, actions);
    }
    for (auto const& module : iModulesToUse) {
      //side effect keeps this module around
      for (auto& wm : workerManagers_) {
        (void)wm.getWorkerForModule(*module);
      }
    }
    if (inserter) {
      for (auto& wm : workerManagers_) {
        (void)wm.getWorkerForModule(*inserter);
      }
    }

    for (auto& pathStatusInserter : pathStatusInserters) {
      std::shared_ptr<PathStatusInserter> inserterPtr = get_underlying(pathStatusInserter);

      for (auto& wm : workerManagers_) {
        (void)wm.getWorkerForModule(*inserterPtr);
      }
    }

    for (auto& endPathStatusInserter : endPathStatusInserters) {
      std::shared_ptr<EndPathStatusInserter> inserterPtr = get_underlying(endPathStatusInserter);
      for (auto& wm : workerManagers_) {
        (void)wm.getWorkerForModule(*inserterPtr);
      }
    }

  }  // GlobalSchedule::GlobalSchedule

  void GlobalSchedule::beginJob(ModuleRegistry& modReg) {
    constexpr static char const* const globalContext = "Processing begin Job";

    GlobalContext gc(GlobalContext::Transition::kBeginJob, processContext_);
    std::exception_ptr exceptionPtr;
    try {
      convertException::wrap([this]() { actReg_->preBeginJobSignal_.emit(*processContext_); });
    } catch (cms::Exception& ex) {
      ex.addContext("Handling pre signal, likely in a service function");
      ex.addContext(globalContext);
      exceptionPtr = std::current_exception();
    }
    if (not exceptionPtr) {
      try {
        runBeginJobForModules(gc, modReg, *actReg_, beginJobFailedForModule_);
      } catch (cms::Exception& ex) {
        ex.addContext(globalContext);
        exceptionPtr = std::current_exception();
      }
    }
    try {
      convertException::wrap([this]() { actReg_->postBeginJobSignal_.emit(); });
    } catch (cms::Exception& ex) {
      if (!exceptionPtr) {
        ex.addContext("Handling post signal, likely in a service function");
        ex.addContext(globalContext);
        exceptionPtr = std::current_exception();
      }
    }
    if (exceptionPtr) {
      std::rethrow_exception(exceptionPtr);
    }
  }

  void GlobalSchedule::endJob(ExceptionCollector& collector, ModuleRegistry& modReg) {
    constexpr static char const* const context = "Processing end Job";
    GlobalContext gc(GlobalContext::Transition::kEndJob, processContext_);
    std::exception_ptr exceptionPtr;
    try {
      convertException::wrap([this]() { actReg_->preEndJobSignal_.emit(); });
    } catch (cms::Exception& ex) {
      ex.addContext("Handling pre signal, likely in a service function");
      ex.addContext(context);
      exceptionPtr = std::current_exception();
    }
    if (not exceptionPtr) {
      runEndJobForModules(gc, modReg, *actReg_, collector, beginJobFailedForModule_);
    }

    try {
      convertException::wrap([this]() { actReg_->postEndJobSignal_.emit(); });
    } catch (cms::Exception& ex) {
      if (!exceptionPtr) {
        ex.addContext("Handling post signal, likely in a service function");
        ex.addContext(context);
        exceptionPtr = std::current_exception();
      }
    }
    if (exceptionPtr) {
      collector.call([&exceptionPtr]() { std::rethrow_exception(exceptionPtr); });
    }
  }

  void GlobalSchedule::replaceModule(maker::ModuleHolder* iMod, std::string const& iLabel) {
    Worker* found = nullptr;
    for (auto& wm : workerManagers_) {
      for (auto const& worker : wm.allWorkers()) {
        if (worker->description()->moduleLabel() == iLabel) {
          found = worker;
          break;
        }
      }
      if (nullptr == found) {
        return;
      }
      iMod->replaceModuleFor(found);
    }
    auto sentry = make_sentry(
        iMod, [&](auto const* mod) { beginJobFailedForModule_.emplace_back(mod->moduleDescription().id()); });
    iMod->beginJob();
    sentry.release();
  }

  void GlobalSchedule::deleteModule(std::string const& iLabel) {
    for (auto& wm : workerManagers_) {
      wm.deleteModuleIfExists(iLabel);
    }
  }

  std::vector<ModuleDescription const*> GlobalSchedule::getAllModuleDescriptions() const {
    std::vector<ModuleDescription const*> result;
    result.reserve(allWorkers().size());

    for (auto const& worker : allWorkers()) {
      ModuleDescription const* p = worker->description();
      result.push_back(p);
    }
    return result;
  }

  void GlobalSchedule::handleException(GlobalContext const* globalContext,
                                       ServiceWeakToken const& weakToken,
                                       bool cleaningUpAfterException,
                                       std::exception_ptr& excpt) {
    //add context information to the exception and print message
    try {
      convertException::wrap([&excpt]() { std::rethrow_exception(excpt); });
    } catch (cms::Exception& ex) {
      std::ostringstream ost;
      // In most cases the exception will already have context at this point,
      // but add some context here in those rare cases where it does not.
      if (ex.context().empty()) {
        exceptionContext(ost, *globalContext);
      }
      ServiceRegistry::Operate op(weakToken.lock());
      addContextAndPrintException(ost.str().c_str(), ex, cleaningUpAfterException);
      excpt = std::current_exception();
    }
    // We are already handling an earlier exception, so ignore it
    // if this signal results in another exception being thrown.
    CMS_SA_ALLOW try {
      if (actReg_) {
        ServiceRegistry::Operate op(weakToken.lock());
        actReg_->preGlobalEarlyTerminationSignal_.emit(*globalContext, TerminationOrigin::ExceptionFromThisContext);
      }
    } catch (...) {
    }
  }

}  // namespace edm

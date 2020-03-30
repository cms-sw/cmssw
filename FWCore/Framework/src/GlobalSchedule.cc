#include "FWCore/Framework/src/GlobalSchedule.h"
#include "FWCore/Framework/src/WorkerMaker.h"
#include "FWCore/Framework/src/TriggerResultInserter.h"
#include "FWCore/Framework/src/PathStatusInserter.h"
#include "FWCore/Framework/src/EndPathStatusInserter.h"
#include "FWCore/Framework/src/PreallocationConfiguration.h"

#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <map>

namespace edm {
  GlobalSchedule::GlobalSchedule(
      std::shared_ptr<TriggerResultInserter> inserter,
      std::vector<edm::propagate_const<std::shared_ptr<PathStatusInserter>>>& pathStatusInserters,
      std::vector<edm::propagate_const<std::shared_ptr<EndPathStatusInserter>>>& endPathStatusInserters,
      std::shared_ptr<ModuleRegistry> modReg,
      std::vector<std::string> const& iModulesToUse,
      ParameterSet& proc_pset,
      ProductRegistry& pregistry,
      PreallocationConfiguration const& prealloc,
      ExceptionToActionTable const& actions,
      std::shared_ptr<ActivityRegistry> areg,
      std::shared_ptr<ProcessConfiguration> processConfiguration,
      ProcessContext const* processContext)
      : actReg_(areg), processContext_(processContext) {
    workerManagers_.reserve(prealloc.numberOfLuminosityBlocks());
    for (unsigned int i = 0; i < prealloc.numberOfLuminosityBlocks(); ++i) {
      workerManagers_.emplace_back(modReg, areg, actions);
    }
    for (auto const& moduleLabel : iModulesToUse) {
      bool isTracked;
      ParameterSet* modpset = proc_pset.getPSetForUpdate(moduleLabel, isTracked);
      if (modpset != nullptr) {  // It will be null for PathStatusInserters, it should
                                 // be impossible to be null for anything else
        assert(isTracked);

        //side effect keeps this module around
        for (auto& wm : workerManagers_) {
          wm.addToAllWorkers(wm.getWorker(*modpset, pregistry, &prealloc, processConfiguration, moduleLabel));
        }
      }
    }
    if (inserter) {
      inserter->doPreallocate(prealloc);
      for (auto& wm : workerManagers_) {
        auto results_inserter = WorkerPtr(new edm::WorkerT<TriggerResultInserter::ModuleType>(
            inserter, inserter->moduleDescription(), &actions));  // propagate_const<T> has no reset() function
        results_inserter->setActivityRegistry(actReg_);
        wm.addToAllWorkers(results_inserter.get());
        extraWorkers_.emplace_back(std::move(results_inserter));
      }
    }

    for (auto& pathStatusInserter : pathStatusInserters) {
      std::shared_ptr<PathStatusInserter> inserterPtr = get_underlying(pathStatusInserter);
      inserterPtr->doPreallocate(prealloc);

      for (auto& wm : workerManagers_) {
        WorkerPtr workerPtr(
            new edm::WorkerT<PathStatusInserter::ModuleType>(inserterPtr, inserterPtr->moduleDescription(), &actions));
        workerPtr->setActivityRegistry(actReg_);
        wm.addToAllWorkers(workerPtr.get());
        extraWorkers_.emplace_back(std::move(workerPtr));
      }
    }

    for (auto& endPathStatusInserter : endPathStatusInserters) {
      std::shared_ptr<EndPathStatusInserter> inserterPtr = get_underlying(endPathStatusInserter);
      inserterPtr->doPreallocate(prealloc);
      for (auto& wm : workerManagers_) {
        WorkerPtr workerPtr(new edm::WorkerT<EndPathStatusInserter::ModuleType>(
            inserterPtr, inserterPtr->moduleDescription(), &actions));
        workerPtr->setActivityRegistry(actReg_);
        wm.addToAllWorkers(workerPtr.get());
        extraWorkers_.emplace_back(std::move(workerPtr));
      }
    }

  }  // GlobalSchedule::GlobalSchedule

  void GlobalSchedule::endJob(ExceptionCollector& collector) { workerManagers_[0].endJob(collector); }

  void GlobalSchedule::beginJob(ProductRegistry const& iRegistry,
                                eventsetup::ESRecordsToProxyIndices const& iESIndices) {
    workerManagers_[0].beginJob(iRegistry, iESIndices);
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
      found->beginJob();
    }
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
}  // namespace edm

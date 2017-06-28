#include "FWCore/Framework/src/GlobalSchedule.h"
#include "FWCore/Framework/src/WorkerMaker.h"
#include "FWCore/Framework/src/TriggerResultInserter.h"
#include "FWCore/Framework/src/PathStatusInserter.h"
#include "FWCore/Framework/src/EndPathStatusInserter.h"

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
  GlobalSchedule::GlobalSchedule(std::shared_ptr<TriggerResultInserter> inserter,
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
                                 ProcessContext const* processContext) :
    workerManager_(modReg,areg,actions),
    actReg_(areg),
    processContext_(processContext)
  {
    for (auto const& moduleLabel : iModulesToUse) {
      bool isTracked;
      ParameterSet* modpset = proc_pset.getPSetForUpdate(moduleLabel, isTracked);
      if (modpset != nullptr) { // It will be null for PathStatusInserters, it should
                                // be impossible to be null for anything else
        assert(isTracked);

        //side effect keeps this module around
        addToAllWorkers(workerManager_.getWorker(*modpset, pregistry, &prealloc,processConfiguration, moduleLabel));
      }
    }
    if(inserter) {
      results_inserter_ = WorkerPtr(new edm::WorkerT<TriggerResultInserter::ModuleType>(inserter, inserter->moduleDescription(), &actions)); // propagate_const<T> has no reset() function
      inserter->doPreallocate(prealloc);
      results_inserter_->setActivityRegistry(actReg_);
      addToAllWorkers(results_inserter_.get());
    }

    for(auto & pathStatusInserter : pathStatusInserters) {
      std::shared_ptr<PathStatusInserter> inserterPtr = get_underlying(pathStatusInserter);
      WorkerPtr workerPtr(new edm::WorkerT<PathStatusInserter::ModuleType>(inserterPtr,
                                                                           inserterPtr->moduleDescription(),
                                                                           &actions));
      pathStatusInserterWorkers_.emplace_back(workerPtr);
      inserterPtr->doPreallocate(prealloc);
      workerPtr->setActivityRegistry(actReg_);
      addToAllWorkers(workerPtr.get());
    }

    for(auto & endPathStatusInserter : endPathStatusInserters) {
      std::shared_ptr<EndPathStatusInserter> inserterPtr = get_underlying(endPathStatusInserter);
      WorkerPtr workerPtr(new edm::WorkerT<EndPathStatusInserter::ModuleType>(inserterPtr,
                                                                              inserterPtr->moduleDescription(),
                                                                              &actions));
      endPathStatusInserterWorkers_.emplace_back(workerPtr);
      inserterPtr->doPreallocate(prealloc);
      workerPtr->setActivityRegistry(actReg_);
      addToAllWorkers(workerPtr.get());
    }

  } // GlobalSchedule::GlobalSchedule

  void GlobalSchedule::endJob(ExceptionCollector & collector) {
    workerManager_.endJob(collector);
  }

  void GlobalSchedule::beginJob(ProductRegistry const& iRegistry) {
    workerManager_.beginJob(iRegistry);
  }
  
  void GlobalSchedule::replaceModule(maker::ModuleHolder* iMod,
                                     std::string const& iLabel) {
    Worker* found = nullptr;
    for (auto const& worker : allWorkers()) {
      if (worker->description().moduleLabel() == iLabel) {
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

  std::vector<ModuleDescription const*>
  GlobalSchedule::getAllModuleDescriptions() const {
    std::vector<ModuleDescription const*> result;
    result.reserve(allWorkers().size());

    for (auto const& worker : allWorkers()) {
      ModuleDescription const* p = worker->descPtr();
      result.push_back(p);
    }
    return result;
  }

  void
  GlobalSchedule::addToAllWorkers(Worker* w) {
    workerManager_.addToAllWorkers(w);
  }

}

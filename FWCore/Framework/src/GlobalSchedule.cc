#include "FWCore/Framework/src/GlobalSchedule.h"
#include "FWCore/Framework/src/WorkerMaker.h"
#include "FWCore/Framework/src/TriggerResultInserter.h"

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
  GlobalSchedule::GlobalSchedule(TriggerResultInserter* inserter,
                                 boost::shared_ptr<ModuleRegistry> modReg,
                                 std::vector<std::string> const& iModulesToUse,
                                 ParameterSet& proc_pset,
                                 ProductRegistry& pregistry,
                                 ExceptionToActionTable const& actions,
                                 boost::shared_ptr<ActivityRegistry> areg,
                                 boost::shared_ptr<ProcessConfiguration> processConfiguration,
                                 ProcessContext const* processContext) :
    workerManager_(modReg,areg,actions),
    actReg_(areg),
    processContext_(processContext)
  {
    for (auto const& moduleLabel : iModulesToUse) {
      bool isTracked;
      ParameterSet* modpset = proc_pset.getPSetForUpdate(moduleLabel, isTracked);
      if (modpset == 0) {
        throw Exception(errors::Configuration) <<
        "The unknown module label \"" << moduleLabel <<
        "\"\n please check spelling";
      }
      assert(isTracked);
      
      //side effect keeps this module around
      addToAllWorkers(workerManager_.getWorker(*modpset, pregistry, processConfiguration, moduleLabel));

    }
    if(inserter) {
      results_inserter_.reset(new edm::WorkerT<TriggerResultInserter::ModuleType>(inserter, inserter->moduleDescription(), &actions));
      results_inserter_->setActivityRegistry(actReg_);
      addToAllWorkers(results_inserter_.get());
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
    workerManager_.addToAllWorkers(w, false);
  }

}

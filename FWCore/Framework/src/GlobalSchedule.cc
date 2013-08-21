#include "FWCore/Framework/src/GlobalSchedule.h"

#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/OutputModuleDescription.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Framework/interface/TriggerReport.h"
#include "FWCore/Framework/src/Factory.h"
#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/src/OutputModuleCommunicator.h"
#include "FWCore/Framework/src/TriggerResultInserter.h"
#include "FWCore/Framework/src/WorkerInPath.h"
#include "FWCore/Framework/src/WorkerMaker.h"
#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FillProductRegistryTransients.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/ExceptionCollector.h"
#include "FWCore/Utilities/interface/DictionaryTools.h"

#include "boost/bind.hpp"
#include "boost/ref.hpp"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <list>
#include <map>
#include <exception>

namespace edm {
  namespace {

    // Function template to transform each element in the input range to
    // a value placed into the output range. The supplied function
    // should take a const_reference to the 'input', and write to a
    // reference to the 'output'.
    template <typename InputIterator, typename ForwardIterator, typename Func>
    void
    transform_into(InputIterator begin, InputIterator end,
                   ForwardIterator out, Func func) {
      for (; begin != end; ++begin, ++out) func(*begin, *out);
    }

    // Function template that takes a sequence 'from', a sequence
    // 'to', and a callable object 'func'. It and applies
    // transform_into to fill the 'to' sequence with the values
    // calcuated by the callable object, taking care to fill the
    // outupt only if all calls succeed.
    template <typename FROM, typename TO, typename FUNC>
    void
    fill_summary(FROM const& from, TO& to, FUNC func) {
      TO temp(from.size());
      transform_into(from.begin(), from.end(), temp.begin(), func);
      to.swap(temp);
    }

    // -----------------------------

    bool binary_search_string(std::vector<std::string> const& v, std::string const& s) {
      return std::binary_search(v.begin(), v.end(), s);
    }
  }

  // -----------------------------

  typedef std::vector<std::string> vstring;

  // -----------------------------

  GlobalSchedule::GlobalSchedule(boost::shared_ptr<ModuleRegistry> modReg,
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

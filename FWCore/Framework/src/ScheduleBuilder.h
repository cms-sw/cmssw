#ifndef FWCore_Framework_ScheduleBuilder_h
#define FWCore_Framework_ScheduleBuilder_h

#include "FWCore/Framework/interface/ModuleRegistry.h"
#include "FWCore/Framework/interface/ModuleInPath.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include <memory>
#include <vector>
#include <string>
namespace edm {
  class ParameterSet;
  class PreallocationConfiguration;
  class SignallingProductRegistryFiller;
  class ModuleDescription;
  class TriggerResultInserter;
  class PathStatusInserter;
  class EndPathStatusInserter;
  class ProcessConfiguration;

  namespace service {
    class TriggerNamesService;
  }

  struct ScheduleBuilder {
  public:
    ScheduleBuilder(ModuleRegistry& iModuleRegistry,
                    ParameterSet& ioProcessPSet,
                    std::vector<std::string> const& iPathNames,
                    std::vector<std::string> const& iEndPathNames,
                    PreallocationConfiguration const& iPrealloc,
                    SignallingProductRegistryFiller& oProductRegistry,
                    ActivityRegistry& iActivityRegistry,
                    std::shared_ptr<ProcessConfiguration const> iProcessConfiguration);

    //the order matches the order from iPathNames and iEndPathNames
    std::vector<std::pair<std::string, std::vector<ModuleInPath>>> pathNameAndModules_;
    std::vector<std::pair<std::string, std::vector<ModuleInPath>>> endpathNameAndModules_;

    std::vector<edm::ModuleDescription const*> allNeededModules_;
    std::vector<edm::ModuleDescription const*> unscheduledModules_;

    std::shared_ptr<TriggerResultInserter> resultsInserter_;
    std::vector<edm::propagate_const<std::shared_ptr<PathStatusInserter>>> pathStatusInserters_;
    std::vector<edm::propagate_const<std::shared_ptr<EndPathStatusInserter>>> endPathStatusInserters_;
  };
}  // namespace edm
#endif
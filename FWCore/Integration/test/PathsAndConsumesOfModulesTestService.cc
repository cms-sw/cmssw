#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/PathsAndConsumesOfModulesBase.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include <set>
#include <string>
#include <utility>
#include <vector>

namespace edmtest {
  class PathsAndConsumesOfModulesTestService {
  public:
    PathsAndConsumesOfModulesTestService(edm::ParameterSet const& pset, edm::ActivityRegistry& iRegistry)
        : modulesConsumes_(pset.getParameter<decltype(modulesConsumes_)>("modulesAndConsumes")) {
      iRegistry.watchPreBeginJob(this, &PathsAndConsumesOfModulesTestService::preBeginJob);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;

      edm::ParameterSetDescription validator;
      validator.add<std::string>("key");
      validator.add<std::vector<std::string>>("value");
      desc.addVPSet("modulesAndConsumes", validator, std::vector<edm::ParameterSet>());

      descriptions.addWithDefaultLabel(desc);
      descriptions.setComment("This service is intended to be used in framework tests.");
    }

    void preBeginJob(edm::PathsAndConsumesOfModulesBase const& pathsAndConsumes, edm::ProcessContext const&) const {
      auto const& allModules = pathsAndConsumes.allModules();
      for (auto const& moduleToCheck : modulesConsumes_) {
        auto found =
            std::find_if(allModules.begin(), allModules.end(), [&moduleToCheck](edm::ModuleDescription const* desc) {
              return desc->moduleLabel() == moduleToCheck.first;
            });
        if (found == allModules.end()) {
          cms::Exception ex("TestFailure");
          ex << "Module " << moduleToCheck.first
             << " not found in PathsAndConsumesOfModulesBase, that has the following modules:\n";
          for (edm::ModuleDescription const* desc : allModules) {
            if (desc) {
              ex << " " << desc->moduleLabel() << "\n";
            } else {
              ex << " nullptr\n";
            }
          }
          throw ex;
        }

        std::set<std::string> tocheck(moduleToCheck.second.begin(), moduleToCheck.second.end());
        for (edm::ModuleDescription const* desc : pathsAndConsumes.modulesWhoseProductsAreConsumedBy((*found)->id())) {
          auto found = tocheck.find(desc->moduleLabel());
          if (found == tocheck.end()) {
            cms::Exception ex("TestFailure");
            ex << "Module " << moduleToCheck.first << " consumes " << desc->moduleLabel()
               << " that was not one of the expected modules:\n";
            for (auto const& m : moduleToCheck.second) {
              ex << " " << m << "\n";
            }
            throw ex;
          }
          tocheck.erase(found);
        }
        if (not tocheck.empty()) {
          cms::Exception ex("TestFailure");
          ex << "Module " << moduleToCheck.first << " was expected to consume the following modules, but it did not\n";
          for (auto const& m : tocheck) {
            ex << " " << m << "\n";
          }
          throw ex;
        }
      }
    }

  private:
    std::vector<std::pair<std::string, std::vector<std::string>>> modulesConsumes_;
  };
}  // namespace edmtest

using edmtest::PathsAndConsumesOfModulesTestService;
DEFINE_FWK_SERVICE(PathsAndConsumesOfModulesTestService);

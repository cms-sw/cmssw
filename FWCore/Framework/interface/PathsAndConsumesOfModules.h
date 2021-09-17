#ifndef FWCore_Framework_PathsAndConsumesOfModules_h
#define FWCore_Framework_PathsAndConsumesOfModules_h

/**\class edm::PathsAndConsumesOfModules

 Description: See comments in the base class

 Usage:

*/
//
// Original Author: W. David Dagenhart
//         Created: 11/5/2014

#include "FWCore/ServiceRegistry/interface/ConsumesInfo.h"
#include "FWCore/ServiceRegistry/interface/PathsAndConsumesOfModulesBase.h"

#include "FWCore/Framework/interface/ModuleProcessName.h"
#include "FWCore/Utilities/interface/BranchType.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace edm {

  class ModuleDescription;
  class ProductRegistry;
  class Schedule;

  class PathsAndConsumesOfModules : public PathsAndConsumesOfModulesBase {
  public:
    PathsAndConsumesOfModules();
    ~PathsAndConsumesOfModules() override;

    void initialize(Schedule const*, std::shared_ptr<ProductRegistry const>);

    void removeModules(std::vector<ModuleDescription const*> const& modules);

    std::vector<ModuleProcessName> const& modulesInPreviousProcessesWhoseProductsAreConsumedBy(
        unsigned int moduleID) const;

  private:
    std::vector<std::string> const& doPaths() const override { return paths_; }
    std::vector<std::string> const& doEndPaths() const override { return endPaths_; }

    std::vector<ModuleDescription const*> const& doAllModules() const override { return allModuleDescriptions_; }
    ModuleDescription const* doModuleDescription(unsigned int moduleID) const override;

    std::vector<ModuleDescription const*> const& doModulesOnPath(unsigned int pathIndex) const override;
    std::vector<ModuleDescription const*> const& doModulesOnEndPath(unsigned int endPathIndex) const override;
    std::vector<ModuleDescription const*> const& doModulesWhoseProductsAreConsumedBy(
        unsigned int moduleID, BranchType branchType) const override;

    std::vector<ConsumesInfo> doConsumesInfo(unsigned int moduleID) const override;

    unsigned int doLargestModuleID() const override;

    unsigned int moduleIndex(unsigned int moduleID) const;

    // data members

    std::vector<std::string> paths_;
    std::vector<std::string> endPaths_;

    std::vector<ModuleDescription const*> allModuleDescriptions_;

    std::vector<std::vector<ModuleDescription const*> > modulesOnPaths_;
    std::vector<std::vector<ModuleDescription const*> > modulesOnEndPaths_;

    // Gives a translation from the module ID to the index into the
    // following data member
    std::vector<std::pair<unsigned int, unsigned int> > moduleIDToIndex_;

    std::array<std::vector<std::vector<ModuleDescription const*> >, NumBranchTypes> modulesWhoseProductsAreConsumedBy_;
    std::vector<std::vector<ModuleProcessName> > modulesInPreviousProcessesWhoseProductsAreConsumedBy_;

    Schedule const* schedule_;
    std::shared_ptr<ProductRegistry const> preg_;
  };

  std::vector<ModuleDescription const*> nonConsumedUnscheduledModules(
      edm::PathsAndConsumesOfModulesBase const& iPnC, std::vector<ModuleProcessName>& consumedByChildren);

  void checkForModuleDependencyCorrectness(edm::PathsAndConsumesOfModulesBase const& iPnC, bool iPrintDependencies);
}  // namespace edm
#endif

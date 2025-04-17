#ifndef FWCore_Framework_PathsAndConsumesOfModules_h
#define FWCore_Framework_PathsAndConsumesOfModules_h

/**\class edm::PathsAndConsumesOfModules

 Description: See comments in the base class

 Usage:

*/
//
// Original Author: W. David Dagenhart
//         Created: 11/5/2014

#include "FWCore/ServiceRegistry/interface/PathsAndConsumesOfModulesBase.h"

#include "FWCore/Framework/interface/ESRecordsToProductResolverIndices.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistryfwd.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/Transition.h"

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace edm {

  class ModuleDescription;
  class ProductRegistry;
  class Schedule;

  namespace eventsetup {
    struct ComponentDescription;
    class ESProductResolverProvider;
    class EventSetupProvider;
  }  // namespace eventsetup

  class PathsAndConsumesOfModules : public PathsAndConsumesOfModulesBase {
  public:
    PathsAndConsumesOfModules() = default;
    ~PathsAndConsumesOfModules() override = default;

    void initialize(Schedule const*, std::shared_ptr<ProductRegistry const>);
    void initializeForEventSetup(eventsetup::ESRecordsToProductResolverIndices&&,
                                 eventsetup::EventSetupProvider const&);
    void checkEventSetupInitialization() const;

    void removeModules(std::vector<ModuleDescription const*> const& modules);

  private:
    std::vector<std::string> const& doPaths() const override;
    std::vector<std::string> const& doEndPaths() const override;

    std::vector<ModuleDescription const*> const& doAllModules() const override;
    ModuleDescription const* doModuleDescription(unsigned int moduleID) const override;

    std::vector<ModuleDescription const*> const& doModulesOnPath(unsigned int pathIndex) const override;
    std::vector<ModuleDescription const*> const& doModulesOnEndPath(unsigned int endPathIndex) const override;
    std::vector<ModuleDescription const*> const& doModulesWhoseProductsAreConsumedBy(
        unsigned int moduleID, BranchType branchType) const override;

    std::vector<eventsetup::ComponentDescription const*> const& doESModulesWhoseProductsAreConsumedBy(
        unsigned int moduleID, Transition) const override;

    std::vector<ModuleConsumesInfo> doModuleConsumesInfos(unsigned int moduleID) const override;
    std::vector<ModuleConsumesESInfo> doModuleConsumesESInfos(unsigned int moduleID) const override;

    unsigned int doLargestModuleID() const override;

    std::vector<eventsetup::ComponentDescription const*> const& doAllESModules() const override;
    eventsetup::ComponentDescription const* doComponentDescription(unsigned int esModuleID) const override;

    std::vector<std::vector<eventsetup::ComponentDescription const*>> const&
    doESModulesWhoseProductsAreConsumedByESModule() const override;
    std::vector<std::vector<ESModuleConsumesInfo>> doESModuleConsumesInfos(unsigned int esModuleID) const override;

    unsigned int moduleIndex(unsigned int moduleID) const;
    unsigned int esModuleIndex(unsigned int esModuleID) const;

    // data members

    std::vector<std::string> paths_;
    std::vector<std::string> endPaths_;

    std::vector<ModuleDescription const*> allModuleDescriptions_;

    std::vector<std::vector<ModuleDescription const*>> modulesOnPaths_;
    std::vector<std::vector<ModuleDescription const*>> modulesOnEndPaths_;

    // Gives a translation from the module ID to the index into the
    // following data member
    std::vector<std::pair<unsigned int, unsigned int>> moduleIDToIndex_;

    std::array<std::vector<std::vector<ModuleDescription const*>>, NumBranchTypes> modulesWhoseProductsAreConsumedBy_;

    std::array<std::vector<std::vector<eventsetup::ComponentDescription const*>>, kNumberOfEventSetupTransitions>
        esModulesWhoseProductsAreConsumedBy_;

    // Gives a translation from the componentID to the index into allComponentDescriptions_,
    // allESProductResolverProviders_, and the outer vector of esModulesWhoseProductsAreConsumedByESModule_
    std::vector<std::pair<unsigned int, unsigned int>> esModuleIDToIndex_;

    std::vector<eventsetup::ComponentDescription const*> allComponentDescriptions_;
    std::vector<eventsetup::ESProductResolverProvider const*> allESProductResolverProviders_;
    std::vector<std::vector<eventsetup::ComponentDescription const*>> esModulesWhoseProductsAreConsumedByESModule_;

    Schedule const* schedule_ = nullptr;
    eventsetup::ESRecordsToProductResolverIndices esRecordsToProductResolverIndices_;
    std::shared_ptr<ProductRegistry const> preg_;
    bool eventSetupInfoInitialized_ = false;
  };

  std::vector<ModuleDescription const*> nonConsumedUnscheduledModules(edm::PathsAndConsumesOfModulesBase const& iPnC);

  void checkForModuleDependencyCorrectness(edm::PathsAndConsumesOfModulesBase const& iPnC, bool iPrintDependencies);
}  // namespace edm
#endif

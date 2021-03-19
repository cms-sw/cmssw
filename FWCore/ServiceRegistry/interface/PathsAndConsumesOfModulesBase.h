#ifndef FWCore_ServiceRegistry_PathsAndConsumesOfModulesBase_h
#define FWCore_ServiceRegistry_PathsAndConsumesOfModulesBase_h

/**\class edm::PathsAndConsumesOfModulesBase

 Description: Contains information about paths and end paths
 as well as the modules on them. Also contains information
 about all modules that might run. Also contains information
 about the products a module is declared to consume and the
 dependences between modules which can be derived from
 those declarations.

 Usage: This is typically passed as an argument to the
 PreBeginJob callback for a service.

 In a SubProcess job, an instance of this class this will
 contain information about 1 Process/SubProcess, but a
 service will be passed a separate object for its process
 and each SubProcess descended from it.
*/
//
// Original Author: W. David Dagenhart
//         Created: 11/5/2014

#include "FWCore/ServiceRegistry/interface/ConsumesInfo.h"
#include "FWCore/Utilities/interface/BranchType.h"

#include <string>
#include <vector>

namespace edm {

  class ModuleDescription;

  class PathsAndConsumesOfModulesBase {
  public:
    virtual ~PathsAndConsumesOfModulesBase();

    std::vector<std::string> const& paths() const { return doPaths(); }
    std::vector<std::string> const& endPaths() const { return doEndPaths(); }

    std::vector<ModuleDescription const*> const& allModules() const { return doAllModules(); }

    ModuleDescription const* moduleDescription(unsigned int moduleID) const { return doModuleDescription(moduleID); }

    std::vector<ModuleDescription const*> const& modulesOnPath(unsigned int pathIndex) const {
      return doModulesOnPath(pathIndex);
    }

    std::vector<ModuleDescription const*> const& modulesOnEndPath(unsigned int endPathIndex) const {
      return doModulesOnEndPath(endPathIndex);
    }

    // The modules in the returned vector will be from the current process
    // (not the prior process, and it will never include the source even
    // though the source can make products) and these modules will declare
    // they produce (they might or might not really produce) at least one
    // product in the event (not run, not lumi) that the module corresponding
    // to the moduleID argument declares it consumes (includes declarations using
    // consumes, maybeConsumes, or consumesMany). Note that if a module declares
    // it consumes a module label that is an EDAlias, the corresponding module
    // description will be included in the returned vector (but the label in the
    // module description is not the EDAlias label).
    std::vector<ModuleDescription const*> const& modulesWhoseProductsAreConsumedBy(
        unsigned int moduleID, BranchType branchType = InEvent) const {
      return doModulesWhoseProductsAreConsumedBy(moduleID, branchType);
    }

    // This returns the declared consumes information for a module.
    // Note the other functions above return a reference to an object
    // that is held in memory throughout the job, while the following
    // function returns a newly created object each time.  We do not
    // expect this to be called during a normal production job where
    // performance and memory are important. These objects are bigger
    // than just a pointer.
    std::vector<ConsumesInfo> consumesInfo(unsigned int moduleID) const { return doConsumesInfo(moduleID); }

    unsigned int largestModuleID() const { return doLargestModuleID(); }

  private:
    virtual std::vector<std::string> const& doPaths() const = 0;
    virtual std::vector<std::string> const& doEndPaths() const = 0;
    virtual std::vector<ModuleDescription const*> const& doAllModules() const = 0;
    virtual ModuleDescription const* doModuleDescription(unsigned int moduleID) const = 0;
    virtual std::vector<ModuleDescription const*> const& doModulesOnPath(unsigned int pathIndex) const = 0;
    virtual std::vector<ModuleDescription const*> const& doModulesOnEndPath(unsigned int endPathIndex) const = 0;
    virtual std::vector<ModuleDescription const*> const& doModulesWhoseProductsAreConsumedBy(
        unsigned int moduleID, BranchType branchType) const = 0;
    virtual std::vector<ConsumesInfo> doConsumesInfo(unsigned int moduleID) const = 0;
    virtual unsigned int doLargestModuleID() const = 0;
  };
}  // namespace edm
#endif

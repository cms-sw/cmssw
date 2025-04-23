#ifndef FWCore_ServiceRegistry_PathsAndConsumesOfModulesBase_h
#define FWCore_ServiceRegistry_PathsAndConsumesOfModulesBase_h

/**\class edm::PathsAndConsumesOfModulesBase

 Description: Contains information about paths and end paths
 as well as the modules on them. Also contains information
 about all modules that might run. Also contains information
 about the products a module is declared to consume and the
 dependences between modules which can be derived from
 those declarations. Also contains similar information
 related to EventSetup modules.

 Usage: This is typically passed as an argument to the
 callback method for the LookupInitializationComplete
 signal that a Service may watch.
*/
//
// Original Author: W. David Dagenhart
//         Created: 11/5/2014

#include "FWCore/ServiceRegistry/interface/ServiceRegistryfwd.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/Transition.h"

#include <string>
#include <vector>

namespace edm {

  class ModuleDescription;

  namespace eventsetup {
    struct ComponentDescription;
  }  // namespace eventsetup

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

    // The ED modules in the returned vector will be from the current process
    // (not a prior process, and it will never include the source even
    // though the source can make products) and these modules will declare
    // they produce (they might or might not really produce) at least one
    // product associated with the branchType that the module corresponding
    // to the moduleID argument declares it consumes (includes declarations using
    // consumes or maybeConsumes). Note that if a module declares
    // it consumes a module label that is an EDAlias, the corresponding module
    // description will be included in the returned vector (but the label in the
    // module description is not the EDAlias label).
    std::vector<ModuleDescription const*> const& modulesWhoseProductsAreConsumedBy(
        unsigned int moduleID, BranchType branchType = InEvent) const {
      return doModulesWhoseProductsAreConsumedBy(moduleID, branchType);
    }

    // The EventSetup modules in the returned vector are associated with
    // the ED module identified by the moduleID argument. The other
    // argument named transition could for example be beginRun. The
    // returned EventSetup modules produce the data products consumed
    // by the ED module during the specified transition. In cases
    // where more than one module produces a product, only the preferred
    // producer is included. There may be some consumed products without
    // a producing module.
    std::vector<eventsetup::ComponentDescription const*> const& esModulesWhoseProductsAreConsumedBy(
        unsigned int moduleID, Transition transition) const {
      return doESModulesWhoseProductsAreConsumedBy(moduleID, transition);
    }

    // These next two functions return the declared consumes information
    // for a single EDModule (the one associated with the moduleID).
    // ModuleConsumesInfo objects for all transitions are in the same vector,
    // although the elements contain a member that indicates the branch type
    // for ED products and the transition associated with the consumes call
    // for EventSetup products.
    // Note the other functions above return a reference to
    // an object that is held in memory throughout the job, while the
    // following function returns a newly created object each time.  We
    // do not expect this to be called during a normal production job where
    // performance and memory are important. These objects are bigger
    // than just a pointer.
    std::vector<ModuleConsumesInfo> moduleConsumesInfos(unsigned int moduleID) const;

    std::vector<ModuleConsumesESInfo> moduleConsumesESInfos(unsigned int moduleID) const;

    unsigned int largestModuleID() const { return doLargestModuleID(); }

    // The returned vector contains an entry for every EventSetup module.
    std::vector<eventsetup::ComponentDescription const*> const& allESModules() const { return doAllESModules(); }

    eventsetup::ComponentDescription const* componentDescription(unsigned int esModuleID) const {
      return doComponentDescription(esModuleID);
    }

    // The returned container indicates which EventSetup modules depend on which other
    // EventSetup modules. Each entry in the outer vector corresponds to an entry
    // in the vector returned by allESModules (same size and order).
    // The inner vector will contain an entry for each EventSetup module that can produce
    // a consumed data product. In the case of "may consumes", the producers for
    // all the products that might be consumed are included. The inner vector
    // includes producers for all produce methods together in the same vector and
    // has had duplicates removed.
    std::vector<std::vector<eventsetup::ComponentDescription const*>> const&
    esModulesWhoseProductsAreConsumedByESModule() const {
      return doESModulesWhoseProductsAreConsumedByESModule();
    }

    // Returns a container of information for one EventSetup module. The outer vector
    // is indexed by the produceMethodID, which is a counter of setWhatProduced calls in
    // the order they occur in the ESProducer constructor. The inner vector has
    // elements that correspond to the individual calls to the consumes function.
    // In an individual element, an empty module type indicates there is no
    // ESProducer or ESSource to produce the consumed data.
    // There is also a flag to indicate a module labeled mismatch.
    // "May consumes" elements are different because there can be multiple entries
    // for one "may consumes" function call, one for each available product that
    // matches the EventSetupRecordKey type and product type. If there are no
    // matches there is still one entry with a flag indicating there were
    // no matches.
    std::vector<std::vector<ESModuleConsumesInfo>> esModuleConsumesInfos(unsigned int esModuleID) const;

  private:
    virtual std::vector<std::string> const& doPaths() const = 0;
    virtual std::vector<std::string> const& doEndPaths() const = 0;
    virtual std::vector<ModuleDescription const*> const& doAllModules() const = 0;
    virtual ModuleDescription const* doModuleDescription(unsigned int moduleID) const = 0;
    virtual std::vector<ModuleDescription const*> const& doModulesOnPath(unsigned int pathIndex) const = 0;
    virtual std::vector<ModuleDescription const*> const& doModulesOnEndPath(unsigned int endPathIndex) const = 0;
    virtual std::vector<ModuleDescription const*> const& doModulesWhoseProductsAreConsumedBy(
        unsigned int moduleID, BranchType branchType) const = 0;
    virtual std::vector<eventsetup::ComponentDescription const*> const& doESModulesWhoseProductsAreConsumedBy(
        unsigned int moduleID, Transition) const = 0;
    virtual std::vector<ModuleConsumesInfo> doModuleConsumesInfos(unsigned int moduleID) const = 0;
    virtual std::vector<ModuleConsumesESInfo> doModuleConsumesESInfos(unsigned int moduleID) const = 0;
    virtual unsigned int doLargestModuleID() const = 0;
    virtual std::vector<eventsetup::ComponentDescription const*> const& doAllESModules() const = 0;
    virtual eventsetup::ComponentDescription const* doComponentDescription(unsigned int esModuleID) const = 0;
    virtual std::vector<std::vector<eventsetup::ComponentDescription const*>> const&
    doESModulesWhoseProductsAreConsumedByESModule() const = 0;
    virtual std::vector<std::vector<ESModuleConsumesInfo>> doESModuleConsumesInfos(unsigned int esModuleID) const = 0;
  };
}  // namespace edm
#endif

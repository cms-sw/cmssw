#ifndef FWCore_Framework_ModuleRegistryUtilities_h
#define FWCore_Framework_ModuleRegistryUtilities_h

#include <string>
#include <vector>
namespace edm {
  class ModuleRegistry;
  class ActivityRegistry;
  class ProductRegistry;
  namespace eventsetup {
    class ESRecordsToProductResolverIndices;
  }
  class ProcessBlockHelperBase;
  class ExceptionCollector;

  void finishModulesInitialization(ModuleRegistry& iModuleRegistry,
                                   ProductRegistry const& iProductRegistry,
                                   eventsetup::ESRecordsToProductResolverIndices const& iESIndices,
                                   ProcessBlockHelperBase const& processBlockHelperBase,
                                   std::string const& processName);
  /** beginJobCalledForModule tracks wich modules have had beginJob called on them in 
     * case there was an exception during the call to this function. The vector should be
     * passed to `runEndJobForModules`.
     * If an exception is thrown, it will be of type cms::Exception.
     */
  void runBeginJobForModules(ModuleRegistry& iModuleRegistry,
                             edm::ActivityRegistry& iActivityRegistry,
                             std::vector<bool>& beginJobCalledForModule) noexcept(false);

  /// The vector determines if the endJob of a module should be called. An empty vector means all modules
  /// should have their endJob called.
  void runEndJobForModules(ModuleRegistry& iModuleRegistry,
                           ActivityRegistry& iRegistry,
                           ExceptionCollector& collector,
                           std::vector<bool> const& beginJobCalledForModule,
                           const char* context) noexcept;

}  // namespace edm
#endif  // FWCore_Framework_ModuleRegistryUtilities_h
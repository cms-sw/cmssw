#ifndef FWCore_Framework_ModuleRegistryUtilities_h
#define FWCore_Framework_ModuleRegistryUtilities_h

#include <string>
#include <vector>
#include <mutex>
namespace edm {
  class ModuleRegistry;
  class ActivityRegistry;
  class ProductRegistry;
  class StreamContext;
  class GlobalContext;
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
  /** beginJobFailedForModule has the module id of each module which threw an exception during
     * the call to beginJob function.  The vector should be passed to `runEndJobForModules`.
     * If an exception is thrown, it will be of type cms::Exception.
     */
  void runBeginJobForModules(GlobalContext const& iGlobalContext,
                             ModuleRegistry& iModuleRegistry,
                             edm::ActivityRegistry& iActivityRegistry,
                             std::vector<unsigned int>& beginJobFailedForModule) noexcept(false);

  /// The vector holds module id for modules which should not have their endJob called.
  void runEndJobForModules(GlobalContext const& iGlobalContext,
                           ModuleRegistry& iModuleRegistry,
                           ActivityRegistry& iRegistry,
                           ExceptionCollector& collector,
                           std::vector<unsigned int> const& beginJobFailedForModule) noexcept;

  /** beginStreamFailedForModule holds module id for each module which threw an exception during
     * the call to beginStream function. This vector is used to determine which modules should not
     * have their endStream called. The vector should be passed to `runEndStreamForModules`.
     * If an exception is thrown, it will be of type cms::Exception.
     */
  void runBeginStreamForModules(StreamContext const& iStreamContext,
                                ModuleRegistry& iModuleRegistry,
                                edm::ActivityRegistry& iActivityRegistry,
                                std::vector<unsigned int>& beginStreamFailedForModule) noexcept(false);

  /// The vector hold module id for modules which should not have their endStream called.
  void runEndStreamForModules(StreamContext const& iStreamContext,
                              ModuleRegistry& iModuleRegistry,
                              ActivityRegistry& iRegistry,
                              ExceptionCollector& collector,
                              std::mutex& collectorMutex,
                              std::vector<unsigned int> const& beginStreamFailedForModule) noexcept;

}  // namespace edm
#endif  // FWCore_Framework_ModuleRegistryUtilities_h

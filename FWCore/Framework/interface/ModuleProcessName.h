#ifndef FWCore_Framework_ModuleProcessName_h
#define FWCore_Framework_ModuleProcessName_h

#include <string_view>

namespace edm {
  /**
   * Helper class to hold a module label and a process name
   *
   * Note: does NOT own the string storage, be careful to use.
   */
  class ModuleProcessName {
  public:
    explicit ModuleProcessName(std::string_view module, std::string_view process)
        : moduleLabel_{module}, processName_{process} {}

    std::string_view moduleLabel() const { return moduleLabel_; }
    std::string_view processName() const { return processName_; }

  private:
    std::string_view moduleLabel_;
    std::string_view processName_;
  };

  inline bool operator<(ModuleProcessName const& a, ModuleProcessName const& b) {
    return a.processName() == b.processName() ? a.moduleLabel() < b.moduleLabel() : a.processName() < b.processName();
  }
}  // namespace edm

#endif

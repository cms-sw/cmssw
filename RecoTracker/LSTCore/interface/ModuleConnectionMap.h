#ifndef RecoTracker_LSTCore_interface_ModuleConnectionMap_h
#define RecoTracker_LSTCore_interface_ModuleConnectionMap_h

#include <array>
#include <map>
#include <string>
#include <vector>

namespace lst {
  class ModuleConnectionMap {
  private:
    std::map<unsigned int, std::vector<unsigned int>> moduleConnections_;

  public:
    ModuleConnectionMap();
    ModuleConnectionMap(std::string const& filename);

    void load(std::string const&);
    void add(std::string const&);
    void print();

    const std::vector<unsigned int>& getConnectedModuleDetIds(unsigned int detid) const;
    int size() const;
  };

  using MapPLStoLayer = std::array<std::array<ModuleConnectionMap, 4>, 3>;
}  // namespace lst

#endif

#ifndef RecoTracker_LSTCore_interface_ModuleConnectionMap_h
#define RecoTracker_LSTCore_interface_ModuleConnectionMap_h

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>

namespace lst {
  class ModuleConnectionMap {
  private:
    std::map<unsigned int, std::vector<unsigned int>> moduleConnections_;

  public:
    ModuleConnectionMap();
    ModuleConnectionMap(std::string const& filename);
    ~ModuleConnectionMap();

    void load(std::string const&);
    void add(std::string const&);
    void print();

    const std::vector<unsigned int>& getConnectedModuleDetIds(unsigned int detid) const;
    int size() const;
  };

  using MapPLStoLayer = std::array<std::array<ModuleConnectionMap, 4>, 3>;
}  // namespace lst

#endif

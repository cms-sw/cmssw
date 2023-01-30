#ifndef HeterogeneousCore_ROCmServices_interface_ROCmService_h
#define HeterogeneousCore_ROCmServices_interface_ROCmService_h

#include <utility>
#include <vector>

#include "FWCore/Utilities/interface/StreamID.h"

namespace edm {
  class ParameterSet;
  class ActivityRegistry;
  class ConfigurationDescriptions;
}  // namespace edm

class ROCmService {
public:
  ROCmService(edm::ParameterSet const& config);
  ~ROCmService();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  bool enabled() const { return enabled_; }

  int numberOfDevices() const { return numberOfDevices_; }

  // major, minor
  std::pair<int, int> computeCapability(int device) const { return computeCapabilities_.at(device); }

  // Returns the id of device with most free memory. If none is found, returns -1.
  int deviceWithMostFreeMemory() const;

private:
  int numberOfDevices_ = 0;
  std::vector<std::pair<int, int>> computeCapabilities_;
  bool enabled_ = false;
  bool verbose_ = false;
};

namespace edm {
  namespace service {
    inline bool isProcessWideService(ROCmService const*) { return true; }
  }  // namespace service
}  // namespace edm

#endif  // HeterogeneousCore_ROCmServices_interface_ROCmService_h

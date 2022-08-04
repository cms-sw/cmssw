#ifndef HeterogeneousCore_AlpakaServices_interface_AlpakaService_h
#define HeterogeneousCore_AlpakaServices_interface_AlpakaService_h

#include <vector>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace edm {
  class ActivityRegistry;
  class ConfigurationDescriptions;
  class ParameterSet;
}  // namespace edm

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class AlpakaService {
  public:
    AlpakaService(edm::ParameterSet const& config, edm::ActivityRegistry&);
    ~AlpakaService() = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    bool enabled() const { return enabled_; }

    std::vector<Device> const& devices() const { return devices_; }

    Device const& device(uint32_t index) const { return devices_.at(index); }

  private:
    bool enabled_ = false;
    bool verbose_ = false;
    std::vector<Device> devices_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DECLARE_ALPAKA_TYPE_ALIAS(AlpakaService);

#endif  // HeterogeneousCore_AlpakaServices_interface_AlpakaService_h

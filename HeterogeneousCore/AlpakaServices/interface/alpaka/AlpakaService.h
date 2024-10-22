#ifndef HeterogeneousCore_AlpakaServices_interface_AlpakaService_h
#define HeterogeneousCore_AlpakaServices_interface_AlpakaService_h

#include <vector>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/AlpakaServiceFwd.h"

namespace edm {
  class ActivityRegistry;
  class ConfigurationDescriptions;
  class ParameterSet;
}  // namespace edm

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class AlpakaService {
  public:
    AlpakaService(edm::ParameterSet const& config, edm::ActivityRegistry&);
    ~AlpakaService();

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    bool enabled() const { return enabled_; }

  private:
    bool enabled_ = false;
    bool verbose_ = false;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DECLARE_ALPAKA_TYPE_ALIAS(AlpakaService);

#endif  // HeterogeneousCore_AlpakaServices_interface_AlpakaService_h

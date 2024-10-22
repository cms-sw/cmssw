#ifndef RecoLocalTracker_StripClusterizerAlgorithmFactory_h
#define RecoLocalTracker_StripClusterizerAlgorithmFactory_h

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
}  // namespace edm
class StripClusterizerAlgorithm;
#include <memory>

#include "FWCore/Framework/interface/ConsumesCollector.h"

class StripClusterizerAlgorithmFactory {
public:
  static std::unique_ptr<StripClusterizerAlgorithm> create(edm::ConsumesCollector&&, const edm::ParameterSet&);
  static void fillDescriptions(edm::ParameterSetDescription& clusterizer);
};
#endif

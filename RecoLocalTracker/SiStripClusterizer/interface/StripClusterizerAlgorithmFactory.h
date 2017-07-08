#ifndef RecoLocalTracker_StripClusterizerAlgorithmFactory_h
#define RecoLocalTracker_StripClusterizerAlgorithmFactory_h

namespace edm {class ParameterSet;}
class StripClusterizerAlgorithm;
#include <memory>

class StripClusterizerAlgorithmFactory {
 public:
  static std::unique_ptr<StripClusterizerAlgorithm> create(const edm::ParameterSet&);
};
#endif

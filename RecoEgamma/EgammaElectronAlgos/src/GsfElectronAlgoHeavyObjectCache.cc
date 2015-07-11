#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronAlgoHeavyObjectCache.h"

namespace gsfAlgoHelpers {
  HeavyObjectCache::HeavyObjectCache(const edm::ParameterSet& conf) {
    SoftElectronMVAEstimator::Configuration config;
    config.vweightsfiles = 
      conf.getParameter<std::vector<std::string>>("SoftElecMVAFilesString");
    sElectronMVAEstimator.reset(new SoftElectronMVAEstimator(config));
  }
}

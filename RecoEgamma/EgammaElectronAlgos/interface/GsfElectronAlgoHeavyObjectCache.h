#ifndef __RecoEgamma_GsfElectronAlgos_gsfAlgoHelpsHeavyObjectCache_h__
#define __RecoEgamma_GsfElectronAlgos_gsfAlgoHelpsHeavyObjectCache_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoEgamma/ElectronIdentification/interface/SoftElectronMVAEstimator.h"
#include <memory>

namespace gsfAlgoHelpers {
  class HeavyObjectCache {
  public:
    HeavyObjectCache(const edm::ParameterSet&);
    std::unique_ptr<const SoftElectronMVAEstimator> sElectronMVAEstimator;
  };
}

#endif

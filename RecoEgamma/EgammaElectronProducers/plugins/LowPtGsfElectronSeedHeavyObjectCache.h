#ifndef RecoEgamma_EgammaElectronProducers_LowPtGsfElectronSeedHeavyObjectCache_h
#define RecoEgamma_EgammaElectronProducers_LowPtGsfElectronSeedHeavyObjectCache_h

#include "CondFormats/GBRForest/interface/GBRForest.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include <vector>

namespace reco {
  class BeamSpot;
  class PreId;
}  // namespace reco

namespace lowptgsfeleseed {

  class HeavyObjectCache {
  public:
    HeavyObjectCache(const edm::ParameterSet&);

    std::vector<std::string> modelNames() const { return names_; }

    bool eval(const std::string& name,
              reco::PreId& ecal,
              reco::PreId& hcal,
              double rho,
              const reco::BeamSpot& spot,
              noZS::EcalClusterLazyTools& ecalTools) const;

  private:
    std::vector<std::string> names_;
    std::vector<std::unique_ptr<const GBRForest> > models_;
    std::vector<double> thresholds_;
  };
}  // namespace lowptgsfeleseed

#endif  // RecoEgamma_EgammaElectronProducers_LowPtGsfElectronSeedHeavyObjectCache_h

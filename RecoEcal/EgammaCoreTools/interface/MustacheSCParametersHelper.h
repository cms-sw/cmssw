#ifndef RecoEcal_EgammaCoreTools_MustacheSCParametersHelper_h
#define RecoEcal_EgammaCoreTools_MustacheSCParametersHelper_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/EcalObjects/interface/EcalMustacheSCParameters.h"

namespace reco {

  class MustacheSCParametersHelper {
  public:
    MustacheSCParametersHelper(EcalMustacheSCParameters &params, const edm::ParameterSet &iConfig);
    ~MustacheSCParametersHelper() = default;

    void setSqrtLogClustETuning(const float sqrtLogClustETuning);

    void addParabolaParameters(const EcalMustacheSCParameters::ParabolaParameters &parabolaParams);
    void sortParabolaParametersCollection();

  private:
    EcalMustacheSCParameters &parameters_;
  };

}  // namespace reco

#endif

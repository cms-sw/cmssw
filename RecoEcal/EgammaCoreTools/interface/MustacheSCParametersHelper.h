#ifndef RecoEcal_EgammaCoreTools_MustacheSCParametersHelper_h
#define RecoEcal_EgammaCoreTools_MustacheSCParametersHelper_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/EcalObjects/interface/EcalMustacheSCParameters.h"

namespace reco {

  class MustacheSCParametersHelper : public EcalMustacheSCParameters {
  public:
    MustacheSCParametersHelper(const EcalMustacheSCParameters &params);
    MustacheSCParametersHelper(const edm::ParameterSet &iConfig);
    ~MustacheSCParametersHelper() override{};

    float sqrtLogClustETuning() const;
    void setSqrtLogClustETuning(const float sqrtLogClustETuning);

    ParabolaParameters parabolaParameters(float log10ClustE, float absSeedEta) const;
    void addParabolaParameters(const EcalMustacheSCParameters::ParabolaParameters &params);
    void sortParabolaParametersCollection();
  };

}  // namespace reco

#endif

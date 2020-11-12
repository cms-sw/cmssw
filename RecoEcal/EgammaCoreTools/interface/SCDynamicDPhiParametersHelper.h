#ifndef RecoEcal_EgammaCoreTools_SCDynamicDPhiParametersHelper_h
#define RecoEcal_EgammaCoreTools_SCDynamicDPhiParametersHelper_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/EcalObjects/interface/EcalSCDynamicDPhiParameters.h"

namespace reco {

  class SCDynamicDPhiParametersHelper : public EcalSCDynamicDPhiParameters {
  public:
    SCDynamicDPhiParametersHelper(const EcalSCDynamicDPhiParameters &params);
    SCDynamicDPhiParametersHelper(const edm::ParameterSet &iConfig);
    ~SCDynamicDPhiParametersHelper() override{};

    DynamicDPhiParameters dynamicDPhiParameters(double clustE, double absSeedEta) const;
    void addDynamicDPhiParameters(const EcalSCDynamicDPhiParameters::DynamicDPhiParameters &params);
    void sortDynamicDPhiParametersCollection();
  };

}  // namespace reco

#endif

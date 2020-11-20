#ifndef RecoEcal_EgammaCoreTools_SCDynamicDPhiParametersHelper_h
#define RecoEcal_EgammaCoreTools_SCDynamicDPhiParametersHelper_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/EcalObjects/interface/EcalSCDynamicDPhiParameters.h"

namespace reco {

  class SCDynamicDPhiParametersHelper {
  public:
    SCDynamicDPhiParametersHelper(EcalSCDynamicDPhiParameters &params);
    SCDynamicDPhiParametersHelper(EcalSCDynamicDPhiParameters &params, const edm::ParameterSet &iConfig);
    ~SCDynamicDPhiParametersHelper() = default;

    void addDynamicDPhiParameters(const EcalSCDynamicDPhiParameters::DynamicDPhiParameters &params);
    void sortDynamicDPhiParametersCollection();

  private:
    EcalSCDynamicDPhiParameters &parameters_;
  };

}  // namespace reco

#endif

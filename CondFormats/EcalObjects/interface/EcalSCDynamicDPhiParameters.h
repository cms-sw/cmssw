#ifndef CondFormats_EcalObjects_EcalSCDynamicDPhiParameters_h
#define CondFormats_EcalObjects_EcalSCDynamicDPhiParameters_h

#include <iostream>
#include <vector>

#include "CondFormats/Serialization/interface/Serializable.h"

namespace reco {
  class SCDynamicDPhiParametersHelper;
}  // namespace reco

class EcalSCDynamicDPhiParameters {
public:
  EcalSCDynamicDPhiParameters(){};
  virtual ~EcalSCDynamicDPhiParameters() = default;

  struct DynamicDPhiParameters {
    double eMin;
    double etaMin;
    double yoffset;
    double scale;
    double xoffset;
    double width;
    double saturation;
    double cutoff;

    COND_SERIALIZABLE;
  };

  const DynamicDPhiParameters* dynamicDPhiParameters(double clustE, double absSeedEta) const;

  // helper class to set parameters
  friend class reco::SCDynamicDPhiParametersHelper;

  // print parameters to stream:
  void print(std::ostream&) const;
  friend std::ostream& operator<<(std::ostream& out, const EcalSCDynamicDPhiParameters& params) {
    params.print(out);
    return out;
  }

protected:
  // collection is expected to be sorted in ascending DynamicDPhiParameters.eMin and ascending DynamicDPhiParameters.etaMax
  std::vector<DynamicDPhiParameters> dynamicDPhiParametersCollection_;

  COND_SERIALIZABLE;
};

#endif

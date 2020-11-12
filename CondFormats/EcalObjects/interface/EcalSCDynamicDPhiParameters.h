#ifndef CondFormats_EcalObjects_EcalSCDynamicDPhiParameters_h
#define CondFormats_EcalObjects_EcalSCDynamicDPhiParameters_h

#include <iostream>
#include <vector>

#include "CondFormats/Serialization/interface/Serializable.h"

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

  // print parameters to stream:
  void print(std::ostream&) const;
  friend std::ostream& operator<<(std::ostream& out, const EcalSCDynamicDPhiParameters& params) {
    params.print(out);
    return out;
  }

protected:
  // collection is expected to be sorted in descending DynamicDPhiParameters.etaMax and ascending DynamicDPhiParameters.minEt
  std::vector<DynamicDPhiParameters> dynamicDPhiParametersCollection_;

  COND_SERIALIZABLE;
};

#endif

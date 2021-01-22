#ifndef CondFormats_EcalObjects_EcalMustacheSCParameters_h
#define CondFormats_EcalObjects_EcalMustacheSCParameters_h

#include <iostream>
#include <vector>

#include "CondFormats/Serialization/interface/Serializable.h"

namespace reco {
  class MustacheSCParametersHelper;
}  // namespace reco

class EcalMustacheSCParameters {
public:
  EcalMustacheSCParameters(){};
  virtual ~EcalMustacheSCParameters() = default;

  struct ParabolaParameters {
    double log10EMin;
    double etaMin;
    std::vector<double> pUp;
    std::vector<double> pLow;
    std::vector<double> w0Up;
    std::vector<double> w1Up;
    std::vector<double> w0Low;
    std::vector<double> w1Low;

    COND_SERIALIZABLE;
  };

  float sqrtLogClustETuning() const;

  const ParabolaParameters* parabolaParameters(float log10ClustE, float absSeedEta) const;

  // helper class to set parameters
  friend class reco::MustacheSCParametersHelper;

  // print parameters to stream:
  void print(std::ostream&) const;
  friend std::ostream& operator<<(std::ostream& out, const EcalMustacheSCParameters& params) {
    params.print(out);
    return out;
  }

protected:
  float sqrtLogClustETuning_;

  // collection is expected to be sorted in ascending ParabolaParameters.log10EMin and ascending ParabolaParameters.etaMax
  std::vector<ParabolaParameters> parabolaParametersCollection_;

  COND_SERIALIZABLE;
};

#endif

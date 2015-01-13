#include "CondFormats/EcalObjects/interface/EcalPulseCovariances.h"

EcalPulseCovariance::EcalPulseCovariance() {
  for(int k=0; k<std::pow(EcalPulseShape::TEMPLATESAMPLES,2); ++k) {
    int i = k/EcalPulseShape::TEMPLATESAMPLES;
    int j = k%EcalPulseShape::TEMPLATESAMPLES;
    covval[i][j] = 0.;
  }
}

#include "RecoEgamma/EgammaTools/interface/ShowerDepth.h"
#include <cmath>

using namespace hgcal;

float ShowerDepth::getClusterDepthCompatibility(float measuredDepth,
                                                float emEnergy,
                                                float& expectedDepth,
                                                float& expectedSigma) const {
  float lny = (emEnergy > criticalEnergy_) ? std::log(emEnergy / criticalEnergy_) : 0.;

  // inject here parametrization results
  float meantmax = meant0_ + meant1_ * lny;
  float meanalpha = meanalpha0_ + meanalpha1_ * lny;
  if (meanalpha < 1.)
    meanalpha = 1.1;  // no poles
  float sigmalntmax = 1. / (sigmalnt0_ + sigmalnt1_ * lny);
  float sigmalnalpha = 1. / (sigmalnalpha0_ + sigmalnalpha1_ * lny);
  float corrlnalphalntmax = corrlnalphalnt0_ + corrlnalphalnt1_ * lny;

  float invbeta = meantmax / (meanalpha - 1.);
  float predictedDepth = meanalpha * invbeta;
  predictedDepth *= radiationLength_;

  float predictedSigma = sigmalnalpha * sigmalnalpha / ((meanalpha - 1.) * (meanalpha - 1.));
  predictedSigma += sigmalntmax * sigmalntmax;
  predictedSigma -= 2 * sigmalnalpha * sigmalntmax * corrlnalphalntmax / (meanalpha - 1.);
  if (predictedSigma < 0.)
    predictedSigma = 1.e10;  // say we can't predict anything
  predictedSigma = predictedDepth * std::sqrt(predictedSigma);

  expectedDepth = predictedDepth;
  expectedSigma = predictedSigma;
  return (measuredDepth - predictedDepth) / predictedSigma;
}

#include "RecoEgamma/EgammaTools/interface/ShowerDepth.h"
#include <cmath>

using namespace hgcal;

float ShowerDepth::getClusterDepthCompatibility(float measuredDepth, float emEnergy, float& expectedDepth, float& expectedSigma) const {
    double lny = (emEnergy > criticalEnergy_) ? std::log(emEnergy/criticalEnergy_) : 0.;

    // inject here parametrization results
    double meantmax = meant0_ + meant1_*lny;
    double meanalpha = meanalpha0_ + meanalpha1_*lny;
    if ( meanalpha < 1. ) meanalpha = 1.1; // no poles
    double sigmalntmax = 1. / (sigmalnt0_+sigmalnt1_*lny);
    double sigmalnalpha = 1. / (sigmalnalpha0_+sigmalnalpha1_*lny);
    double corrlnalphalntmax = corrlnalphalnt0_+corrlnalphalnt1_*lny;

    double invbeta = meantmax/(meanalpha-1.);
    double predictedDepth = meanalpha*invbeta;
    predictedDepth *= radiationLength_;

    double predictedSigma = sigmalnalpha*sigmalnalpha/((meanalpha-1.)*(meanalpha-1.));
    predictedSigma += sigmalntmax*sigmalntmax;
    predictedSigma -= 2*sigmalnalpha*sigmalntmax*corrlnalphalntmax/(meanalpha-1.);
    if ( predictedSigma < 0. ) predictedSigma = 1.e10; // say we can't predict anything
    predictedSigma = predictedDepth*std::sqrt(predictedSigma);

    expectedDepth = (float) predictedDepth;
    expectedSigma = (float) predictedSigma;
    return (float) (measuredDepth-predictedDepth)/predictedSigma;
}

#include "RecoEgamma/EgammaTools/interface/ShowerDepth.h"
#include <ctgmath>
#include <iostream>

ShowerDepth::ShowerDepth() {
    // HGCAL average medium
    criticalEnergy_ = 0.00536; // in GeV
    radiationLength_ = 0.968; // in cm

    // longitudinal parameters
    // mean values
    // shower max <T> = t0 + t1*lny
    // <alpha> = alpha0 + alpha1*lny
    // shower average = alpha/beta
    meant0_ = -1.396;
    meant1_ = 1.007;
    meanalpha0_ = -0.0433;
    meanalpha1_ = 0.540;
    // sigmas
    // sigma(lnT) = 1 /sigmalnt0 + sigmalnt1*lny;
    // sigma(lnalpha) = 1 /sigmalnt0 + sigmalnt1*lny;
    sigmalnt0_ = -2.506;
    sigmalnt1_ = 1.245;
    sigmalnalpha0_ = -0.08442;
    sigmalnalpha1_ = 0.7904;
    // corr(lnalpha,lnt) = corrlnalpha0_+corrlnalphalnt1_*y
    corrlnalphalnt0_ = 0.7858;
    corrlnalphalnt1_ = -0.0232;
    debug_ = false;
}

float ShowerDepth::getClusterDepthCompatibility(float length, float emEnergy,float & expectedDepth, float & expectedSigma) const {
    float lny = emEnergy/criticalEnergy_>1. ? std::log(emEnergy/criticalEnergy_) : 0.;
    // inject here parametrization results
    float meantmax = meant0_ + meant1_*lny;
    float meanalpha = meanalpha0_ + meanalpha1_*lny;
    float sigmalntmax = 1.0 / (sigmalnt0_+sigmalnt1_*lny);
    float sigmalnalpha = 1.0 / (sigmalnalpha0_+sigmalnalpha1_*lny);
    float corrlnalphalntmax = corrlnalphalnt0_+corrlnalphalnt1_*lny;

    float invbeta = meantmax/(meanalpha-1.);
    float predictedLength = meanalpha*invbeta;
    predictedLength *= radiationLength_;

    float sigmaalpha = meanalpha*sigmalnalpha;
    if (sigmaalpha<0.) sigmaalpha = 1.;
    float sigmatmax = meantmax*sigmalntmax;
    if (sigmatmax<0.) sigmatmax = 1.;

    float predictedSigma = sigmalnalpha*sigmalnalpha/((meanalpha-1.)*(meanalpha-1.));
    predictedSigma += sigmalntmax*sigmalntmax;
    predictedSigma -= 2*sigmalnalpha*sigmalntmax*corrlnalphalntmax/(meanalpha-1.);
    predictedSigma = predictedLength*std::sqrt(predictedSigma);
    if (debug_){
            std::cout  << " Predicted length " << predictedLength << " Predicted Sigma " << predictedSigma << std::endl;
        }
    expectedDepth = predictedLength;
    expectedSigma = predictedSigma;
    return (length-predictedLength)/predictedSigma;
}

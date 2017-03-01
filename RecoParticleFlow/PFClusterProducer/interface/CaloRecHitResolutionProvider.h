#ifndef RecoParticleFlow_PFClusterProducer_CaloRecHitResolutionProvider_h
#define RecoParticleFlow_PFClusterProducer_CaloRecHitResolutionProvider_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"


class CaloRecHitResolutionProvider{
 public:
  CaloRecHitResolutionProvider(const edm::ParameterSet& iConfig) {
    noiseTerm_ = iConfig.getParameter<double>("noiseTerm"); 
    constantTerm2_ = std::pow(iConfig.getParameter<double>("constantTerm"), 2); 
    noiseTermLowE_ = iConfig.getParameter<double>("noiseTermLowE"); 
    corrTermLowE_ = iConfig.getParameter<double>("corrTermLowE"); 
    constantTermLowE2_ = std::pow(iConfig.getParameter<double>("constantTermLowE"), 2); 
    threshLowE_ = iConfig.getParameter<double>("threshLowE"); 
    threshHighE_ = iConfig.getParameter<double>("threshHighE"); 

    resHighE2_ = std::pow((noiseTerm_/threshHighE_), 2) + constantTerm2_;
  }

  double timeResolution2(double energy)
  {
    double res2 = 10000.;

    if (energy <= 0.)
      return res2;
    else if (energy < threshLowE_)
    {
      if (corrTermLowE_ > 0.) {// different parametrisation
        const double res = noiseTermLowE_/energy + corrTermLowE_/(energy*energy);
        res2 = res*res;
      }
      else {
        const double noiseDivE = noiseTermLowE_/energy;
        res2 = noiseDivE*noiseDivE + constantTermLowE2_;
      }
    }
    else if (energy < threshHighE_) {
      const double noiseDivE = noiseTerm_/energy;
      res2 = noiseDivE*noiseDivE + constantTerm2_;
    }
    else // if (energy >=threshHighE_)
      res2 = resHighE2_;

    if (res2 > 10000.)
      return 10000.;
    return res2;
    
  }
private:

  double noiseTerm_; // Noise term
  double constantTerm2_; // Constant term

  double noiseTermLowE_; // Noise term for low E
  double constantTermLowE2_; // Constant term for low E
  double corrTermLowE_; // 2nd term for low E, different parametrisation

  double threshLowE_; // different parametrisation below
  double threshHighE_; // resolution constant above

  double resHighE2_; // precompute res at high E

};

#endif



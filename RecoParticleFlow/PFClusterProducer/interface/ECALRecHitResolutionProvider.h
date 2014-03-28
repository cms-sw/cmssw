#ifndef RecoParticleFlow_PFClusterProducer_ECALRecHitResolutionProvider_h
#define RecoParticleFlow_PFClusterProducer_ECALRecHitResolutionProvider_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"


class ECALRecHitResolutionProvider{
 public:
  ECALRecHitResolutionProvider(const edm::ParameterSet& iConfig) {
    noiseTerm_ = iConfig.getParameter<double>("noiseTerm"); 
    constantTerm_ = iConfig.getParameter<double>("constantTerm"); 
    noiseTermLowE_ = iConfig.getParameter<double>("noiseTermLowE"); 
    corrTermLowE_ = iConfig.getParameter<double>("corrTermLowE"); 
    constantTermLowE_ = iConfig.getParameter<double>("constantTermLowE"); 
    threshLowE_ = iConfig.getParameter<double>("threshLowE"); 
    threshHighE_ = iConfig.getParameter<double>("threshHighE"); 

    resHighE_ = sqrt(std::pow((noiseTerm_/threshHighE_), 2) + constantTerm_*constantTerm_);
  }


double timeResolution(double energy)
{
  double res = 100.;

  if (energy <= 0.)
    return res;
  else if (energy < threshLowE_)
  {
    if (corrTermLowE_ > 0.) // different parametrisation
      res = noiseTermLowE_/energy + corrTermLowE_/energy/energy;
    else
      res = sqrt(std::pow(noiseTermLowE_/energy, 2) + constantTermLowE_*constantTermLowE_);
  }
  else if (energy < threshHighE_)
    res = sqrt(std::pow((noiseTerm_/energy), 2) + constantTerm_*constantTerm_);
  else // if (energy >=threshHighE_)
    res = resHighE_;

  if (res > 100.)
    return 100.;
  return res;
  
}

private:

  double noiseTerm_; // Noise term
  double constantTerm_; // Constant term

  double noiseTermLowE_; // Noise term for low E
  double constantTermLowE_; // Constant term for low E
  double corrTermLowE_; // 2nd term for low E, different parametrisation

  double threshLowE_; // different parametrisation below
  double threshHighE_; // resolution constant above

  double resHighE_; // precompute res at high E

};

#endif



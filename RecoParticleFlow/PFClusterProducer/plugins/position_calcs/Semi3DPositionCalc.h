#ifndef __Semi3DPositionCalc_H__
#define __Semi3DPositionCalc_H__

#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "RecoParticleFlow/PFClusterProducer/interface/ECALRecHitResolutionProvider.h"

class Semi3DPositionCalc : public PFCPositionCalculatorBase {
 public:
  Semi3DPositionCalc(const edm::ParameterSet& conf) :
    PFCPositionCalculatorBase(conf),    
    _posCalcNCrystals(conf.getParameter<int>("posCalcNCrystals")),
    _logWeightDenom(conf.getParameter<double>("logWeightDenominator")),
    _minAllowedNorm(conf.getParameter<double>("minAllowedNormalization"))

  {  
    _timeResolutionCalcBarrel.reset(NULL);
    if( conf.exists("timeResolutionCalcBarrel") ) {
      const edm::ParameterSet& timeResConf = 
        conf.getParameterSet("timeResolutionCalcBarrel");
        _timeResolutionCalcBarrel.reset(new ECALRecHitResolutionProvider(timeResConf));
    }
    _timeResolutionCalcEndcap.reset(NULL);
    if( conf.exists("timeResolutionCalcEndcap") ) {
      const edm::ParameterSet& timeResConf = 
        conf.getParameterSet("timeResolutionCalcEndcap");
        _timeResolutionCalcEndcap.reset(new ECALRecHitResolutionProvider(timeResConf));
    }
  }
  Semi3DPositionCalc(const Semi3DPositionCalc&) = delete;
  Semi3DPositionCalc& operator=(const Semi3DPositionCalc&) = delete;

  void calculateAndSetPosition(reco::PFCluster&);
  void calculateAndSetPositions(reco::PFClusterCollection&);

 private:
  const int _posCalcNCrystals;
  const double _logWeightDenom;
  const double _minAllowedNorm;
  
  std::unique_ptr<ECALRecHitResolutionProvider> _timeResolutionCalcBarrel;
  std::unique_ptr<ECALRecHitResolutionProvider> _timeResolutionCalcEndcap;

  void calculateAndSetPositionActual(reco::PFCluster&) const;
};

DEFINE_EDM_PLUGIN(PFCPositionCalculatorFactory,
		  Semi3DPositionCalc,
		  "Semi3DPositionCalc");

#endif

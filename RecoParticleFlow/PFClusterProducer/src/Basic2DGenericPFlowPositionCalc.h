#ifndef __Basic2DGenericPFlowPositionCalc_H__
#define __Basic2DGenericPFlowPositionCalc_H__

#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "RecoParticleFlow/PFClusterProducer/interface/ECALRecHitResolutionProvider.h"

class Basic2DGenericPFlowPositionCalc : public PFCPositionCalculatorBase {
 public:
  Basic2DGenericPFlowPositionCalc(const edm::ParameterSet& conf) :
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
  Basic2DGenericPFlowPositionCalc(const Basic2DGenericPFlowPositionCalc&) = delete;
  Basic2DGenericPFlowPositionCalc& operator=(const Basic2DGenericPFlowPositionCalc&) = delete;

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
		  Basic2DGenericPFlowPositionCalc,
		  "Basic2DGenericPFlowPositionCalc");

#endif

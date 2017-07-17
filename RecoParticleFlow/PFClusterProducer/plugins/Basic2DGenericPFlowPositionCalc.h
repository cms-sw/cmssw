#ifndef __Basic2DGenericPFlowPositionCalc_H__
#define __Basic2DGenericPFlowPositionCalc_H__

#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoParticleFlow/PFClusterProducer/interface/CaloRecHitResolutionProvider.h"

class Basic2DGenericPFlowPositionCalc : public PFCPositionCalculatorBase {
 public:
  Basic2DGenericPFlowPositionCalc(const edm::ParameterSet& conf) :
    PFCPositionCalculatorBase(conf),    
    _posCalcNCrystals(conf.getParameter<int>("posCalcNCrystals")),
    _logWeightDenom(1./conf.getParameter<double>("logWeightDenominator")),
    _minAllowedNorm(conf.getParameter<double>("minAllowedNormalization"))

  {  
    _timeResolutionCalcBarrel.reset(nullptr);
    if( conf.exists("timeResolutionCalcBarrel") ) {
      const edm::ParameterSet& timeResConf = 
        conf.getParameterSet("timeResolutionCalcBarrel");
        _timeResolutionCalcBarrel.reset(new CaloRecHitResolutionProvider(timeResConf));
    }
    _timeResolutionCalcEndcap.reset(nullptr);
    if( conf.exists("timeResolutionCalcEndcap") ) {
      const edm::ParameterSet& timeResConf = 
        conf.getParameterSet("timeResolutionCalcEndcap");
        _timeResolutionCalcEndcap.reset(new CaloRecHitResolutionProvider(timeResConf));
    }

   switch( _posCalcNCrystals ) {
    case 5:
    case 9:
    case -1:
      break;
    default:
      edm::LogError("Basic2DGenericPFlowPositionCalc") << "posCalcNCrystals not valid";
      assert(0); // bug
   }


  }

  Basic2DGenericPFlowPositionCalc(const Basic2DGenericPFlowPositionCalc&) = delete;
  Basic2DGenericPFlowPositionCalc& operator=(const Basic2DGenericPFlowPositionCalc&) = delete;

  void calculateAndSetPosition(reco::PFCluster&) override;
  void calculateAndSetPositions(reco::PFClusterCollection&) override;

 private:
  const int _posCalcNCrystals;
  const float _logWeightDenom;
  const float _minAllowedNorm;
  
  std::unique_ptr<CaloRecHitResolutionProvider> _timeResolutionCalcBarrel;
  std::unique_ptr<CaloRecHitResolutionProvider> _timeResolutionCalcEndcap;

  void calculateAndSetPositionActual(reco::PFCluster&) const;
};

DEFINE_EDM_PLUGIN(PFCPositionCalculatorFactory,
		  Basic2DGenericPFlowPositionCalc,
		  "Basic2DGenericPFlowPositionCalc");

#endif

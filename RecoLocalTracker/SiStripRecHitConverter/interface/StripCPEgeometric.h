#ifndef RecoLocalTracker_SiStripRecHitConverter_StripCPEgeometric_H
#define RecoLocalTracker_SiStripRecHitConverter_StripCPEgeometric_H

#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"

class StripCPEgeometric : public StripCPE 
{

 public:
  
  StripClusterParameterEstimator::LocalValues 
    localParameters( const SiStripCluster&, const LocalTrajectoryParameters&) const; 

  StripClusterParameterEstimator::LocalValues 
    localParameters( const SiStripCluster&, const GeomDetUnit&, const LocalTrajectoryParameters&) const;

  StripCPEgeometric( edm::ParameterSet& conf, 
		     const MagneticField* mag, 
		     const TrackerGeometry* geom, 
		     const SiStripLorentzAngle* LorentzAngle)
    : StripCPE(conf, mag, geom, LorentzAngle ) {}

 private:

  typedef vector<uint8_t>::const_iterator chargeIt_t;
  std::pair<float,float> position_sigma_inStrips( uint16_t, chargeIt_t, chargeIt_t, float,SiStripDetId::SubDetector) const;
  bool useNMinusOne(chargeIt_t, chargeIt_t,float, SiStripDetId::SubDetector) const;
  bool hasMultiPeak(chargeIt_t, chargeIt_t) const;

  static const float TOBeta, TIBeta, invsqrt12, tandriftangle;
  static const unsigned crossoverRate;
};

const unsigned StripCPEgeometric::crossoverRate=15;
const float StripCPEgeometric::TOBeta=0.835;
const float StripCPEgeometric::TIBeta=0.890;
const float StripCPEgeometric::invsqrt12=1/sqrt(12);
const float StripCPEgeometric::tandriftangle=0.01;

#endif

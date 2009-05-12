#ifndef RecoLocalTracker_SiStripRecHitConverter_StripCPEgeometric_H
#define RecoLocalTracker_SiStripRecHitConverter_StripCPEgeometric_H

#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/ClusterParameterEstimator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"

class StripCPEgeometric : public StripCPE 
{
 public:
  
  StripCPEgeometric(edm::ParameterSet & conf, 
			 const MagneticField * mag, 
			 const TrackerGeometry* geom, 
			 const SiStripLorentzAngle* LorentzAngle)
    : StripCPE(conf,mag, geom, LorentzAngle ) {}

  StripClusterParameterEstimator::LocalValues localParameters( const SiStripCluster & cl,
							       const GeomDetUnit& det, 
							       const LocalTrajectoryParameters & ltp) const{
    return localParameters(cl,ltp);
  };
  StripClusterParameterEstimator::LocalValues localParameters( const SiStripCluster & cl, const LocalTrajectoryParameters & ltp) const; 


 private:

  typedef vector<uint8_t>::const_iterator chargeIt_t;
  std::pair<float,float> position_sigma_inStrips( uint16_t, chargeIt_t, chargeIt_t, float) const;
  bool useNMinusOne(chargeIt_t, chargeIt_t,float) const;
  bool hasMultiPeak(chargeIt_t, chargeIt_t) const;


  static const float width3threshold, invsqrt12, minProj;
  static const unsigned crossoverRate;
};

const unsigned StripCPEgeometric::crossoverRate=15;
const float StripCPEgeometric::width3threshold=0.15;
const float StripCPEgeometric::invsqrt12=1/sqrt(12);
const float StripCPEgeometric::minProj=0.08;

#endif

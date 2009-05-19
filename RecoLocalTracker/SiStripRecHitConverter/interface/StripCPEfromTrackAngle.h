#ifndef RecoLocalTracker_SiStripRecHitConverter_StripCPEfromTrackAngle_H
#define RecoLocalTracker_SiStripRecHitConverter_StripCPEfromTrackAngle_H

#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"

class StripCPEfromTrackAngle : public StripCPE 
{

 public:
  
  StripClusterParameterEstimator::LocalValues 
    localParameters( const SiStripCluster&, const LocalTrajectoryParameters&) const; 

  StripClusterParameterEstimator::LocalValues 
    localParameters( const SiStripCluster&, const GeomDetUnit&, const LocalTrajectoryParameters&) const;

  StripCPEfromTrackAngle( edm::ParameterSet& conf, 
			  const MagneticField* mag, 
			  const TrackerGeometry* geom, 
			  const SiStripLorentzAngle* LorentzAngle)
    : StripCPE(conf, mag, geom, LorentzAngle ) {}
  
 private:
  
  float stripErrorSquared(const unsigned&, const float&) const;
  
};
#endif

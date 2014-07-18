#ifndef RecoLocalTracker_SiStripRecHitConverter_StripCPEfromTrackAngle_H
#define RecoLocalTracker_SiStripRecHitConverter_StripCPEfromTrackAngle_H

#include <map>
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"

class StripCPEfromTrackAngle : public StripCPE 
{

 public:
  using StripCPE::localParameters;
  
  //Error parameterization, low cluster width function
  const float LC_P0;
  const float LC_P1;
  const float LC_P2;

  //High cluster width is broken down by sub-det
  std::map<SiStripDetId::SubDetector, float> HC_P0;
  std::map<SiStripDetId::SubDetector, float> HC_P1;
  
  StripClusterParameterEstimator::LocalValues
  localParameters( const SiStripCluster&, const GeomDetUnit&, const LocalTrajectoryParameters&) const;
  
  float stripErrorSquared(const unsigned N, const float uProj, const SiStripDetId::SubDetector loc ) const ;

  StripCPEfromTrackAngle( edm::ParameterSet & conf, 
			  const MagneticField& mag, 
			  const TrackerGeometry& geom, 
			  const SiStripLorentzAngle& lorentz,
                          const SiStripBackPlaneCorrection& backPlaneCorrection,
			  const SiStripConfObject& confObj,
			  const SiStripLatency& latency) 
  : StripCPE(conf, mag, geom, lorentz, backPlaneCorrection, confObj, latency ),
    LC_P0 (conf.getParameter<double>("LC_P0" )),
    LC_P1 (conf.getParameter<double>("LC_P1" )),
    LC_P2 (conf.getParameter<double>("LC_P2" ))
  {
    HC_P0.emplace(SiStripDetId::TIB,conf.getParameter<double>("TIB_P0"));
    HC_P0.emplace(SiStripDetId::TOB,conf.getParameter<double>("TOB_P0"));
    HC_P0.emplace(SiStripDetId::TID,conf.getParameter<double>("TID_P0"));
    HC_P0.emplace(SiStripDetId::TEC,conf.getParameter<double>("TEC_P0"));
    HC_P1.emplace(SiStripDetId::TIB,conf.getParameter<double>("TIB_P1"));
    HC_P1.emplace(SiStripDetId::TOB,conf.getParameter<double>("TOB_P1"));
    HC_P1.emplace(SiStripDetId::TID,conf.getParameter<double>("TID_P1"));
    HC_P1.emplace(SiStripDetId::TEC,conf.getParameter<double>("TEC_P1"));
  }
};
#endif

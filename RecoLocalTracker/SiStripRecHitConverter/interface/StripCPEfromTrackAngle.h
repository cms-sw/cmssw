#ifndef RecoLocalTracker_SiStripRecHitConverter_StripCPEfromTrackAngle_H
#define RecoLocalTracker_SiStripRecHitConverter_StripCPEfromTrackAngle_H

#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"

class StripCPEfromTrackAngle : public StripCPE 
{

 public:
  using StripCPE::localParameters;
  
  //Error parameterization, low cluster width function
  const double LC_P0;
  const double LC_P1;
  const double LC_P2;

  //High cluster width is broken down by sub-det
  const double TIB_P0;
  const double TIB_P1;
  const double TOB_P0;
  const double TOB_P1;
  const double TID_P0;
  const double TID_P1;
  const double TEC_P0;
  const double TEC_P1;
  
  StripClusterParameterEstimator::LocalValues
  localParameters( const SiStripCluster&, const GeomDetUnit&, const LocalTrajectoryParameters&) const;
  
  float stripErrorSquared(const unsigned N, const float uProj, const int& loc ) const ;

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
    LC_P2 (conf.getParameter<double>("LC_P2" )),
    TIB_P0(conf.getParameter<double>("TIB_P0")),
    TIB_P1(conf.getParameter<double>("TIB_P1")),
    TOB_P0(conf.getParameter<double>("TOB_P0")),
    TOB_P1(conf.getParameter<double>("TOB_P1")),
    TID_P0(conf.getParameter<double>("TID_P0")),
    TID_P1(conf.getParameter<double>("TID_P1")),
    TEC_P0(conf.getParameter<double>("TEC_P0")),
    TEC_P1(conf.getParameter<double>("TEC_P1"))
    {}  
};
#endif

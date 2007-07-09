#ifndef RecoLocalTracker_SiStripRecHitConverter_StripCPEfromTrackAngle_H
#define RecoLocalTracker_SiStripRecHitConverter_StripCPEfromTrackAngle_H

#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/ClusterParameterEstimator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"

class StripCPEfromTrackAngle : public StripCPE 
{
 public:
  
  StripCPEfromTrackAngle(edm::ParameterSet & conf, 
			 const MagneticField * mag, 
			 const TrackerGeometry* geom, 
			 const SiStripLorentzAngle* LorentzAngle)
    :StripCPE(conf,mag, geom, LorentzAngle ){}


  StripCPEfromTrackAngle(edm::ParameterSet & conf, 
			 const MagneticField * mag, 
			 const TrackerGeometry* geom)
    :StripCPE(conf,mag, geom){}

  // LocalValues is typedef for pair<LocalPoint,LocalError> 

  StripClusterParameterEstimator::LocalValues localParameters( const SiStripCluster & cl,
							       const GeomDetUnit& det, 
							       const LocalTrajectoryParameters & ltp) const{
    return localParameters(cl,ltp);
  }; 

  StripClusterParameterEstimator::LocalValues localParameters( const SiStripCluster & cl, const LocalTrajectoryParameters & ltp) const; 



};

#endif

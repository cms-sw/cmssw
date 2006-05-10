#ifndef RecoLocalTracker_SiStripRecHitConverter_StripCPE_H
#define RecoLocalTracker_SiStripRecHitConverter_StripCPE_H

#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/ClusterParameterEstimator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

class StripCPE : public StripClusterParameterEstimator 
{
 public:
  
  StripCPE(edm::ParameterSet & conf, const MagneticField * mag, const TrackerGeometry* geom);
    
  // LocalValues is typedef for pair<LocalPoint,LocalError> 
  StripClusterParameterEstimator::LocalValues localParameters( const SiStripCluster & cl,const GeomDetUnit& det) const {
    return localParameters(cl);
  }; 
  StripClusterParameterEstimator::LocalValues localParameters( const SiStripCluster & cl)const; 
  

  
  LocalVector driftDirection(const StripGeomDetUnit* det)const;

 protected:  
  const TrackerGeometry * geom_;
  const MagneticField * magfield_ ;
  float theTanLorentzAnglePerTesla_;
  bool useMagneticField_;
};

#endif





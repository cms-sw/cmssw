#ifndef RecoLocalTracker_SiStripRecHitConverter_StripCPE_H
#define RecoLocalTracker_SiStripRecHitConverter_StripCPE_H

#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/ClusterParameterEstimator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"

class StripCPE : public StripClusterParameterEstimator 
{
 public:
  
  StripCPE(edm::ParameterSet & conf, const MagneticField * mag, const TrackerGeometry* geom);

  StripCPE(edm::ParameterSet & conf, const MagneticField * mag, const TrackerGeometry* geom, const SiStripLorentzAngle* LorentzAngle);
    
  // LocalValues is typedef for pair<LocalPoint,LocalError> 
  StripClusterParameterEstimator::LocalValues localParameters( const SiStripCluster & cl,const GeomDetUnit& det) const {
    return localParameters(cl);
  }; 
  StripClusterParameterEstimator::LocalValues localParameters( const SiStripCluster & cl)const; 
  

  
  LocalVector driftDirection(const StripGeomDetUnit* det)const;

 protected:  
  edm::ParameterSet conf_;
  const TrackerGeometry * geom_;
  const MagneticField * magfield_ ;
  double appliedVoltage_;
  double rhall_;
  double mulow_;
  double vsat_;
  double beta_;
  const SiStripLorentzAngle* LorentzAngleMap_;
};

#endif





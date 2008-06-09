#ifndef RecoLocalTracker_SiStripRecHitConverter_StripCPE_H
#define RecoLocalTracker_SiStripRecHitConverter_StripCPE_H

#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/ClusterParameterEstimator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"

#include <ext/hash_map>

class StripTopology;


class StripCPE : public StripClusterParameterEstimator 
{
public:


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
  const SiStripLorentzAngle* LorentzAngleMap_;
  mutable LocalVector theCachedDrift;
  mutable unsigned int theCachedDetId;


public:
  
  void clearCache() {
    m_Params.clear();
  }

public:
  struct Param {
    Param() : topology(0){}
      StripTopology const * topology;
    LocalVector drift;
    float thickness;
    float maxLength;
    int nstrips;
  };
  
  Param const & param(DetId detId) const;

private:
  Param & fillParam(Param & p, const GeomDetUnit *  det);
  typedef  __gnu_cxx::hash_map< unsigned int, Param> Params;
  
  Params m_Params;
  
};

#endif





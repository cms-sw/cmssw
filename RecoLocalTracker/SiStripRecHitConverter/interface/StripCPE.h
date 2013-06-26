#ifndef RecoLocalTracker_SiStripRecHitConverter_StripCPE_H
#define RecoLocalTracker_SiStripRecHitConverter_StripCPE_H

#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"
#include "CondFormats/SiStripObjects/interface/SiStripConfObject.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include <ext/hash_map>
class StripTopology;

class StripCPE : public StripClusterParameterEstimator 
{
public:

  StripClusterParameterEstimator::LocalValues localParameters( const SiStripCluster& cl, const GeomDetUnit&) const;
  
  StripCPE( edm::ParameterSet & conf, 
	    const MagneticField&, 
	    const TrackerGeometry&, 
	    const SiStripLorentzAngle&,
	    const SiStripBackPlaneCorrection&,
	    const SiStripConfObject&,
	    const SiStripLatency&);    
  LocalVector driftDirection(const StripGeomDetUnit* det) const;

 protected:  

  const bool peakMode_;
  const TrackerGeometry & geom_;
  const MagneticField& magfield_ ;
  const SiStripLorentzAngle& LorentzAngleMap_;
  const SiStripBackPlaneCorrection& BackPlaneCorrectionMap_;
  std::vector<float> xtalk1;
  std::vector<float> xtalk2;

  struct Param {
    Param() : topology(0) {}
    StripTopology const * topology;
    LocalVector drift;
    float thickness, pitch_rel_err2, maxLength;
    int nstrips;
    float backplanecorrection;
    SiStripDetId::ModuleGeometry moduleGeom;
    float coveredStrips(const LocalVector&, const LocalPoint&) const;
  };
  Param const & param(const GeomDetUnit& det) const {
    return m_Params[det.index()-m_off];
  }

private:

  void fillParams();
  typedef  std::vector<Param> Params;  
  Params m_Params;
  unsigned int m_off;

};
#endif

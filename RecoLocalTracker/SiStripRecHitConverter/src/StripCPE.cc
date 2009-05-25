#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include <algorithm>
#include<cmath>

StripCPE::StripCPE( edm::ParameterSet & conf, 
		    const MagneticField * mag, 
		    const TrackerGeometry* geom, 
		    const SiStripLorentzAngle* LorentzAngle)
  : geom_(geom),
    magfield_(mag),
    LorentzAngleMap_(LorentzAngle) {}


StripClusterParameterEstimator::LocalValues StripCPE::
localParameters( const SiStripCluster& cluster) const {
  StripCPE::Param const & p = param(DetId(cluster.geographicalId()));
  const float strip = p.driftCorrected( cluster.barycenter() );
  return std::make_pair( p.topology->localPosition(strip),
			 p.topology->localError(strip, 1/12.) );
}

float StripCPE::Param::
driftCorrected(const float& strip) const {
  return driftCorrected(strip, topology->localPosition(strip));
}

float StripCPE::Param::
driftCorrected(const float& strip, const LocalPoint& lpos) const {
  return strip - 0.5*coveredStrips(drift, lpos);
}


float StripCPE::Param::
coveredStrips(const LocalVector& lvec, const LocalPoint& lpos) const {  
  return 
    topology->measurementPosition(lpos + 0.5*lvec).x() 
    - topology->measurementPosition(lpos - 0.5*lvec).x();
}


LocalVector StripCPE::
driftDirection(const StripGeomDetUnit* det) const { 
  LocalVector lbfield = (det->surface()).toLocal(magfield_->inTesla(det->surface().position()));  
  
  float tanLorentzAnglePerTesla = LorentzAngleMap_->getLorentzAngle(det->geographicalId().rawId());
  
  float dir_x = -tanLorentzAnglePerTesla * lbfield.y();
  float dir_y =  tanLorentzAnglePerTesla * lbfield.x();
  float dir_z =  1.; // E field always in z direction
  
  return LocalVector(dir_x,dir_y,dir_z);
}



StripCPE::Param const & StripCPE::
param(DetId detId) const {
  Param & p = const_cast<StripCPE*>(this)->m_Params[detId.rawId()];
  if (p.topology) return p;
  else return const_cast<StripCPE*>(this)->fillParam(p, geom_->idToDetUnit(detId));
}

StripCPE::Param & StripCPE::
fillParam(StripCPE::Param & p, const GeomDetUnit *  det) {  
  const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)(det);
  const Bounds& bounds = stripdet->specificSurface().bounds();
  
  p.maxLength = std::sqrt( std::pow(bounds.length(),2)+std::pow(bounds.width(),2) );
  p.thickness = bounds.thickness();
  p.drift = driftDirection(stripdet) * p.thickness;
  p.topology=(StripTopology*)(&stripdet->topology());    
  p.nstrips = p.topology->nstrips(); 
  p.subdet = SiStripDetId(stripdet->geographicalId()).subDetector();
  
  return p;
}

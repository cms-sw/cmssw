#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
//typedef std::pair<LocalPoint,LocalError>  LocalValues;
#include <algorithm>
#include<cmath>


StripCPE::Param & StripCPE::fillParam(StripCPE::Param & p, const GeomDetUnit *  det) {
  
  const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)(det);
  p.topology=(StripTopology*)(&stripdet->topology());
  
  p.drift = driftDirection(stripdet);

  p.thickness=stripdet->specificSurface().bounds().thickness();
  p.drift*=p.thickness;

  
  //p.drift = driftDirection(stripdet);
  //p.thickness=stripdet->surface().bounds().thickness();
  
  const Bounds& bounds = stripdet->surface().bounds();
  
  p.maxLength = std::sqrt( std::pow(bounds.length(),2)+std::pow(bounds.width(),2) );

  //  p.maxLength = sqrt( bounds.length()*bounds.length()+bounds.width()*bounds.width());
  // p.drift *= fabs(p.thickness/p.drift.z());       
  
  p.nstrips = p.topology->nstrips(); 
  return p;
}
  


StripCPE::Param const & StripCPE::param(DetId detId) const {
  Param & p = const_cast<StripCPE*>(this)->m_Params[detId.rawId()];
  if (p.topology) return p;
  else return const_cast<StripCPE*>(this)->fillParam(p, geom_->idToDetUnit(detId));
}



StripCPE::StripCPE(edm::ParameterSet & conf, const MagneticField * mag, const TrackerGeometry* geom, const SiStripLorentzAngle* LorentzAngle)
{
  magfield_  = mag;
  geom_ = geom;
  LorentzAngleMap_=LorentzAngle;
}

StripClusterParameterEstimator::LocalValues StripCPE::localParameters( const SiStripCluster & cl)const {
  //
  // get the det from the geometry
  //

  StripCPE::Param const & p = param(DetId(cl.geographicalId()));

  const StripTopology &topol= *(p.topology);

  LocalPoint position = topol.localPosition(cl.barycenter());
  LocalError eresult = topol.localError(cl.barycenter(),1/12.);

  LocalPoint  result=LocalPoint(position.x()-0.5*p.drift.x(),position.y()-0.5*p.drift.y(),0);
  return std::make_pair(result,eresult);
}

LocalVector StripCPE::driftDirection(const StripGeomDetUnit* det)const{
 
  LocalVector lbfield=(det->surface()).toLocal(magfield_->inTesla(det->surface().position()));
  
  float tanLorentzAnglePerTesla=LorentzAngleMap_->getLorentzAngle(det->geographicalId().rawId());
  
  
  float dir_x =-tanLorentzAnglePerTesla * lbfield.y();
  float dir_y =tanLorentzAnglePerTesla * lbfield.x();
  float dir_z = 1.; // E field always in z direction
  
  LocalVector theDrift = LocalVector(dir_x,dir_y,dir_z);
 
  return theDrift;
  
}

#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonTopologies/interface/RadialStripTopology.h"
#include <algorithm>
#include<cmath>

StripCPE::StripCPE( edm::ParameterSet & conf, 
		    const MagneticField * mag, 
		    const TrackerGeometry* geom, 
		    const SiStripLorentzAngle* LorentzAngle)
  : geom_(geom),
    magfield_(mag),
    LorentzAngleMap_(LorentzAngle)
{
  edm::ParameterSet outoftime = conf.getParameter<edm::ParameterSet>("OutOfTime");
  late.push_back( std::make_pair(outoftime.getParameter<double>("TIBlateFP"),outoftime.getParameter<double>("TIBlateBP")));
  late.push_back( std::make_pair(outoftime.getParameter<double>("TIDlateFP"),outoftime.getParameter<double>("TIDlateBP")));
  late.push_back( std::make_pair(outoftime.getParameter<double>("TOBlateFP"),outoftime.getParameter<double>("TOBlateBP")));
  late.push_back( std::make_pair(outoftime.getParameter<double>("TEClateFP"),outoftime.getParameter<double>("TEClateBP")));
}


StripClusterParameterEstimator::LocalValues StripCPE::
localParameters( const SiStripCluster& cluster) const {
  StripCPE::Param const & p = param(cluster.geographicalId());
  const float lfp(lateFrontPlane(p.subdet)), lbp(lateBackPlane(p.subdet));
  const float barycenter = cluster.barycenter();
  const float fullProjection = p.coveredStrips( p.drift + LocalVector(0,0,-p.thickness), LocalPoint(barycenter,0,0));
  const float strip = barycenter - 0.5 * (1-lbp+lfp) * fullProjection;
  // + 0.5*p.coveredStrips(track, ltp.position()); unnecessary, since hypothesized perpendicular track covers no strips.

  return std::make_pair( p.topology->localPosition(strip),
			 p.topology->localError(strip, 1/12.) );
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
param(const uint32_t detid) const {
  Param & p = const_cast<StripCPE*>(this)->m_Params[detid];
  if (p.topology) return p;
  else return const_cast<StripCPE*>(this)->fillParam(p, geom_->idToDetUnit(detid));
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
  
  const RadialStripTopology* rtop = dynamic_cast<const RadialStripTopology*>(p.topology);
  p.pitch_rel_err2 = (rtop) 
    ? pow( 0.5 * rtop->angularWidth() * rtop->stripLength()/rtop->localPitch(LocalPoint(0,0,0)), 2) / 12.
    : 0;
  
  return p;
}

float StripCPE::
lateFrontPlane(SiStripDetId::SubDetector subdet) const {
  return late[subdet-SiStripDetId::TIB].first;
}

float StripCPE::
lateBackPlane(SiStripDetId::SubDetector subdet) const {
  return late[subdet-SiStripDetId::TIB].second;
}

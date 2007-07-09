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



StripCPE::StripCPE(edm::ParameterSet & conf, const MagneticField * mag, const TrackerGeometry* geom)
{
  appliedVoltage_   = conf.getParameter<double>("AppliedVoltage");
  double chargeMobility   = conf.getParameter<double>("ChargeMobility");
  double temperature = conf.getParameter<double>("Temperature");
  rhall_            = conf.getParameter<double>("HoleRHAllParameter");
  double holeBeta    = conf.getParameter<double>("HoleBeta");
  double holeSaturationVelocity = conf.getParameter<double>("HoleSaturationVelocity");
  useDB_ = conf.getParameter<bool>("UseCalibrationFromDB"); 

  mulow_ = chargeMobility*std::pow((temperature/300.),-2.5);
  vsat_ = holeSaturationVelocity*std::pow((temperature/300.),0.52);
  beta_ = holeBeta*std::pow((temperature/300.),0.17);
  
  magfield_  = mag;
  geom_ = geom;
  theCachedDetId=0;
  LorentzAngleMap_=0;
}


StripCPE::StripCPE(edm::ParameterSet & conf, const MagneticField * mag, const TrackerGeometry* geom, const SiStripLorentzAngle* LorentzAngle)
{
  useDB_ = conf.getParameter<bool>("UseCalibrationFromDB") ;
  magfield_  = mag;
  geom_ = geom;
  theCachedDetId=0;
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
  if ( theCachedDetId != det->geographicalId().rawId() ){
    LocalVector lbfield=(det->surface()).toLocal(magfield_->inTesla(det->surface().position()));
    
    float tanLorentzAnglePerTesla=0;
    
    if(useDB_) 
      tanLorentzAnglePerTesla = LorentzAngleMap_->getLorentzAngle(det->geographicalId().rawId());
    else{
      float thickness=det->specificSurface().bounds().thickness();
      float e = appliedVoltage_/thickness;
      float mu = ( mulow_/(std::pow(double((1+std::pow((mulow_*e/vsat_),beta_))),1./beta_)));
      tanLorentzAnglePerTesla = 1.E-4 *mu*rhall_;
    }
    
    float dir_x =-tanLorentzAnglePerTesla * lbfield.y();
    float dir_y =tanLorentzAnglePerTesla * lbfield.x();
    float dir_z = 1.; // E field always in z direction
      
    theCachedDrift = LocalVector(dir_x,dir_y,dir_z);
  }
  //  if((drift-drift_old).mag()>1.E-7)std::cout<<"old drift= "<<drift_old<<" new drift="<<drift<<std::endl;
  return theCachedDrift;

}

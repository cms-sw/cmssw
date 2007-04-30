#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
//typedef std::pair<LocalPoint,LocalError>  LocalValues;

StripCPE::StripCPE(edm::ParameterSet & conf, const MagneticField * mag, const TrackerGeometry* geom)
{
  conf_=conf;
  appliedVoltage_   = conf.getParameter<double>("AppliedVoltage");
  double chargeMobility   = conf.getParameter<double>("ChargeMobility");
  double temperature = conf.getParameter<double>("Temperature");
  rhall_            = conf.getParameter<double>("HoleRHAllParameter");
  double holeBeta    = conf.getParameter<double>("HoleBeta");
  double holeSaturationVelocity = conf.getParameter<double>("HoleSaturationVelocity");
  
  mulow_ = chargeMobility*std::pow((temperature/300.),-2.5);
  vsat_ = holeSaturationVelocity*std::pow((temperature/300.),0.52);
  beta_ = holeBeta*std::pow((temperature/300.),0.17);
  
  magfield_  = mag;
  geom_ = geom;
  theCachedDetId=0;
  LorentzAngleMap_=0;
}


StripCPE::StripCPE(edm::ParameterSet & conf, const MagneticField * mag, const TrackerGeometry* geom,const SiStripLorentzAngle* LorentzAngle)
{
  conf_=conf;
  magfield_  = mag;
  geom_ = geom;
  theCachedDetId=0;
  LorentzAngleMap_=LorentzAngle;
}

StripClusterParameterEstimator::LocalValues StripCPE::localParameters( const SiStripCluster & cl)const {
  //
  // get the det from the geometry
  //
  DetId detId(cl.geographicalId());
  const GeomDetUnit *  det = geom_->idToDetUnit(detId);

  LocalPoint position;
  LocalError eresult;
  LocalVector drift=LocalVector(0,0,1);
  const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)(det);
  //  DetId detId(det.geographicalId());
  const StripTopology &topol=(StripTopology&)stripdet->topology();
  position = topol.localPosition(cl.barycenter());
  eresult = topol.localError(cl.barycenter(),1/12.);
  drift = driftDirection(stripdet);
  float thickness=stripdet->specificSurface().bounds().thickness();
  drift*=thickness;
  LocalPoint  result=LocalPoint(position.x()+drift.x()/2,position.y()+drift.y()/2,0);
  return std::make_pair(result,eresult);
}

LocalVector StripCPE::driftDirection(const StripGeomDetUnit* det)const{
  if ( theCachedDetId != det->geographicalId().rawId() ){
    LocalVector lbfield=(det->surface()).toLocal(magfield_->inTesla(det->surface().position()));
    float thickness=det->specificSurface().bounds().thickness();
    
    float tanLorentzAnglePerTesla=0;
    
    if(conf_.getParameter<bool>("UseCalibrationFromDB"))tanLorentzAnglePerTesla=  LorentzAngleMap_->getLorentzAngle(det->geographicalId().rawId());
    else{
      float e = appliedVoltage_/thickness;
      float mu = ( mulow_/(std::pow(double((1+std::pow((mulow_*e/vsat_),beta_))),1./beta_)));
      tanLorentzAnglePerTesla = 1.E-4 *mu*rhall_;
    }
    
    float dir_x =tanLorentzAnglePerTesla * lbfield.y();
    float dir_y =-tanLorentzAnglePerTesla * lbfield.x();
    float dir_z = 1.; // E field always in z direction
      
    theCachedDrift = LocalVector(dir_x,dir_y,dir_z);
  }
  //  if((drift-drift_old).mag()>1.E-7)std::cout<<"old drift= "<<drift_old<<" new drift="<<drift<<std::endl;
  return theCachedDrift;

}

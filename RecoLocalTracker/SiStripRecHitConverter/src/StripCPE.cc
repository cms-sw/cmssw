#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
//typedef std::pair<LocalPoint,LocalError>  LocalValues;

StripCPE::StripCPE(edm::ParameterSet & conf, const MagneticField * mag, const TrackerGeometry* geom){
  //--- Lorentz angle tangent per Tesla
  //theTanLorentzAnglePerTesla_ =
    //    conf.getParameter<double>("TanLorentzAnglePerTesla");  

   appliedVoltage_   = conf.getParameter<double>("AppliedVoltage");
   chargeMobility_   = conf.getParameter<double>("ChargeMobility");
   temperature_      = conf.getParameter<double>("Temperature");
   rhall_            = conf.getParameter<double>("HoleRHAllParameter");
   holeBeta_         = conf.getParameter<double>("HoleBeta");
   holeSaturationVelocity_ = conf.getParameter<double>("HoleSaturationVelocity");

   //useMagneticField_=  conf.getParameter<bool>("UseMagneticField");  
    
  magfield_  = mag;
  geom_ = geom;
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
  LocalVector lbfield=(det->surface()).toLocal(magfield_->inTesla(det->surface().position()));

  float thickness=det->specificSurface().bounds().thickness();

  float mulow = chargeMobility_*pow((temperature_/300.),-2.5);
  float vsat = holeSaturationVelocity_*pow((temperature_/300.),0.52);
  float beta = holeBeta_*pow((temperature_/300.),0.17);
 
  float e = appliedVoltage_/thickness;
  float mu = ( mulow/(pow(double((1+pow((mulow*e/vsat),beta))),1./beta)));
  float hallMobility = mu*rhall_;
 
   float dir_x = 1.E-4 * hallMobility * lbfield.y();
   float dir_y = -1.E-4 * hallMobility * lbfield.x();
   float dir_z = 1.; // E field always in z direction

   LocalVector drift = LocalVector(dir_x,dir_y,dir_z);
  return drift;

}

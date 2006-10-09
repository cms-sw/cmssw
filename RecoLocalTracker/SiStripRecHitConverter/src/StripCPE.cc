#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
//typedef std::pair<LocalPoint,LocalError>  LocalValues;

StripCPE::StripCPE(edm::ParameterSet & conf, const MagneticField * mag, const TrackerGeometry* geom){
  //--- Lorentz angle tangent per Tesla
  theTanLorentzAnglePerTesla_ =
    conf.getParameter<double>("TanLorentzAnglePerTesla");  
  useMagneticField_=  conf.getParameter<bool>("UseMagneticField");  
    
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
  if(useMagneticField_) drift = driftDirection(stripdet);
  float thickness=stripdet->specificSurface().bounds().thickness();
  drift*=thickness;
  LocalPoint  result=LocalPoint(position.x()+drift.x()/2,position.y()+drift.y()/2,0);
  return std::make_pair(result,eresult);
}

LocalVector StripCPE::driftDirection(const StripGeomDetUnit* det)const{
  LocalVector lbfield=(det->surface()).toLocal(magfield_->inTesla(det->surface().position()));
   float dir_x = theTanLorentzAnglePerTesla_ * lbfield.y();
   float dir_y = -theTanLorentzAnglePerTesla_ * lbfield.x();
   float dir_z = 1.; // E field always in z direction
   LocalVector drift = LocalVector(dir_x,dir_y,dir_z);
  return drift;

}

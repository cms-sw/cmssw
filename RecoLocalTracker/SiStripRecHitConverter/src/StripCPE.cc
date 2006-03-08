#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
//typedef std::pair<LocalPoint,LocalError>  LocalValues;

StripCPE::StripCPE(edm::ParameterSet & conf, const MagneticField * mag, const TrackingGeometry* geom){
  //--- Lorentz angle tangent per Tesla
  theTanLorentzAnglePerTesla =
    conf.getParameter<double>("TanLorentzAnglePerTesla");  

  magfield_  = mag;
  geom_ = geom;
}

StripClusterParameterEstimator::LocalValues StripCPE::localParameters( const SiStripCluster & cl){
  //
  // get the det from the geometry
  //
  DetId detId(cl.geographicalId());
  const GeomDetUnit *  det = geom_->idToDet(detId);

  LocalPoint result;
  LocalError eresult;

  const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)(det);
  //  DetId detId(det.geographicalId());
  const StripTopology &topol=(StripTopology&)stripdet->topology();
  result = topol.localPosition(cl.barycenter());
  eresult = topol.localError(cl.barycenter(),1/12.);
  
  //Apply  lorentz drift
  LocalVector drift = driftDirection(stripdet);
  float thickness=stripdet->specificSurface().bounds().thickness();
  drift*=thickness;
  return std::make_pair(result+drift,eresult);
}

LocalVector StripCPE::driftDirection(const StripGeomDetUnit* det){
  LocalVector lbfield=(det->surface()).toLocal(magfield_->inTesla(det->surface().position()));
   float dir_x = -theTanLorentzAnglePerTesla * lbfield.y();
   float dir_y = theTanLorentzAnglePerTesla * lbfield.x();
   float dir_z = 1.; // E field always in z direction
   LocalVector drift = LocalVector(dir_x,dir_y,dir_z);
  return drift;

}

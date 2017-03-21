
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPE.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"


Phase2StripCPE::Phase2StripCPE(edm::ParameterSet & conf, const MagneticField & magf)
{
  magfield_ = &magf;
  use_LorentzAngle_DB_ = conf.getParameter<bool>("LorentzAngle_DB");
  if (use_LorentzAngle_DB_) {
    throw cms::Exception("Lorentz Angle from DB not implemented yet");
    tanLorentzAnglePerTesla_ = 0;
    // old code: LorentzAngleMap_.getLorentzAngle(det->geographicalId().rawId());
  } else {
    tanLorentzAnglePerTesla_ = conf.getParameter<double>("TanLorentzAnglePerTesla");
  }
}


Phase2StripCPE::LocalValues Phase2StripCPE::localParameters(
  const Phase2TrackerCluster1D & cluster,
  const GeomDetUnit & detunit) const
{
  const Phase2TrackerGeomDetUnit & det = (const Phase2TrackerGeomDetUnit &) detunit;
  const Phase2TrackerTopology * topo = &det.specificTopology();

  float pitch_x = topo->pitch().first;
  float pitch_y = topo->pitch().second;

  // see https://github.com/cms-sw/cmssw/blob/CMSSW_8_1_X/RecoLocalTracker/SiStripRecHitConverter/src/StripCPE.cc

  float thickness = det.specificSurface().bounds().thickness();
  LocalVector drift = driftDirection(det) * thickness;
  LocalVector lvec = drift + LocalVector(0,0,-thickness);
  float coveredStrips = lvec.x() / pitch_x; // simplifies wrt Phase0 tracker because only rectangular modules

  float ix = cluster.center() - 0.5 * coveredStrips;
  float iy = cluster.column()+0.5; // halfway the column

  LocalPoint lp( topo->localX(ix), topo->localY(iy), 0 );          // x, y, z
  LocalError le( pow(pitch_x, 2) / 12, 0, pow(pitch_y, 2) / 12);   // e2_xx, e2_xy, e2_yy

  return std::make_pair( lp, le );
}


LocalVector Phase2StripCPE::driftDirection(
  const Phase2TrackerGeomDetUnit & det) const
{
  LocalVector lbfield = (det.surface()).toLocal(magfield_->inTesla(det.surface().position()));  

  float dir_x = -tanLorentzAnglePerTesla_ * lbfield.y();
  float dir_y =  tanLorentzAnglePerTesla_ * lbfield.x();
  float dir_z =  1.f; // E field always in z direction

  return LocalVector(dir_x,dir_y,dir_z);
}

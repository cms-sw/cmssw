#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPE.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

Phase2StripCPE::Phase2StripCPE(edm::ParameterSet & conf, const MagneticField & magf,const TrackerGeometry& geom) :
   magfield_(magf),
   geom_(geom),
   tanLorentzAnglePerTesla_(conf.getParameter<double>("TanLorentzAnglePerTesla"))
{
  use_LorentzAngle_DB_ = conf.getParameter<bool>("LorentzAngle_DB");
  if (use_LorentzAngle_DB_) {
    throw cms::Exception("Lorentz Angle from DB not implemented yet");
    tanLorentzAnglePerTesla_ = 0;
    // old code: LorentzAngleMap_.getLorentzAngle(det->geographicalId().rawId());
  }
  fillParam();
}


Phase2StripCPE::LocalValues Phase2StripCPE::localParameters(
  const Phase2TrackerCluster1D & cluster,
  const GeomDetUnit & detunit) const
{
  auto const & p =  m_Params[detunit.index()-m_off];
  auto const & topo = *p.topology;
  float ix = cluster.center() - 0.5f * p.coveredStrips;
  float iy = float(cluster.column())+0.5f; // halfway the column

  LocalPoint lp( topo.localX(ix), topo.localY(iy), 0 );          // x, y, z

  return std::make_pair( lp, p.localErr );
}


LocalVector Phase2StripCPE::driftDirection(
  const Phase2TrackerGeomDetUnit & det) const
{
  LocalVector lbfield = (det.surface()).toLocal(magfield_.inTesla(det.surface().position()));  

  float dir_x = -tanLorentzAnglePerTesla_ * lbfield.y();
  float dir_y =  tanLorentzAnglePerTesla_ * lbfield.x();
  float dir_z =  1.f; // E field always in z direction

  return LocalVector(dir_x,dir_y,dir_z);
}


void Phase2StripCPE::fillParam() {

   // in phase 2 they are all pixel topologies...
   auto const & dus = geom_.detUnits();
   m_off = dus.size();
   // skip Barrel and Foward pixels...
   for(unsigned int i=3;i<7;++i) {
     LogDebug("LookingForFirstPhase2OT") << " Subdetector " << i
      << " GeomDetEnumerator " << GeomDetEnumerators::tkDetEnum[i]
      << " offset " << geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i]) << std::endl;
     if(geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i]) != dus.size()) {
       if(geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i]) < m_off) m_off = geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i]);
     }
   }
   LogDebug("LookingForFirstPhase2OT") << " Chosen offset: " << m_off;

   m_Params.resize(dus.size()-m_off);
   // very very minimal, for sure it will need to expand...
   for (auto i=m_off; i!=dus.size();++i) {
     auto & p= m_Params[i-m_off];
 
     const Phase2TrackerGeomDetUnit & det = (const Phase2TrackerGeomDetUnit &)(*dus[i]);
     assert(det.index()==int(i));
     p.topology = &det.specificTopology();

     auto pitch_x = p.topology->pitch().first;
     auto pitch_y = p.topology->pitch().second;

     // see https://github.com/cms-sw/cmssw/blob/CMSSW_8_1_X/RecoLocalTracker/SiStripRecHitConverter/src/StripCPE.cc
     auto thickness = det.specificSurface().bounds().thickness();
     auto drift = driftDirection(det) * thickness;
     auto lvec = drift + LocalVector(0,0,-thickness);
     p.coveredStrips = lvec.x() / pitch_x; // simplifies wrt Phase0 tracker because only rectangular modules

     constexpr float o12 = 1./12;
     p.localErr = LocalError( o12*pitch_x*pitch_x, 0, o12*pitch_y*pitch_y);   // e2_xx, e2_xy, e2_yy
   }
}

#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEfromTrackAngle.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"                                                           

#include "vdt/vdtMath.h"

float StripCPEfromTrackAngle::stripErrorSquared(const unsigned N, const float uProj, const int& loc ) const {
  float uerr = 0;
  if(N <= 4)
    uerr = LC_P0*uProj*vdt::fast_expf(-uProj*LC_P1)+LC_P2;
  else
    switch(loc){
    case SiStripDetId::TEC :
      uerr = TEC_P0+uProj*TEC_P1; break;
    case SiStripDetId::TID :
      uerr = TID_P0+uProj*TID_P1; break;
    case SiStripDetId::TOB :
      uerr = TOB_P0+uProj*TOB_P1; break;
    case SiStripDetId::TIB :
      uerr = TIB_P0+uProj*TIB_P1; break;
    default:
      throw cms::Exception("StripCPEfromTrackAngle::stripErrorSquared", "Incompatible sub-detector.");
      break;
    }
  return uerr*uerr;
}

StripClusterParameterEstimator::LocalValues StripCPEfromTrackAngle::
localParameters( const SiStripCluster& cluster, const GeomDetUnit& det, const LocalTrajectoryParameters& ltp) const {
  
  StripCPE::Param const & p = param(det);
  SiStripDetId ssdid = SiStripDetId( det.geographicalId() );  
 
  LocalVector track = ltp.momentum();
  track *= 
    (track.z()<0) ?  std::abs(p.thickness/track.z()) : 
    (track.z()>0) ? -std::abs(p.thickness/track.z()) :  
                         p.maxLength/track.mag() ;

  const unsigned N = cluster.amplitudes().size();
  const float fullProjection = p.coveredStrips( track+p.drift, ltp.position());
  const float uerr2 = stripErrorSquared( N, std::abs(fullProjection),ssdid.subdetId() );
  const float strip = cluster.barycenter() -  0.5f*(1.f-p.backplanecorrection) * fullProjection
    + 0.5f*p.coveredStrips(track, ltp.position());
  
  return std::make_pair( p.topology->localPosition(strip, ltp.vector()),
			 p.topology->localError(strip, uerr2, ltp.vector()) );
}


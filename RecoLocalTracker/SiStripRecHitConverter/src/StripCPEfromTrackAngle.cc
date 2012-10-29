#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEfromTrackAngle.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"                                                           

#include "vdt/vdtMath.h"

namespace {
  inline
  float stripErrorSquared(const unsigned N, const float uProj) {
    if( (float(N)-uProj) > 3.5f )  
      return float(N*N)/12.f;
    else {
      typedef float Float;
      constexpr Float P1=-0.339;
      constexpr Float P2=0.90;
      constexpr Float P3=0.279;
      const float uerr = P1*uProj*vdt::fast_expf(-uProj*P2)+P3;
      // const Float uerr = P1*uProj*std::exp(-uProj*P2)+P3;
      return uerr*uerr;
    }
  }
}

StripClusterParameterEstimator::LocalValues StripCPEfromTrackAngle::
localParameters( const SiStripCluster& cluster, const GeomDetUnit& det, const LocalTrajectoryParameters& ltp) const {
  
  StripCPE::Param const & p = param(det);
  
  LocalVector track = ltp.momentum();
  track *= 
    (track.z()<0) ?  std::abs(p.thickness/track.z()) : 
    (track.z()>0) ? -std::abs(p.thickness/track.z()) :  
                         p.maxLength/track.mag() ;

  const unsigned N = cluster.amplitudes().size();
  const float fullProjection = p.coveredStrips( track+p.drift, ltp.position());
  const float uerr2 = stripErrorSquared( N, std::abs(fullProjection) );
  const float strip = cluster.barycenter() -  0.5f*(1.f-shift[p.moduleGeom]) * fullProjection
    + 0.5f*p.coveredStrips(track, ltp.position());
  
  return std::make_pair( p.topology->localPosition(strip, ltp.vector()),
			 p.topology->localError(strip, uerr2, ltp.vector()) );
}


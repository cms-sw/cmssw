#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEfromTrackAngle.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"                                                           


StripClusterParameterEstimator::LocalValues StripCPEfromTrackAngle::
localParameters( const SiStripCluster& cluster, const GeomDetUnit& det, const LocalTrajectoryParameters& ltp) const {
  return localParameters(cluster,ltp);
}

StripClusterParameterEstimator::LocalValues StripCPEfromTrackAngle::
localParameters( const SiStripCluster& cluster, const LocalTrajectoryParameters& ltp) const {
  StripCPE::Param const & p = param(cluster.geographicalId());
  
  LocalVector track = ltp.momentum();
  track *= 
    (track.z()<0) ?  fabs(p.thickness/track.z()) : 
    (track.z()>0) ? -fabs(p.thickness/track.z()) :  
                         p.maxLength/track.mag() ;

  const unsigned N = cluster.amplitudes().size();
  const float uProj = p.coveredStrips( track+p.drift, ltp.position());
  const float uerr2 = stripErrorSquared( N, fabs(uProj) );
  const float strip = p.driftCorrected( cluster.barycenter(), ltp.position());

  return std::make_pair( p.topology->localPosition(strip),
			 p.topology->localError(strip, uerr2) );
}

inline
float StripCPEfromTrackAngle::
stripErrorSquared(const unsigned N, const float uProj) const
{
  if( (N-uProj) > 3.5 )  
    return N*N/12.;
  else {
    const float P1=-0.339;
    const float P2=0.90;
    const float P3=0.279;
    const float uerr = P1*uProj*exp(-uProj*P2)+P3;
    return uerr*uerr;
  }
}

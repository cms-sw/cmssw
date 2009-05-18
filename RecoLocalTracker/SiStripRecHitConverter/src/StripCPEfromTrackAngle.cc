#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEfromTrackAngle.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"                                                           

StripClusterParameterEstimator::LocalValues StripCPEfromTrackAngle::
localParameters( const SiStripCluster& cluster, const LocalTrajectoryParameters& ltp) const
{
  StripCPE::Param const & p = param(DetId(cluster.geographicalId()));
  
  LocalVector track = ltp.momentum();
  track *= 
    (track.z()<0) ?  fabs(p.thickness/track.z()) : 
    (track.z()>0) ? -fabs(p.thickness/track.z()) :  
                         p.maxLength/track.mag() ;
  const float localPitch = p.topology->localPitch(ltp.position());
  const float uProj = fabs( (track+p.drift).x() ) / localPitch;
  const unsigned N = cluster.amplitudes().size();

  float uerr2;
  if( (N-uProj) > 3.5 )  
    uerr2 = N*N/12.;
  else {
    const float P1=-0.339;
    const float P2=0.90;
    const float P3=0.279;
    const float uerr = P1*uProj*exp(-uProj*P2)+P3;
    uerr2 = uerr*uerr;
  }

  float position = cluster.barycenter() - 0.5*p.drift.x()/localPitch;
  return std::make_pair( p.topology->localPosition(position),
			 p.topology->localError(position, uerr2) );
}

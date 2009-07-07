#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEgeometric.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/CrosstalkInversion.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include <numeric>

StripCPEgeometric::StripCPEgeometric( edm::ParameterSet& conf, 
				      const MagneticField* mag, 
				      const TrackerGeometry* geom, 
				      const SiStripLorentzAngle* LorentzAngle)
  : StripCPE(conf, mag, geom, LorentzAngle ),
    tan_diffusion_angle(conf.getParameter<double>("TanDiffusionAngle")),    
    thickness_rel_err2(pow(conf.getParameter<double>("ThicknessRelativeUncertainty"), 2)),
    noise_threshold(conf.getParameter<double>("NoiseThreshold")),
    maybe_noise_threshold(conf.getParameter<double>("MaybeNoiseThreshold")),
    scaling_squared(pow(conf.getParameter<double>("UncertaintyScaling"), 2))
{
  std::string mode = conf.getParameter<bool>("APVpeakmode") ? "Peak" : "Dec";
  crosstalk.resize(7,0);
  crosstalk[SiStripDetId::TIB] = conf.getParameter<double>("CouplingConstant"+mode+"TIB");
  crosstalk[SiStripDetId::TID] = conf.getParameter<double>("CouplingConstant"+mode+"TID");
  crosstalk[SiStripDetId::TOB] = conf.getParameter<double>("CouplingConstant"+mode+"TOB");
  crosstalk[SiStripDetId::TEC] = conf.getParameter<double>("CouplingConstant"+mode+"TEC");
}

StripClusterParameterEstimator::LocalValues StripCPEgeometric::
localParameters( const SiStripCluster& cluster, const GeomDetUnit& det, const LocalTrajectoryParameters& ltp) const {
  return localParameters(cluster,ltp);
}

StripClusterParameterEstimator::LocalValues StripCPEgeometric::
localParameters( const SiStripCluster& cluster, const LocalTrajectoryParameters& ltp) const {
  StripCPE::Param const& p = param(DetId(cluster.geographicalId()));

  const LocalPoint& pos = ltp.position();
  LocalVector track = ltp.momentum();
  track *=  (track.z()<0) ?  fabs(p.thickness/track.z()) : 
            (track.z()>0) ? -fabs(p.thickness/track.z()) :  
                             p.maxLength/track.mag() ;

  const std::vector<stats_t<float> > Q = reco::InverseCrosstalkMatrix::unfold( cluster.amplitudes(), crosstalk[p.subdet] );
  
  const stats_t<float> projection = find_projection( p, track, pos );

  const stats_t<float> strip = cluster.firstStrip() + offset_from_firstStrip( Q, projection );

  const float corrected = p.driftCorrected( strip() , pos );
  
  return std::make_pair( p.topology->localPosition( corrected ),
			 p.topology->localError( corrected, strip.error2() ) );
}


stats_t<float> StripCPEgeometric::
find_projection(const StripCPE::Param& p, const LocalVector& track, const LocalPoint& position) const {
  const float projection = fabs( p.coveredStrips( track+p.drift, position ));
  const float minProjection = 2*p.thickness*tan_diffusion_angle/p.topology->localPitch(position);
  const float projection_rel_err2 = thickness_rel_err2 + p.pitch_rel_err2;
  return stats_t<float>::from_relative_uncertainty2( std::max(projection, minProjection), projection_rel_err2);
}


stats_t<float> StripCPEgeometric::
offset_from_firstStrip( const std::vector<stats_t<float> >& Q, const stats_t<float>& proj) const {
  WrappedCluster wc(Q);
  while( useNMinusOne( wc, proj) ) 
    wc.dropSmallerEdgeStrip();

  if( proj() < wc.N-2 || wc.deformed() )                             
    return stats_t<float>( wc.middle(),   wc.N * wc.N / 12.);

  if( proj > wc.maxProjection() ) 
    return stats_t<float>( wc.centroid()(),         1 / 12.);

  if( wc.N > 1 && wc.smallerEdgeStrip().sigmaFrom(0) < maybe_noise_threshold ) {
    WrappedCluster wc2(wc); 
    wc2.dropSmallerEdgeStrip();
    return between_positions( geometric_position( wc, proj), 
			      geometric_position( wc2, proj) );
  }
  return geometric_position(wc, proj);
}

stats_t<float> StripCPEgeometric::
geometric_position(const StripCPEgeometric::WrappedCluster& wc, const stats_t<float>& proj) const {
  const stats_t<float> x = wc.middle()  +  0.5 * proj * wc.eta();
  return wc.N==1 
    ?  stats_t<float>( x(), pow( 1-0.82*proj(), 2 ) / 12 ) 
    :  stats_t<float>( x(), scaling_squared * x.error2() ) ;
}

inline
stats_t<float> StripCPEgeometric::
between_positions(const stats_t<float>& one, const stats_t<float>& two) const {
  const float position = ( one() + two() ) / 2;
  const float error = fabs(position-one()) + std::max(one.error(), two.error());
  return stats_t<float>( position, error*error );
}

inline
bool StripCPEgeometric::
useNMinusOne(const WrappedCluster& wc, const stats_t<float>& proj) const {
  if( proj() > wc.N-1 ||
      wc.smallerEdgeStrip().sigmaFrom(0) > maybe_noise_threshold )  
    return false;
  if( wc.smallerEdgeStrip() < 0 ||
      proj() < wc.N-2) 
    return true;

  WrappedCluster wcTest(wc); wcTest.dropSmallerEdgeStrip();
  if( proj >= wcTest.maxProjection() || 
      wc.eta().sigmaFrom(0) < 3 ) return false;
  if( wc.sign()*wc.eta()() > 1./(wc.N-1) ) return true;
  if( wc.N==2 || wc.N==3) {
    return wc.smallerEdgeStrip().sigmaFrom(0) < noise_threshold;
  }
  return  wcTest.dedxRatio(proj).sigmaFrom(1) < wc.dedxRatio(proj).sigmaFrom(1) ; 
}


StripCPEgeometric::WrappedCluster::
WrappedCluster(const std::vector<stats_t<float> >& Q) : 
  N(Q.size()),
  Qbegin(Q.begin()),
  first(Q.begin())
  {}

inline
void StripCPEgeometric::WrappedCluster::
dropSmallerEdgeStrip() {
  if( *first < last() ) { first++; N-=1; } else 
  if( last() < *first ) {          N-=1; } else  
                        { first++; N-=2; } 
}

inline
float StripCPEgeometric::WrappedCluster::
middle() const 
{ return (first-Qbegin) + N/2.;}

inline
stats_t<float> StripCPEgeometric::WrappedCluster::
centroid() const { 
  stats_t<float> sumXQ(0);
  for(std::vector<stats_t<float> >::const_iterator i = first; i<first+N; i++) sumXQ += (i-Qbegin)*(*i);
  return sumXQ/sumQ() + 0.5;
}

inline
stats_t<float> StripCPEgeometric::WrappedCluster::
sumQ() const
{ return accumulate(first, first+N, stats_t<float>(0));}

inline
stats_t<float> StripCPEgeometric::WrappedCluster::
eta() const 
{ return  ( last() - *first ) / sumQ() ;}

inline
bool StripCPEgeometric::WrappedCluster::
deformed() const 
{  return  N>2  &&  std::max((*first)(),last()()) > accumulate(first+1,first+N-1,stats_t<float>(0))() / (N-2);}

inline
stats_t<float> StripCPEgeometric::WrappedCluster::
maxProjection() const
{ return N * (1 + sign()*eta() ).inverse(); }

inline
stats_t<float> StripCPEgeometric::WrappedCluster::
smallerEdgeStrip() const
{ return std::min(*first, last()); }

inline
stats_t<float> StripCPEgeometric::WrappedCluster::
dedxRatio(const stats_t<float>& projection) const 
{ return ( sumQ()/(*first+last()) - 1 ) * ( projection/(N-2) - 1 ); }

inline
int StripCPEgeometric::WrappedCluster::
sign() const
{ return ( *first < last() ) ? 1  : -1; }

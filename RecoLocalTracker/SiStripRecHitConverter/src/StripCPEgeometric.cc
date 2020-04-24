#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEgeometric.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/CrosstalkInversion.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include <numeric>

StripClusterParameterEstimator::LocalValues StripCPEgeometric::
localParameters( const SiStripCluster& cluster, const GeomDetUnit& det, const LocalTrajectoryParameters& ltp) const {
  StripCPE::Param const& p = param(det);

  const LocalPoint& pos = ltp.position();
  LocalVector track = ltp.momentum();
  track *=  (track.z()<0) ?  fabs(p.thickness/track.z()) : 
            (track.z()>0) ? -fabs(p.thickness/track.z()) :  
                             p.maxLength/track.mag() ;

  const float fullProjection = p.coveredStrips( track+p.drift, pos );
  stats_t<float> projection;
  {
    const float absProj = fabs( fullProjection );
    const float minProj = 2*p.thickness*tan_diffusion_angle/p.topology->localPitch(pos);
    const float projection_rel_err2 = thickness_rel_err2 + p.pitch_rel_err2;
    projection = stats_t<float>::from_relative_uncertainty2( std::max(absProj, minProj), projection_rel_err2);
  }

  const std::vector<stats_t<float> > Q = reco::InverseCrosstalkMatrix::unfold( cluster, xtalk1[p.moduleGeom] );
  const stats_t<float> strip = cluster.firstStrip() + offset_from_firstStrip( Q, projection );

  const float corrected = strip() -  0.5*(1-p.backplanecorrection) * fullProjection
    + 0.5*p.coveredStrips(track, ltp.position());
  const float error2 = std::max( strip.error2(), minimum_uncertainty_squared );  

  return std::make_pair( p.topology->localPosition( corrected, ltp.vector()),
			 p.topology->localError( corrected, error2, ltp.vector()));
}


stats_t<float> StripCPEgeometric::
offset_from_firstStrip( const std::vector<stats_t<float> >& Q, const stats_t<float>& proj) const {
  WrappedCluster wc(Q);
  if(    useNPlusOne( wc, proj))  wc.addSuppressedEdgeStrip();  else 
  while( useNMinusOne( wc, proj)) wc.dropSmallerEdgeStrip();

  if( proj() < wc.N-2 )           return stats_t<float>( wc.middle(),  pow(wc.N-proj(),2) / 12.);
  if( wc.deformed()   )           return stats_t<float>( wc.centroid()(),               1 / 12.);
  if( proj > wc.maxProjection() ) return stats_t<float>( wc.centroid()(),               1 / 12.);

  if( ambiguousSize( wc, proj) ) {
    const stats_t<float> probably = geometric_position( wc, proj);
    wc.dropSmallerEdgeStrip();
    const stats_t<float> maybe = geometric_position( wc, proj);
    return stats_t<float>( probably(), std::max( probably.error2(), float(maybe.error2() + pow( probably()-maybe() ,2)/12 )) );
  }
  return geometric_position( wc, proj);
}

stats_t<float> StripCPEgeometric::
geometric_position(const StripCPEgeometric::WrappedCluster& wc, const stats_t<float>& proj) const {
  const stats_t<float> x = wc.middle()  +  0.5 * proj * wc.eta();
  return wc.N==1 
    ?  stats_t<float>( x(), pow( 1-0.82*proj(), 2 ) / 12 ) 
    :  stats_t<float>( x(), scaling_squared * x.error2() ) ;
}

inline
bool StripCPEgeometric::
useNPlusOne(const WrappedCluster& wc, const stats_t<float>& proj) const 
{ return wc.maxProjection() < proj && proj() < wc.N+1 ; }

inline
bool StripCPEgeometric::
useNMinusOne(const WrappedCluster& wc, const stats_t<float>& proj) const {
  if( proj() > wc.N-1) return false;
  if( wc.smallerEdgeStrip() < 0 ) return true;
  if( proj() < wc.N-3) return false;
  if( proj() < wc.N-2) return true;
  if( wc.eta().sigmaFrom(0) < 3) return false;

  WrappedCluster wcTest(wc); wcTest.dropSmallerEdgeStrip();
  if( proj >= wcTest.maxProjection() ) return false;
  if( wc.sign()*wc.eta()() > 1./(wc.N-1) ) return true;

  return wc.smallerEdgeStrip().sigmaFrom(0) < noise_threshold;
}

inline
bool StripCPEgeometric::
ambiguousSize( const WrappedCluster& wc, const stats_t<float>& proj) const 
{  return 
     proj() < wc.N-1 && 
     wc.smallerEdgeStrip()()>0 && 
     wc.smallerEdgeStrip().sigmaFrom(0) < maybe_noise_threshold; }

StripCPEgeometric::WrappedCluster::
WrappedCluster(const std::vector<stats_t<float> >& Q) : 
  N(Q.size()-2),
  clusterFirst(Q.begin()+1),
  first(clusterFirst)
  {}

inline
void StripCPEgeometric::WrappedCluster::
addSuppressedEdgeStrip() {
  if( eta().sigmaFrom(0) < 1 ) { first--; N+=2; } 
  else if( *first > last() )   { first--; N+=1; } 
  else                         {          N+=1; }
}

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
{ return (first-clusterFirst) + N/2.;}

inline
stats_t<float> StripCPEgeometric::WrappedCluster::
centroid() const { 
  stats_t<float> sumXQ(0);
  for(std::vector<stats_t<float> >::const_iterator i = first; i<first+N; i++) sumXQ += (i-clusterFirst)*(*i);
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
int StripCPEgeometric::WrappedCluster::
sign() const
{ return ( *first < last() ) ? 1  : -1; }

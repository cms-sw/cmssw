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
    noise_threshold(conf.getParameter<double>("NoiseThreshold"))
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

  const std::vector<float> Q = InverseCrosstalkMatrix::unfold( cluster.amplitudes(), crosstalk[p.subdet] );

  const uncertain_t projection = find_projection( p, track, pos );

  const uncertain_t offset = offset_from_firstStrip( Q, projection );

  const float corrected = p.driftCorrected( cluster.firstStrip() + offset() , pos );

  return std::make_pair( p.topology->localPosition( corrected ),
			 p.topology->localError( corrected, offset.err2 ) );
}


StripCPEgeometric::uncertain_t StripCPEgeometric::
find_projection(const StripCPE::Param& p, const LocalVector& track, const LocalPoint& position) const {
  const float projection = fabs( p.coveredStrips( track+p.drift, position ));
  const float minProjection = 2*p.thickness*tan_diffusion_angle/p.topology->localPitch(position);
  const float projection_rel_err2 = thickness_rel_err2 + p.pitch_rel_err2;
  return uncertain_t::from_relative_uncertainty2( std::max(projection, minProjection), projection_rel_err2);
}


StripCPEgeometric::uncertain_t StripCPEgeometric::
offset_from_firstStrip( const std::vector<float>& Q, const uncertain_t& proj) const {
  WrappedCluster wc(Q);
  while( useNMinusOne( wc, proj) ) 
    wc.dropSmallerEdgeStrip();

  if( proj() < wc.N-2)                               return uncertain_t( wc.middle(),       wc.N*wc.N/12.);
  if( proj() > wc.maxProjection() || wc.deformed() ) return uncertain_t( wc.centroid(),             1/12.);

  const uncertain_t eta = wc.eta();
  const float sigma2 = 
    wc.N==1 ? 
    pow( 1-0.82*proj(), 2 ) / 12 :   
    ( pow(eta(),2) * proj.err2 + pow(proj(),2) * eta.err2 )/4 ;

  return uncertain_t( wc.middle() +0.5*proj()*eta(), sigma2 );
}


inline
bool StripCPEgeometric::
useNMinusOne(const WrappedCluster& wc, const uncertain_t& proj) const {
  if( proj() > wc.N-1 || 
      proj() < wc.N-5) return false;
  if( proj() < wc.N-2) return true;

  WrappedCluster wcTest(wc); wcTest.dropSmallerEdgeStrip();
  if( proj() >= wcTest.maxProjection() ) return false;
  if( wc.eta()() > 1./(wc.N-1) ) return true;
  if( wc.N==2 || wc.N==3)  
    return wc.smallerEdgeCharge() < noise_threshold;
  return fabs(  wcTest.dedxRatio(proj())-1 )   <   fabs(  wc.dedxRatio(proj())-1 ); 
}


StripCPEgeometric::WrappedCluster::
WrappedCluster(const std::vector<float>& Q) : 
  N(Q.size()),
  Qbegin(Q.begin()),
  first(Q.begin()),
  sumQ(accumulate(first, first+N, float(0)))
  {}

inline
void StripCPEgeometric::WrappedCluster::
dropSmallerEdgeStrip() {
  if(*first == last())  { sumQ-= *first; first++; 
                          sumQ-= last();          N-=2; } 
  else if(*first<last()){ sumQ-= *first; first++; N-=1; }
  else                  { sumQ-= last();          N-=1; }
}

inline
float StripCPEgeometric::WrappedCluster::
middle() const 
{ return (first-Qbegin) + N/2.;}

inline
float StripCPEgeometric::WrappedCluster::
centroid() const { 
  float sumXQ(0);
  for(std::vector<float>::const_iterator i = first; i<first+N; i++) sumXQ += (i-Qbegin)*(*i);
  return sumXQ/sumQ + 0.5;
}

inline
StripCPEgeometric::uncertain_t StripCPEgeometric::WrappedCluster::
eta() const 
{ return uncertain_t( (last()-*first) / sumQ , 
		      ( pow( last()-*first, 2) / sumQ + *first + last() ) / (sumQ*sumQ)  );}

inline
bool StripCPEgeometric::WrappedCluster::
deformed() const 
{  return  N>2  &&  std::max(*first,last()) > (sumQ-*first-last())/(N-2);}

inline
float StripCPEgeometric::WrappedCluster::
maxProjection() const
{ return N/(1+fabs(eta()())); }

inline
float StripCPEgeometric::WrappedCluster::
dedxRatio(const float& projection) const 
{ return ( sumQ/(*first+last()) - 1 ) * ( projection/(N-2) - 1 ); }

inline
float StripCPEgeometric::WrappedCluster::
smallerEdgeCharge() const 
{ return (*first<last())?  *first  : last(); }


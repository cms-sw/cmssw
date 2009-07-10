#ifndef RecoLocalTracker_SiStripRecHitConverter_StripCPEgeometric_H
#define RecoLocalTracker_SiStripRecHitConverter_StripCPEgeometric_H

#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/ErrorPropogationTypes.h"

class StripCPEgeometric : public StripCPE 
{

 public:
  
  StripClusterParameterEstimator::LocalValues 
    localParameters( const SiStripCluster&, const LocalTrajectoryParameters&) const; 

  StripClusterParameterEstimator::LocalValues 
    localParameters( const SiStripCluster&, const GeomDetUnit&, const LocalTrajectoryParameters&) const;

  StripCPEgeometric( edm::ParameterSet& conf, 
		     const MagneticField* mag, 
		     const TrackerGeometry* geom, 
		     const SiStripLorentzAngle* LorentzAngle);
 private:

  class WrappedCluster {
  public:
    WrappedCluster(const std::vector<stats_t<float> >&);
    void dropSmallerEdgeStrip();
    void addSuppressedEdgeStrip();
    float middle() const;
    stats_t<float> centroid() const;
    stats_t<float> sumQ() const;
    stats_t<float> eta() const;
    bool deformed() const;
    stats_t<float> maxProjection() const;
    stats_t<float> smallerEdgeStrip() const;
    int sign() const;
    uint16_t N;
  private:
    const stats_t<float>& last() const {return *(first+N-1);}
    std::vector<stats_t<float> >::const_iterator clusterFirst, first;
  };

  stats_t<float> find_projection(const StripCPE::Param&, const LocalVector&, const LocalPoint&) const;
  stats_t<float> offset_from_firstStrip( const std::vector<stats_t<float> >&, const stats_t<float>&) const;
  stats_t<float> geometric_position(const WrappedCluster&, const stats_t<float>&) const;
  bool useNPlusOne(const WrappedCluster&, const stats_t<float>&) const;
  bool useNMinusOne(const WrappedCluster&, const stats_t<float>&) const;
  bool ambiguousSize(const WrappedCluster&, const stats_t<float>&) const;

  std::vector<float> crosstalk;
  const float tan_diffusion_angle, thickness_rel_err2, noise_threshold, maybe_noise_threshold, scaling_squared, minimum_uncertainty_squared;

};

#endif

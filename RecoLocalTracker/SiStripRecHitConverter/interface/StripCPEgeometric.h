#ifndef RecoLocalTracker_SiStripRecHitConverter_StripCPEgeometric_H
#define RecoLocalTracker_SiStripRecHitConverter_StripCPEgeometric_H

#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"

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
  
  struct uncertain_t {
    uncertain_t(float v, float e2) : v(v), err2(e2) {}
    float v, err2;
    float operator()() const {return v;}
    static uncertain_t from_relative_uncertainty2(float v, float r2) {return uncertain_t(v, v*v*r2);}
  };
  
  class WrappedCluster {
  public:
    WrappedCluster(const std::vector<float>&);
    void dropSmallerEdgeStrip();
    float middle() const;
    float centroid() const;
    uncertain_t eta() const;
    bool deformed() const;
    float maxProjection() const;
    float dedxRatio(const float&) const;
    float smallerEdgeCharge() const;
    uint16_t N;
  private:
    const float& last() const {return *(first+N-1);}
    std::vector<float>::const_iterator Qbegin, first;
    float sumQ;
  };

  uncertain_t offset_from_firstStrip( const std::vector<float>&, const uncertain_t&) const;
  uncertain_t find_projection(const StripCPE::Param&, const LocalVector&, const LocalPoint&) const;
  bool useNMinusOne(const WrappedCluster&, const uncertain_t&) const;

  std::vector<float> crosstalk;
  const float tan_diffusion_angle, thickness_rel_err2, noise_threshold;

};

#endif

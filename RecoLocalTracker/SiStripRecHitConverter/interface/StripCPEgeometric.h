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

  class WrappedCluster {
   public:
    WrappedCluster(const SiStripCluster&,const std::vector<float>&);
    uint16_t N;
    SiStripDetId::SubDetector type;
    float eta() const;
    float etaErr2() const;
    float maxProjection() const;
    float middle() const;
    float dedxRatio(const float&) const;
    float smallerEdgeCharge() const;
    float centroid() const;
    bool deformed() const;
    void dropSmallerEdgeStrip();
   private:
    std::vector<float> Q;
    std::vector<float>::const_iterator first, last;
    uint16_t firstStrip;
    float sumQ;
  };

  std::pair<float,float> strip_stripErrorSquared( const SiStripCluster&, const float&, const float&) const;
  bool isMultiPeaked(const SiStripCluster&, const float&) const;
  bool useNMinusOne(const WrappedCluster&, const float&) const;
  float mix(const float&, const float&, const float&) const;

  std::vector<float> crosstalk;
  std::vector<float> edgeRatioCut;
  const float invsqrt12,tandriftangle,thickness_RelErr2,noise_threshold;
  const unsigned crossoverRate;
};

#endif

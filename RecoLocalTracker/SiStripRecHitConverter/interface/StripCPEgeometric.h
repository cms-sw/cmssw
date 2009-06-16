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
    WrappedCluster(const SiStripCluster&);
    uint16_t N;
    SiStripDetId::SubDetector type;
    float eta(const std::vector<float>&) const;
    float maxProjection(const std::vector<float>&) const;
    float middle() const;
    float dedxRatio(const float&) const;
    float smallEdgeRatio() const;
    float centroid() const;
    bool deformed() const;
    void dropSmallerEdgeStrip();
   private:
    vector<uint8_t>::const_iterator first, last;
    uint16_t firstStrip;
    float sumQ;
  };

  std::pair<float,float> strip_stripErrorSquared( const SiStripCluster&, const float&) const;
  bool isMultiPeaked(const SiStripCluster&, const float&) const;
  bool useNMinusOne(const WrappedCluster&, const float&) const;
  float mix(const float&, const float&, const float&) const;
  struct edgeRatioFromCrosstalk{
    edgeRatioFromCrosstalk(float s) : xtalksigma(s) {}
    float operator()(float xtalk) {float y=xtalk+xtalksigma;  return y/(1-2*y);}
    const float xtalksigma;
  };

  std::vector<float> crosstalk;
  std::vector<float> edgeRatioCut;
  const float invsqrt12, crosstalksigma,tandriftangle;
  const unsigned crossoverRate;
};

#endif

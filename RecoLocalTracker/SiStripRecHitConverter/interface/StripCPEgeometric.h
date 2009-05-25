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
		     const SiStripLorentzAngle* LorentzAngle)
    : StripCPE(conf, mag, geom, LorentzAngle ),
    invsqrt12(1/sqrt(12)),
    crossoverRate(15)
    {
      edgeRatioCut.resize(7,0);
      edgeRatioCut[SiStripDetId::TIB] = 0.058;
      edgeRatioCut[SiStripDetId::TID] = 0.111;
      edgeRatioCut[SiStripDetId::TOB] = 0.090;
      edgeRatioCut[SiStripDetId::TEC] = 0.111;

      tandriftangle.resize(7,0);
      tandriftangle[SiStripDetId::TIB] = 0.01;
      tandriftangle[SiStripDetId::TID] = 0.01;
      tandriftangle[SiStripDetId::TOB] = 0.01;
      tandriftangle[SiStripDetId::TEC] = 0.01;
    }

 private:

  class WrappedCluster {
   public:
    WrappedCluster(const SiStripCluster&);
    uint16_t N;
    SiStripDetId::SubDetector type;
    float eta() const;
    float middle() const;
    float dedxRatio(const float&) const;
    float smallEdgeRatio() const;
    float centroid() const;
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

  std::vector<float> edgeRatioCut;
  std::vector<float> tandriftangle;
  const float invsqrt12;
  const unsigned crossoverRate;
};

#endif

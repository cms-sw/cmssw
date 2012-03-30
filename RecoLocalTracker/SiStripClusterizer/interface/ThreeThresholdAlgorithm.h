#ifndef RecoLocalTracker_SiStripClusterizer_ThreeThresholdAlgorithm_h
#define RecoLocalTracker_SiStripClusterizer_ThreeThresholdAlgorithm_h
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithm.h"

class ThreeThresholdAlgorithm : public StripClusterizerAlgorithm {

  friend class StripClusterizerAlgorithmFactory;

 public:

  void clusterizeDetUnit(const    edm::DetSet<SiStripDigi> &, output_t::FastFiller &);
  void clusterizeDetUnit(const edmNew::DetSet<SiStripDigi> &, output_t::FastFiller &);

  bool stripByStripBegin(uint32_t id);
  void stripByStripAdd(uint16_t strip, uint16_t adc, std::vector<SiStripCluster>& out);
  void stripByStripEnd(std::vector<SiStripCluster>& out);

 private:

  template<class T> void clusterizeDetUnit_(const T&, output_t::FastFiller&);
  ThreeThresholdAlgorithm(float, float, float, unsigned, unsigned, unsigned, std::string qualityLabel,
			  bool setDetId);

  //state of the candidate cluster
  std::vector<uint16_t> ADCs;  
  uint16_t lastStrip;
  float noiseSquared;
  bool candidateLacksSeed;

  //constant methods with state information
  uint16_t firstStrip() const {return lastStrip - ADCs.size() + 1;}
  bool candidateEnded(const uint16_t&) const;
  bool candidateAccepted() const;

  //state modification methods
  template<class T> void endCandidate(T&);
  void clearCandidate() { candidateLacksSeed = true;  noiseSquared = 0;  ADCs.clear();}
  void addToCandidate(const SiStripDigi&);
  void appendBadNeighbors();
  void applyGains();

  float ChannelThreshold, SeedThreshold, ClusterThresholdSquared;
  uint8_t MaxSequentialHoles, MaxSequentialBad, MaxAdjacentBad;
};

#endif

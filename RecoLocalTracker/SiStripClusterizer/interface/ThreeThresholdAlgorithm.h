#ifndef RecoLocalTracker_SiStripClusterizer_ThreeThresholdAlgorithm_h
#define RecoLocalTracker_SiStripClusterizer_ThreeThresholdAlgorithm_h
#include "RecoLocalTracker/SiStripClusterizer/interface/StripClusterizerAlgorithm.h"
#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripApvShotCleaner.h"

class ThreeThresholdAlgorithm final : public StripClusterizerAlgorithm {
  friend class StripClusterizerAlgorithmFactory;

public:
  using State = StripClusterizerAlgorithm::State;
  using Det = StripClusterizerAlgorithm::Det;
  void clusterizeDetUnit(const edm::DetSet<SiStripDigi>&, output_t::TSFastFiller&) const override;
  void clusterizeDetUnit(const edmNew::DetSet<SiStripDigi>&, output_t::TSFastFiller&) const override;

  Det stripByStripBegin(uint32_t id) const override;

  // LazyGetter interface
  void stripByStripAdd(State& state, uint16_t strip, uint8_t adc, std::vector<SiStripCluster>& out) const override;
  void stripByStripEnd(State& state, std::vector<SiStripCluster>& out) const override;

  void stripByStripAdd(State& state, uint16_t strip, uint8_t adc, output_t::TSFastFiller& out) const override {
    if (candidateEnded(state, strip))
      endCandidate(state, out);
    addToCandidate(state, strip, adc);
  }

  void stripByStripEnd(State& state, output_t::TSFastFiller& out) const override { endCandidate(state, out); }

private:
  template <class T>
  void clusterizeDetUnit_(const T&, output_t::TSFastFiller&) const;

  ThreeThresholdAlgorithm(float,
                          float,
                          float,
                          unsigned,
                          unsigned,
                          unsigned,
                          std::string qualityLabel,
                          bool removeApvShots,
                          float minGoodCharge);

  //constant methods with state information
  uint16_t firstStrip(State const& state) const { return state.lastStrip - state.ADCs.size() + 1; }
  bool candidateEnded(State const& state, const uint16_t&) const;
  bool candidateAccepted(State const& state) const;

  //state modification methods
  template <class T>
  void endCandidate(State& state, T&) const;
  void clearCandidate(State& state) const {
    state.candidateLacksSeed = true;
    state.noiseSquared = 0;
    state.ADCs.clear();
  }
  void addToCandidate(State& state, const SiStripDigi& digi) const { addToCandidate(state, digi.strip(), digi.adc()); }
  void addToCandidate(State& state, uint16_t strip, uint8_t adc) const;
  void appendBadNeighbors(State& state) const;
  void applyGains(State& state) const;

  float ChannelThreshold, SeedThreshold, ClusterThresholdSquared;
  uint8_t MaxSequentialHoles, MaxSequentialBad, MaxAdjacentBad;
  bool RemoveApvShots;
  float minGoodCharge;
};

#endif

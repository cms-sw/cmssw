#include "RecoLocalTracker/SiStripClusterizer/interface/ThreeThresholdAlgorithm.h"

ThreeThresholdAlgorithm::ThreeThresholdAlgorithm(
    const edm::ESGetToken<SiStripClusterizerConditions, SiStripClusterizerConditionsRcd>& conditionsToken,
    float chan,
    float seed,
    float cluster,
    unsigned holes,
    unsigned bad,
    unsigned adj,
    unsigned maxClusterSize,
    bool removeApvShots,
    float minGoodCharge)
    : StripClusterizerAlgorithm(conditionsToken),
      ChannelThreshold(chan),
      SeedThreshold(seed),
      ClusterThresholdSquared(cluster * cluster),
      MaxSequentialHoles(holes),
      MaxSequentialBad(bad),
      MaxAdjacentBad(adj),
      MaxClusterSize(maxClusterSize),
      RemoveApvShots(removeApvShots),
      minGoodCharge(minGoodCharge) {}

void ThreeThresholdAlgorithm::clusterizeDetUnit(const edm::DetSet<SiStripDigi>& digis,
                                                output_t::TSFastFiller& output) const {
  clusterizeDetUnit_(digis, output);
}

void ThreeThresholdAlgorithm::stripByStripAdd(State& state,
                                              uint16_t strip,
                                              uint8_t adc,
                                              std::vector<SiStripCluster>& out) const {
  if (candidateEnded(state, strip))
    endCandidate(state, out);
  addToCandidate(state, SiStripDigi(strip, adc));
}

void ThreeThresholdAlgorithm::stripByStripEnd(State& state, std::vector<SiStripCluster>& out) const {
  endCandidate(state, out);
}

#include "RecoLocalTracker/SiStripClusterizer/interface/ThreeThresholdAlgorithm.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include <cmath>
#include <numeric>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"

ThreeThresholdAlgorithm::ThreeThresholdAlgorithm(
    const edm::ESGetToken<SiStripClusterizerConditions, SiStripClusterizerConditionsRcd>& conditionsToken,
    float chan,
    float seed,
    float cluster,
    unsigned holes,
    unsigned bad,
    unsigned adj,
    bool removeApvShots,
    float minGoodCharge)
    : StripClusterizerAlgorithm(conditionsToken),
      ChannelThreshold(chan),
      SeedThreshold(seed),
      ClusterThresholdSquared(cluster * cluster),
      MaxSequentialHoles(holes),
      MaxSequentialBad(bad),
      MaxAdjacentBad(adj),
      RemoveApvShots(removeApvShots),
      minGoodCharge(minGoodCharge) {}

template <class digiDetSet>
inline void ThreeThresholdAlgorithm::clusterizeDetUnit_(const digiDetSet& digis, output_t::TSFastFiller& output) const {
  const auto& cond = conditions();
  if (cond.isModuleBad(digis.detId()))
    return;

  auto const& det = cond.findDetId(digis.detId());
  if (!det.valid())
    return;

#ifdef EDM_ML_DEBUG
  if (!cond.isModuleUsable(digis.detId()))
    edm::LogWarning("ThreeThresholdAlgorithm") << " id " << digis.detId() << " not usable???" << std::endl;
#endif

  typename digiDetSet::const_iterator scan(digis.begin()), end(digis.end());

  SiStripApvShotCleaner ApvCleaner;
  if (RemoveApvShots) {
    ApvCleaner.clean(digis, scan, end);
  }

  output.reserve(16);
  State state(det);
  while (scan != end) {
    while (scan != end && !candidateEnded(state, scan->strip()))
      addToCandidate(state, *scan++);
    endCandidate(state, output);
  }
}

inline bool ThreeThresholdAlgorithm::candidateEnded(State const& state, const uint16_t& testStrip) const {
  uint16_t holes = testStrip - state.lastStrip - 1;
  return (((!state.ADCs.empty()) &       // a candidate exists, and
           (holes > MaxSequentialHoles)  // too many holes if not all are bad strips, and
           ) &&
          (holes > MaxSequentialBad ||                             // (too many bad strips anyway, or
           !state.det().allBadBetween(state.lastStrip, testStrip)  // not all holes are bad strips)
           ));
}

inline void ThreeThresholdAlgorithm::addToCandidate(State& state, uint16_t strip, uint8_t adc) const {
  float Noise = state.det().noise(strip);
  if (adc < static_cast<uint8_t>(Noise * ChannelThreshold) || state.det().bad(strip))
    return;

  if (state.candidateLacksSeed)
    state.candidateLacksSeed = adc < static_cast<uint8_t>(Noise * SeedThreshold);
  if (state.ADCs.empty())
    state.lastStrip = strip - 1;  // begin candidate
  while (++state.lastStrip < strip)
    state.ADCs.push_back(0);  // pad holes

  state.ADCs.push_back(adc);
  state.noiseSquared += Noise * Noise;
}

template <class T>
inline void ThreeThresholdAlgorithm::endCandidate(State& state, T& out) const {
  if (candidateAccepted(state)) {
    applyGains(state);
    if (MaxAdjacentBad > 0)
      appendBadNeighbors(state);
    if (minGoodCharge <= 0 ||
        siStripClusterTools::chargePerCM(state.det().detId, state.ADCs.begin(), state.ADCs.end()) > minGoodCharge)
      out.push_back(std::move(SiStripCluster(firstStrip(state), state.ADCs.begin(), state.ADCs.end())));
  }
  clearCandidate(state);
}

inline bool ThreeThresholdAlgorithm::candidateAccepted(State const& state) const {
  return (!state.candidateLacksSeed &&
          state.noiseSquared * ClusterThresholdSquared <=
              std::pow(float(std::accumulate(state.ADCs.begin(), state.ADCs.end(), int(0))), 2.f));
}

inline void ThreeThresholdAlgorithm::applyGains(State& state) const {
  uint16_t strip = firstStrip(state);
  for (auto& adc : state.ADCs) {
#ifdef EDM_ML_DEBUG
    // if(adc > 255) throw InvalidChargeException( SiStripDigi(strip,adc) );
#endif
    // if(adc > 253) continue; //saturated, do not scale
    auto charge = int(float(adc) * state.det().weight(strip++) + 0.5f);  //adding 0.5 turns truncation into rounding
    if (adc < 254)
      adc = (charge > 1022 ? 255 : (charge > 253 ? 254 : charge));
  }
}

inline void ThreeThresholdAlgorithm::appendBadNeighbors(State& state) const {
  uint8_t max = MaxAdjacentBad;
  while (0 < max--) {
    if (state.det().bad(firstStrip(state) - 1)) {
      state.ADCs.insert(state.ADCs.begin(), 0);
    }
    if (state.det().bad(state.lastStrip + 1)) {
      state.ADCs.push_back(0);
      state.lastStrip++;
    }
  }
}

void ThreeThresholdAlgorithm::clusterizeDetUnit(const edm::DetSet<SiStripDigi>& digis,
                                                output_t::TSFastFiller& output) const {
  clusterizeDetUnit_(digis, output);
}
void ThreeThresholdAlgorithm::clusterizeDetUnit(const edmNew::DetSet<SiStripDigi>& digis,
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

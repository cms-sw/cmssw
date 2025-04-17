#ifndef CalibFormats_SiStripObjects_StripClusterizerConditions_h
#define CalibFormats_SiStripObjects_StripClusterizerConditions_h

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"

/**
 * Cache of the noise and quality ranges per module, and the 1/gain value for each APV, for fast access by the clusterizer
 */
class SiStripClusterizerConditions {
public:
  static constexpr unsigned short invalidI = std::numeric_limits<unsigned short>::max();

  struct Det {
    bool valid() const { return ind != invalidI; }
    uint16_t rawNoise(const uint16_t strip) const { return SiStripNoises::getRawNoise(strip, noiseRange); }
    float noise(const uint16_t strip) const { return SiStripNoises::getNoise(strip, noiseRange); }
    float weight(const uint16_t strip) const { return m_weight[strip / 128]; }
    bool bad(const uint16_t strip) const { return quality->IsStripBad(qualityRange, strip); }
    bool allBadBetween(uint16_t L, const uint16_t& R) const {
      while (++L < R && bad(L)) {
      };
      return L == R;
    }
    SiStripQuality const* quality;
    SiStripNoises::Range noiseRange;
    SiStripQuality::Range qualityRange;
    float m_weight[6];
    uint32_t detId = 0;
    unsigned short ind = invalidI;
  };

  explicit SiStripClusterizerConditions(const SiStripQuality* quality) : m_quality(quality) {}

  std::vector<uint32_t> const& allDetIds() const { return m_detIds; }
  auto const& allDets() const { return m_dets; }

  std::vector<const FedChannelConnection*> const& currentConnection(const Det& det) const {
    return m_connections[det.ind];
  }

  Det const& findDetId(const uint32_t id) const {
    auto b = m_detIds.begin();
    auto e = m_detIds.end();
    auto p = std::lower_bound(b, e, id);
    if (p == e || id != (*p)) {
#ifdef NOT_ON_MONTECARLO
      edm::LogWarning("StripClusterizerAlgorithm")
          << "id " << id << " not connected. this is impossible on data " << std::endl;
#endif
      static const Det dummy = Det();
      return dummy;
    }
    return m_dets[p - m_detIds.begin()];
  }
  bool isModuleBad(const uint32_t id) const { return m_quality->IsModuleBad(id); }
  bool isModuleUsable(const uint32_t id) const { return m_quality->IsModuleUsable(id); }

  void reserve(std::size_t length) {
    m_detIds.reserve(length);
    m_dets.reserve(length);
    m_connections.reserve(length);
  }
  void emplace_back(uint32_t id,
                    SiStripQuality::Range qualityRange,
                    SiStripNoises::Range noiseRange,
                    const std::vector<float>& invGains,
                    const std::vector<const FedChannelConnection*>& connections) {
    const unsigned short index = m_detIds.size();
    m_detIds.push_back(id);
    auto& det = m_dets.emplace_back();
    det.quality = m_quality;
    det.qualityRange = qualityRange;
    det.noiseRange = noiseRange;
    for (uint32_t i = 0; i != invGains.size(); ++i) {
      det.m_weight[i] = invGains[i];
    }
    det.detId = id;
    det.ind = index;
    m_connections.push_back(connections);
  }

private:
  const SiStripQuality* m_quality;
  std::vector<uint32_t> m_detIds;
  std::vector<Det> m_dets;
  std::vector<std::vector<const FedChannelConnection*>> m_connections;
};

#endif  // CalibFormats_SiStripObjects_StripClusterizerConditions_h

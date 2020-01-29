#ifndef RecoLocalTracker_StripClusterizerAlgorithm_h
#define RecoLocalTracker_StripClusterizerAlgorithm_h

namespace edm {
  class EventSetup;
}
class SiStripDigi;
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBuffer.h"
#include <limits>

class FedChannelConnection;

class StripClusterizerAlgorithm {
public:
  static constexpr unsigned short invalidI = std::numeric_limits<unsigned short>::max();

  // state of detID
  struct Det {
    bool valid() const { return ind != invalidI; }
    float noise(const uint16_t& strip) const { return SiStripNoises::getNoise(strip, noiseRange); }
    float gain(const uint16_t& strip) const { return SiStripGain::getStripGain(strip, gainRange); }
    bool bad(const uint16_t& strip) const { return quality->IsStripBad(qualityRange, strip); }
    bool allBadBetween(uint16_t L, const uint16_t& R) const {
      while (++L < R && bad(L)) {
      };
      return L == R;
    }
    SiStripQuality const* quality;
    SiStripApvGain::Range gainRange;
    SiStripNoises::Range noiseRange;
    SiStripQuality::Range qualityRange;
    uint32_t detId = 0;
    unsigned short ind = invalidI;
  };

  //state of the candidate cluster
  struct State {
    State(Det const& idet) : m_det(idet) { ADCs.reserve(128); }
    Det const& det() const { return m_det; }
    std::vector<uint8_t> ADCs;
    uint16_t lastStrip = 0;
    float noiseSquared = 0;
    bool candidateLacksSeed = true;

  private:
    Det const& m_det;
  };

  virtual ~StripClusterizerAlgorithm() {}
  virtual void initialize(const edm::EventSetup&);

  //Offline DetSet interface
  typedef edmNew::DetSetVector<SiStripCluster> output_t;
  void clusterize(const edm::DetSetVector<SiStripDigi>&, output_t&) const;
  void clusterize(const edmNew::DetSetVector<SiStripDigi>&, output_t&) const;
  virtual void clusterizeDetUnit(const edm::DetSet<SiStripDigi>&, output_t::TSFastFiller&) const = 0;
  virtual void clusterizeDetUnit(const edmNew::DetSet<SiStripDigi>&, output_t::TSFastFiller&) const = 0;

  //HLT stripByStrip interface
  virtual Det stripByStripBegin(uint32_t id) const = 0;

  virtual void stripByStripAdd(State& state, uint16_t strip, uint8_t adc, std::vector<SiStripCluster>& out) const {}
  virtual void stripByStripEnd(State& state, std::vector<SiStripCluster>& out) const {}

  virtual void stripByStripAdd(State& state, uint16_t strip, uint8_t adc, output_t::TSFastFiller& out) const {}
  virtual void stripByStripEnd(State& state, output_t::TSFastFiller& out) const {}

  struct InvalidChargeException : public cms::Exception {
  public:
    InvalidChargeException(const SiStripDigi&);
  };

  SiStripDetCabling const* cabling() const { return theCabling; }
  std::vector<uint32_t> const& allDetIds() const { return detIds; }

  std::vector<const FedChannelConnection*> const& currentConnection(const Det& det) const {
    return connections[det.ind];
  }

protected:
  StripClusterizerAlgorithm() : qualityLabel(""), noise_cache_id(0), gain_cache_id(0), quality_cache_id(0) {}

  Det findDetId(const uint32_t) const;
  bool isModuleBad(const uint32_t& id) const { return qualityHandle->IsModuleBad(id); }
  bool isModuleUsable(const uint32_t& id) const { return qualityHandle->IsModuleUsable(id); }

  std::string qualityLabel;

private:
  template <class T>
  void clusterize_(const T& input, output_t& output) const {
    for (typename T::const_iterator it = input.begin(); it != input.end(); it++) {
      output_t::TSFastFiller ff(output, it->detId());
      clusterizeDetUnit(*it, ff);
      if (ff.empty())
        ff.abort();
    }
  }

  struct Index {
    unsigned short gi = invalidI, ni = invalidI, qi = invalidI;
  };
  std::vector<uint32_t> detIds;  // from cabling (connected and not bad)
  std::vector<std::vector<const FedChannelConnection*> > connections;
  std::vector<Index> indices;
  edm::ESHandle<SiStripGain> gainHandle;
  edm::ESHandle<SiStripNoises> noiseHandle;
  edm::ESHandle<SiStripQuality> qualityHandle;
  SiStripDetCabling const* theCabling = nullptr;
  uint32_t noise_cache_id, gain_cache_id, quality_cache_id;
};
#endif

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
#include <limits>

#include "RecoLocalTracker/Records/interface/SiStripClusterizerConditionsRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditions.h"

class StripClusterizerAlgorithm {
public:
  using Det = SiStripClusterizerConditions::Det;

  //state of the candidate cluster
  struct State {
    State(Det const& idet) : m_det(idet) { ADCs.reserve(8); }
    Det const& det() const { return m_det; }
    std::vector<uint8_t> ADCs;
    uint16_t lastStrip = 0;
    float noiseSquared = 0;
    bool candidateLacksSeed = true;

  private:
    Det const& m_det;
  };

  virtual ~StripClusterizerAlgorithm() {}

  void initialize(const edm::EventSetup& es) {
    es.get<SiStripClusterizerConditionsRcd>().get(m_conditionsLabel, m_conditionsHandle);
  }
  const SiStripClusterizerConditions& conditions() const { return *m_conditionsHandle; }

  //Offline DetSet interface
  typedef edmNew::DetSetVector<SiStripCluster> output_t;
  void clusterize(const edm::DetSetVector<SiStripDigi>&, output_t&) const;
  void clusterize(const edmNew::DetSetVector<SiStripDigi>&, output_t&) const;
  virtual void clusterizeDetUnit(const edm::DetSet<SiStripDigi>&, output_t::TSFastFiller&) const {}
  virtual void clusterizeDetUnit(const edmNew::DetSet<SiStripDigi>&, output_t::TSFastFiller&) const {}

  //HLT stripByStrip interface
  Det const& stripByStripBegin(uint32_t id) const { return m_conditionsHandle->findDetId(id); }

  virtual void stripByStripAdd(State& state, uint16_t strip, uint8_t adc, std::vector<SiStripCluster>& out) const {}
  virtual void stripByStripEnd(State& state, std::vector<SiStripCluster>& out) const {}

  virtual void stripByStripAdd(State& state, uint16_t strip, uint8_t adc, output_t::TSFastFiller& out) const {}
  virtual void stripByStripEnd(State& state, output_t::TSFastFiller& out) const {}

  struct InvalidChargeException : public cms::Exception {
  public:
    InvalidChargeException(const SiStripDigi&);
  };

protected:
  explicit StripClusterizerAlgorithm(const std::string& conditionsLabel = "") : m_conditionsLabel(conditionsLabel) {}

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

  std::string m_conditionsLabel;
  edm::ESHandle<SiStripClusterizerConditions> m_conditionsHandle;
};
#endif

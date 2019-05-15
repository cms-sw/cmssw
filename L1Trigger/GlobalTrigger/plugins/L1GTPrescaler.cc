#include <array>
#include <cassert>
#include <memory>
#include <vector>

template <class T, std::size_t N>
std::array<T, N> make_array(std::vector<T> const &values) {
  assert(N == values.size());
  std::array<T, N> ret;
  std::copy(values.begin(), values.end(), ret.begin());
  return ret;
}

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class L1GTPrescaler : public edm::one::EDFilter<> {
public:
  L1GTPrescaler(edm::ParameterSet const &config);

  bool filter(edm::Event &event, edm::EventSetup const &setup) override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  const edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> m_l1ResultsToken;
  const std::array<double, 128> m_algoPrescales;
  const std::array<double, 64> m_techPrescales;
  std::array<unsigned int, 128> m_algoCounters;
  std::array<unsigned int, 64> m_techCounters;
};

L1GTPrescaler::L1GTPrescaler(edm::ParameterSet const &config)
    : m_l1ResultsToken(consumes<L1GlobalTriggerReadoutRecord>(config.getParameter<edm::InputTag>("l1Results"))),
      m_algoPrescales(make_array<double, 128>(config.getParameter<std::vector<double>>("l1AlgoPrescales"))),
      m_techPrescales(make_array<double, 64>(config.getParameter<std::vector<double>>("l1TechPrescales"))) {
  m_algoCounters.fill(0);
  m_techCounters.fill(0);
  produces<L1GlobalTriggerReadoutRecord>();
}

bool L1GTPrescaler::filter(edm::Event &event, edm::EventSetup const &setup) {
  edm::Handle<L1GlobalTriggerReadoutRecord> handle;
  event.getByToken(m_l1ResultsToken, handle);
  auto algoWord = handle->decisionWord();          // make a copy of the L1 algo results
  auto techWord = handle->technicalTriggerWord();  // make a copy of the L1 tech results
  bool finalOr = false;

  for (unsigned int i = 0; i < 128; ++i) {
    if (m_algoPrescales[i] == 0) {
      // mask this trigger: reset the bit
      algoWord[i] = false;
    } else if (algoWord[i]) {
      // prescale this trigger
      ++m_algoCounters[i];
      if (std::fmod(m_algoCounters[i], m_algoPrescales[i]) < 1)
        // the prescale is successful, keep the bit set
        finalOr = true;
      else
        // the prescale failed, reset the bit
        algoWord[i] = false;
    }
  }
  for (unsigned int i = 0; i < 64; ++i) {
    if (m_techPrescales[i] == 0) {
      // mask this trigger: reset the bit
      techWord[i] = false;
    } else if (techWord[i]) {
      ++m_techCounters[i];
      if (std::fmod(m_techCounters[i], m_techPrescales[i]) < 1)
        // the prescale is successful, keep the bit set
        finalOr = true;
      else
        // the prescale failed, reset the bit
        techWord[i] = false;
    }
  }

  // make a copy of the L1GlobalTriggerReadoutRecord, and set the new decisions
  std::unique_ptr<L1GlobalTriggerReadoutRecord> result(new L1GlobalTriggerReadoutRecord(*handle));
  result->setDecisionWord(algoWord);
  result->setTechnicalTriggerWord(techWord);
  result->setDecision(finalOr);
  event.put(std::move(result));

  return finalOr;
}

void L1GTPrescaler::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("l1Results", edm::InputTag("gtDigis"));
  desc.add<std::vector<double>>("l1AlgoPrescales", std::vector<double>(128, 1));
  desc.add<std::vector<double>>("l1TechPrescales", std::vector<double>(64, 1));
  descriptions.add("l1GTPrescaler", desc);
}

// register as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1GTPrescaler);

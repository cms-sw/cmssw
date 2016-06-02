#include <vector>
#include <array>
#include <memory>
#include <cassert>

template <class T, std::size_t N>
std::array<T, N> make_array(std::vector<T> const & values) {
  assert(N == values.size());
  std::array<T, N> ret;
  std::copy(values.begin(), values.end(), ret.begin());
  return ret;
}


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"

class L1TGlobalPrescaler : public edm::one::EDFilter<> {
public:
  L1TGlobalPrescaler(edm::ParameterSet const& config);

  virtual bool filter(edm::Event& event, edm::EventSetup const& setup) override;

  static  void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  const edm::EDGetTokenT<GlobalAlgBlkBxCollection> m_l1ResultsToken;
  const std::array<double, GlobalAlgBlk::maxPhysicsTriggers> m_prescales;
  std::array<unsigned int, GlobalAlgBlk::maxPhysicsTriggers> m_counters;

};

L1TGlobalPrescaler::L1TGlobalPrescaler(edm::ParameterSet const& config) :
  m_l1ResultsToken( consumes<GlobalAlgBlkBxCollection>(config.getParameter<edm::InputTag>("l1tResults")) ),
  m_prescales( make_array<double, GlobalAlgBlk::maxPhysicsTriggers>(config.getParameter<std::vector<double>>("l1tPrescales")) )
{
  m_counters.fill(0);
  produces<GlobalAlgBlkBxCollection>();
}

bool L1TGlobalPrescaler::filter(edm::Event& event, edm::EventSetup const& setup) {
  edm::Handle<GlobalAlgBlkBxCollection> handle;
  event.getByToken(m_l1ResultsToken, handle);

  // if the input collection does not have any information for bx 0,
  // produce an empty collection, and fail
  if (handle->isEmpty(0)) {
      std::unique_ptr<GlobalAlgBlkBxCollection> result(new GlobalAlgBlkBxCollection());
      event.put(std::move(result));
      return false;
  }

  // make a copy of the GlobalAlgBlk for bx 0
  GlobalAlgBlk algoBlock = handle->at(0,0);

  bool finalOr = false;

  for (unsigned int i = 0; i < GlobalAlgBlk::maxPhysicsTriggers; ++i) {
    if (m_prescales[i] == 0) {
      // mask this trigger: reset the bit
      algoBlock.setAlgoDecisionFinal(i, false);
    } else if (algoBlock.getAlgoDecisionFinal(i)) {
      // prescale this trigger
      ++m_counters[i];
      if (std::fmod(m_counters[i], m_prescales[i]) < 1)
        // the prescale is successful, keep the bit set
        finalOr = true;
      else
        // the prescale failed, reset the bit
        algoBlock.setAlgoDecisionFinal(i, false);
    }
  }

  // set the final OR
  algoBlock.setFinalORPreVeto(finalOr);
  if (algoBlock.getFinalORVeto())
    finalOr = false;
  algoBlock.setFinalOR(finalOr);

  // create a new GlobalAlgBlkBxCollection, and set the new prescaled decisions for bx 0
  std::unique_ptr<GlobalAlgBlkBxCollection> result(new GlobalAlgBlkBxCollection());
  result->push_back(0, algoBlock);
  event.put(std::move(result));

  return finalOr;
}

void L1TGlobalPrescaler::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("l1tResults", edm::InputTag("gtStage2Digis"));
  desc.add<std::vector<double>>("l1tPrescales", std::vector<double>(GlobalAlgBlk::maxPhysicsTriggers, 1.));
  descriptions.add("l1tGlobalPrescaler", desc);
}


// register as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TGlobalPrescaler);

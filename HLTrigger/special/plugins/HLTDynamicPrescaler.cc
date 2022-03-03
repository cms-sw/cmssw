#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

class HLTDynamicPrescaler : public edm::global::EDFilter<> {
public:
  explicit HLTDynamicPrescaler(edm::ParameterSet const& configuration);
  ~HLTDynamicPrescaler() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool filter(edm::StreamID, edm::Event& event, edm::EventSetup const& setup) const override;

private:
  mutable std::atomic<unsigned int> m_count;  // event counter
};

HLTDynamicPrescaler::HLTDynamicPrescaler(edm::ParameterSet const& configuration) : m_count(0) {}

HLTDynamicPrescaler::~HLTDynamicPrescaler() = default;

void HLTDynamicPrescaler::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("saveTags", false);
  descriptions.add("hltDynamicPrescaler", desc);
}

bool HLTDynamicPrescaler::filter(edm::StreamID, edm::Event& event, edm::EventSetup const& setup) const {
  auto count = ++m_count;

  unsigned int dynamicScale = 1;
  while (count > dynamicScale * 10) {
    dynamicScale *= 10;
  }

  return (0 == count % dynamicScale);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTDynamicPrescaler);

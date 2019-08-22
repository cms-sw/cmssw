#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

class HLTDynamicPrescaler : public edm::EDFilter {
public:
  explicit HLTDynamicPrescaler(edm::ParameterSet const& configuration);
  ~HLTDynamicPrescaler() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool filter(edm::Event& event, edm::EventSetup const& setup) override;

private:
  unsigned int m_count;  // event counter
  unsigned int m_scale;  // accept one event every m_scale, which will change dynamically
};

HLTDynamicPrescaler::HLTDynamicPrescaler(edm::ParameterSet const& configuration) : m_count(0), m_scale(1) {}

HLTDynamicPrescaler::~HLTDynamicPrescaler() = default;

void HLTDynamicPrescaler::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("saveTags", false);
  descriptions.add("hltDynamicPrescaler", desc);
}

bool HLTDynamicPrescaler::filter(edm::Event& event, edm::EventSetup const& setup) {
  ++m_count;

  if (m_count % m_scale)
    return false;

  if (m_count == m_scale * 10)
    m_scale = m_count;

  return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTDynamicPrescaler);

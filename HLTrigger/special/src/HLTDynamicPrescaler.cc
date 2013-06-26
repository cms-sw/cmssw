#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HLTDynamicPrescaler : public edm::EDFilter {
public:
  explicit HLTDynamicPrescaler(edm::ParameterSet const & configuration);
  ~HLTDynamicPrescaler();

  bool filter(edm::Event & event, edm::EventSetup const & setup);

private:
  unsigned int m_count;     // event counter
  unsigned int m_scale;     // accept one event every m_scale, which will change dynamically
};

HLTDynamicPrescaler::HLTDynamicPrescaler(edm::ParameterSet const & configuration) :
  m_count(0),
  m_scale(1) { 
}

HLTDynamicPrescaler::~HLTDynamicPrescaler() {
}

bool HLTDynamicPrescaler::filter(edm::Event & event, edm::EventSetup const & setup) {
  ++m_count;

  if (m_count % m_scale)
    return false;

  if (m_count == m_scale * 10)
    m_scale = m_count;
  
  return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTDynamicPrescaler);

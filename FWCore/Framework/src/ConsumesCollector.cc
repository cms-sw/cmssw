#include "FWCore/Framework/interface/ConsumesCollector.h"

namespace edm {

  ConsumesCollector::ConsumesCollector(ConsumesCollector const& other) : m_consumer(get_underlying(other.m_consumer)) {}

  ConsumesCollector& ConsumesCollector::operator=(ConsumesCollector const& other) {
    m_consumer = get_underlying(other.m_consumer);
    return *this;
  }

}  // namespace edm

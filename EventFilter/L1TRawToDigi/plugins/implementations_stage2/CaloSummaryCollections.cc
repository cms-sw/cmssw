#include "FWCore/Framework/interface/Event.h"

#include "CaloSummaryCollections.h"

namespace l1t {
  namespace stage2 {
    CaloSummaryCollections::~CaloSummaryCollections() { event_.put(std::move(cicadaDigis_)); }
  }  // namespace stage2
}  // namespace l1t

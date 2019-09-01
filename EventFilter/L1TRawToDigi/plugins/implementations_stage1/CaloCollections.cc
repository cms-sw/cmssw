#include "FWCore/Framework/interface/Event.h"

#include "CaloCollections.h"

namespace l1t {
  namespace stage1 {
    CaloCollections::~CaloCollections() {
      event_.put(std::move(towers_));
      event_.put(std::move(egammas_));
      event_.put(std::move(etsums_));
      event_.put(std::move(jets_));
      event_.put(std::move(taus_), "rlxTaus");
      event_.put(std::move(isotaus_), "isoTaus");
      event_.put(std::move(calospareHFBitCounts_), "HFBitCounts");
      event_.put(std::move(calospareHFRingSums_), "HFRingSums");
      event_.put(std::move(caloEmCands_));
      event_.put(std::move(caloRegions_));
    }
  }  // namespace stage1
}  // namespace l1t

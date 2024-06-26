#include "FWCore/Framework/interface/Event.h"

#include "CaloLayer1Collections.h"

namespace l1t {
  namespace stage2 {
    CaloLayer1Collections::~CaloLayer1Collections() {
      event_.put(std::move(ecalDigis_));
      event_.put(std::move(hcalDigis_));
      event_.put(std::move(caloRegions_));

      for (int i = 0; i < 5; ++i) {
        event_.put(std::move(ecalDigisBx_[i]), "EcalDigisBx" + std::to_string(i + 1));
      }
    }
  }  // namespace stage2
}  // namespace l1t

#include "FWCore/Framework/interface/Event.h"

#include "CaloLayer1Collections.h"

namespace l1t {
  namespace stage2 {
    CaloLayer1Collections::CaloLayer1Collections(edm::Event& e)
        : UnpackerCollections(e),
          ecalDigis_(new EcalTrigPrimDigiCollection()),
          hcalDigis_(new HcalTrigPrimDigiCollection()),
          caloRegions_(new L1CaloRegionCollection()) {
      // Pre-allocate:
      //  72 iPhi values
      //  28 iEta values in Ecal, 28 + 12 iEta values in Hcal + HF
      //  2 hemispheres
      ecalDigis_->reserve(72 * 28 * 2);
      hcalDigis_->reserve(72 * 40 * 2);
      // 7 regions * 18 cards * 2 hemispheres
      caloRegions_->reserve(7 * 18 * 2);
    }

    CaloLayer1Collections::~CaloLayer1Collections() {
      event_.put(std::move(ecalDigis_));
      event_.put(std::move(hcalDigis_));
      event_.put(std::move(caloRegions_));
    }
  }  // namespace stage2
}  // namespace l1t

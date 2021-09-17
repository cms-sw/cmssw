#ifndef CaloLayer1Collections_h
#define CaloLayer1Collections_h

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"
#include "L1TObjectCollections.h"

namespace l1t {
  namespace stage2 {
    class CaloLayer1Collections : public L1TObjectCollections {
    public:
      CaloLayer1Collections(edm::Event& e)
          : L1TObjectCollections(e),
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
        std::generate(
            ecalDigisBx_.begin(), ecalDigisBx_.end(), [] { return std::make_unique<EcalTrigPrimDigiCollection>(); });
      };

      ~CaloLayer1Collections() override;

      inline EcalTrigPrimDigiCollection* getEcalDigis() { return ecalDigis_.get(); };
      inline HcalTrigPrimDigiCollection* getHcalDigis() { return hcalDigis_.get(); };
      inline L1CaloRegionCollection* getRegions() { return caloRegions_.get(); };

      inline EcalTrigPrimDigiCollection* getEcalDigisBx(const unsigned int copy) override {
        return ecalDigisBx_[copy].get();
      };

    private:
      std::unique_ptr<EcalTrigPrimDigiCollection> ecalDigis_;
      std::unique_ptr<HcalTrigPrimDigiCollection> hcalDigis_;
      std::unique_ptr<L1CaloRegionCollection> caloRegions_;

      std::array<std::unique_ptr<EcalTrigPrimDigiCollection>, 5> ecalDigisBx_;
    };
  }  // namespace stage2
}  // namespace l1t

#endif

#ifndef L1T_PACKER_STAGE2_CALOLAYER1PACKER_H
#define L1T_PACKER_STAGE2_CALOLAYER1PACKER_H

#include "EventFilter/L1TRawToDigi/interface/Packer.h"
#include "CaloLayer1Tokens.h"
#include "UCTCTP7RawData.h"

namespace l1t {
  namespace stage2 {
    class CaloLayer1Packer : public Packer {
    public:
      Blocks pack(const edm::Event&, const PackerTokens*) override;

    private:
      void makeECalTPGs(uint32_t lPhi, UCTCTP7RawData& ctp7Data, const EcalTrigPrimDigiCollection* ecalTPGs);
      void makeHCalTPGs(uint32_t lPhi, UCTCTP7RawData& ctp7Data, const HcalTrigPrimDigiCollection* hcalTPGs);
      void makeHFTPGs(uint32_t lPhi, UCTCTP7RawData& ctp7Data, const HcalTrigPrimDigiCollection* hcalTPGs);
      void makeRegions(uint32_t lPhi, UCTCTP7RawData& ctp7Data, const L1CaloRegionCollection* regions);
    };
  }  // namespace stage2
}  // namespace l1t

#endif

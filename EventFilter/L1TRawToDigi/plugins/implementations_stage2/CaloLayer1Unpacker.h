#ifndef L1T_PACKER_STAGE2_LAYER1UNPACKER_H
#define L1T_PACKER_STAGE2_LAYER1UNPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"
#include "CaloLayer1Collections.h"
#include "UCTCTP7RawData.h"
#include "UCTCTP7RawData5BX.h"

namespace l1t {
  namespace stage2 {
    class CaloLayer1Unpacker : public Unpacker {
    public:
      bool unpack(const Block& block, UnpackerCollections* coll) override;

    private:
      void makeECalTPGs(uint32_t lPhi, UCTCTP7RawData& ctp7Data, EcalTrigPrimDigiCollection* ecalTPGs);
      void makeHCalTPGs(uint32_t lPhi, UCTCTP7RawData& ctp7Data, HcalTrigPrimDigiCollection* hcalTPGs);
      void makeHFTPGs(uint32_t lPhi, UCTCTP7RawData& ctp7Data, HcalTrigPrimDigiCollection* hcalTPGs);
      void makeRegions(uint32_t lPhi, UCTCTP7RawData& ctp7Data, L1CaloRegionCollection* regions);
      void makeECalTPGs5BX(uint32_t lPhi,
                           UCTCTP7RawData5BX& ctp7Data5BX,
                           EcalTrigPrimDigiCollection* ecalTPGs,
                           uint32_t BX_n);
      void makeHCalTPGs5BX(uint32_t lPhi,
                           UCTCTP7RawData5BX& ctp7Data5BX,
                           HcalTrigPrimDigiCollection* hcalTPGs,
                           uint32_t BX_n);
      void makeHFTPGs5BX(uint32_t lPhi,
                         UCTCTP7RawData5BX& ctp7Data5BX,
                         HcalTrigPrimDigiCollection* hcalTPGs,
                         uint32_t BX_n);
      void makeRegions5BX(uint32_t lPhi, UCTCTP7RawData5BX& ctp7Data5BX, L1CaloRegionCollection* regions, uint32_t BX_n);
    };
  }  // namespace stage2
}  // namespace l1t

#endif

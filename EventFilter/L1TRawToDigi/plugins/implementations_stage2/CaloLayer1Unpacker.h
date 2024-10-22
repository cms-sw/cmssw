#ifndef L1T_PACKER_STAGE2_LAYER1UNPACKER_H
#define L1T_PACKER_STAGE2_LAYER1UNPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"
#include "CaloLayer1Collections.h"
#include "UCTCTP7RawData.h"
#include "UCTCTP7RawData_HCALFB.h"
#include "UCTCTP7RawData5BX.h"
#include "UCTCTP7RawData5BX_HCALFB.h"

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
      void makeECalTPGs_HCALFB(uint32_t lPhi,
                               UCTCTP7RawData_HCALFB& ctp7Data_HCALFB,
                               EcalTrigPrimDigiCollection* ecalTPGs);
      void makeHCalTPGs_HCALFB(uint32_t lPhi,
                               UCTCTP7RawData_HCALFB& ctp7Data_HCALFB,
                               HcalTrigPrimDigiCollection* hcalTPGs);
      void makeHFTPGs_HCALFB(uint32_t lPhi,
                             UCTCTP7RawData_HCALFB& ctp7Data_HCALFB,
                             HcalTrigPrimDigiCollection* hcalTPGs);
      void makeRegions_HCALFB(uint32_t lPhi, UCTCTP7RawData_HCALFB& ctp7Data_HCALFB, L1CaloRegionCollection* regions);
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
      void makeECalTPGs5BX_HCALFB(uint32_t lPhi,
                                  UCTCTP7RawData5BX_HCALFB& ctp7Data5BX_HCALFB,
                                  EcalTrigPrimDigiCollection* ecalTPGs,
                                  uint32_t BX_n);
      void makeHCalTPGs5BX_HCALFB(uint32_t lPhi,
                                  UCTCTP7RawData5BX_HCALFB& ctp7Data5BX_HCALFB,
                                  HcalTrigPrimDigiCollection* hcalTPGs,
                                  uint32_t BX_n);
      void makeHFTPGs5BX_HCALFB(uint32_t lPhi,
                                UCTCTP7RawData5BX_HCALFB& ctp7Data5BX_HCALFB,
                                HcalTrigPrimDigiCollection* hcalTPGs,
                                uint32_t BX_n);
      void makeRegions5BX_HCALFB(uint32_t lPhi,
                                 UCTCTP7RawData5BX_HCALFB& ctp7Data5BX_HCALFB,
                                 L1CaloRegionCollection* regions,
                                 uint32_t BX_n);
    };
  }  // namespace stage2
}  // namespace l1t

#endif

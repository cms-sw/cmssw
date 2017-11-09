#include "FWCore/Framework/interface/Event.h"
#include "EventFilter/L1TRawToDigi/plugins/PackerFactory.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "CaloLayer1Packer.h"

namespace l1t {
namespace stage2 {
   
   // max_iEta_HcalTP = 41; // barrel <= 16, endcap <= 28, hf <= 41
   // there are two TT29’s: one in HE readout in TT28 and another in HF readout in TT30
   // max_iPhi_HcalTP = 72;
   
   Blocks CaloLayer1Packer::pack(const edm::Event& event, const PackerTokens* toks)
   {
      edm::Handle<EcalTrigPrimDigiCollection> ecalDigis;
      event.getByToken(static_cast<const CaloLayer1Tokens*>(toks)->getEcalDigiToken(), ecalDigis);
      edm::Handle<HcalTrigPrimDigiCollection> hcalDigis;
      event.getByToken(static_cast<const CaloLayer1Tokens*>(toks)->getHcalDigiToken(), hcalDigis);
      edm::Handle<L1CaloRegionCollection> caloRegions;
      event.getByToken(static_cast<const CaloLayer1Tokens*>(toks)->getCaloRegionToken(), caloRegions);

      std::vector<uint32_t> load;
      load.resize(192, 0u);

      auto ctp7_phi = board();
      uint32_t * ptr = &*load.begin();
      UCTCTP7RawData ctp7Data(ptr);
      makeECalTPGs(ctp7_phi, ctp7Data, ecalDigis.product());
      makeHCalTPGs(ctp7_phi, ctp7Data, hcalDigis.product());
      makeHFTPGs(ctp7_phi, ctp7Data, hcalDigis.product());
      makeRegions(ctp7_phi, ctp7Data, caloRegions.product());

      unsigned bx_per_l1a = 1;
      // CTP7 uses CMS scheme, starting at 0
      // TODO: expected +2, but +1 apparently?
      unsigned calo_bxid = (event.bunchCrossing()+1) % 3564;

      // a la CTP7Payload::getHeader()
      unsigned blockId = 0;
      unsigned blockSize = 192;
      unsigned capId = 0;
      unsigned blockFlags = ((bx_per_l1a&0xf)<<16) | (calo_bxid&0xfff);
      BlockHeader hdr(blockId, blockSize, capId, blockFlags, CTP7);
      Block block(hdr, &*load.begin(), &*load.end());

      Blocks res;
      res.push_back(block);
      return res;
   }

   void CaloLayer1Packer::makeECalTPGs(uint32_t lPhi, UCTCTP7RawData& ctp7Data, const EcalTrigPrimDigiCollection* ecalTPGs) {
      UCTCTP7RawData::CaloType cType = UCTCTP7RawData::EBEE;
      for(uint32_t iPhi = 0; iPhi < 4; iPhi++) { // Loop over all four phi divisions on card
         int cPhi = - 1 + lPhi * 4 + iPhi; // Calorimeter phi index
         if(cPhi == 0) cPhi = 72;
         else if(cPhi == -1) cPhi = 71;
         else if(cPhi < -1) {
           edm::LogError("CaloLayer1Packer") << "Major error in makeECalTPGs" << std::endl;
            return;
         }
         for(int cEta =   -28; cEta <= 28; cEta++) { // Calorimeter Eta indices (HB/HE for now)
            if(cEta != 0) { // Calorimeter eta = 0 is invalid
               bool negativeEta = false;
               if(cEta < 0) negativeEta = true;
               uint32_t iEta = abs(cEta);

               int zSide = cEta / ((int) iEta);
               const EcalSubdetector ecalTriggerTower = (iEta > 17 ) ? EcalSubdetector::EcalEndcap : EcalSubdetector::EcalBarrel;
               EcalTrigTowerDetId id(zSide, ecalTriggerTower, iEta, cPhi);
               const auto& tp = ecalTPGs->find(id);
               if ( tp != ecalTPGs->end() ) {
                  ctp7Data.setET(cType, negativeEta, iEta, iPhi, tp->compressedEt());
                  ctp7Data.setFB(cType, negativeEta, iEta, iPhi, tp->fineGrain());
               }
            }
         }
      }

   }

   void CaloLayer1Packer::makeHCalTPGs(uint32_t lPhi, UCTCTP7RawData& ctp7Data, const HcalTrigPrimDigiCollection* hcalTPGs) {
      UCTCTP7RawData::CaloType cType = UCTCTP7RawData::HBHE;
      for(uint32_t iPhi = 0; iPhi < 4; iPhi++) { // Loop over all four phi divisions on card
         int cPhi = - 1 + lPhi * 4 + iPhi; // Calorimeter phi index
         if(cPhi == 0) cPhi = 72;
         else if(cPhi == -1) cPhi = 71;
         else if(cPhi < -1) {
           edm::LogError("CaloLayer1Packer") << "Major error in makeHCalTPGs" << std::endl;
            return;
         }
         for(int cEta =   -28; cEta <= 28; cEta++) { // Calorimeter Eta indices (HB/HE for now)
            if(cEta != 0) { // Calorimeter eta = 0 is invalid
               bool negativeEta = false;
               if(cEta < 0) negativeEta = true;
               uint32_t iEta = abs(cEta);

               HcalTrigTowerDetId id(cEta, cPhi);
               const auto tp = hcalTPGs->find(id);
               if ( tp != hcalTPGs->end() ) {
                  ctp7Data.setET(cType, negativeEta, iEta, iPhi, tp->SOI_compressedEt());
                  ctp7Data.setFB(cType, negativeEta, iEta, iPhi, tp->SOI_fineGrain());
               }
            }
         }
      }

   }

   void CaloLayer1Packer::makeHFTPGs(uint32_t lPhi, UCTCTP7RawData& ctp7Data, const HcalTrigPrimDigiCollection* hcalTPGs) {
      UCTCTP7RawData::CaloType cType = UCTCTP7RawData::HF;
      for(uint32_t side = 0; side <= 1; side++) {
         bool negativeEta = false;
         if(side == 0) negativeEta = true;
         for(uint32_t iEta = 30; iEta <= 40; iEta++) {
            for(uint32_t iPhi = 0; iPhi < 2; iPhi++) {
               if(iPhi == 1 && iEta == 40) iEta = 41;
               int cPhi = 1 + lPhi * 4 + iPhi * 2; // Calorimeter phi index: 1, 3, 5, ... 71
               if(iEta == 41) cPhi -= 2; // Last two HF are 3, 7, 11, ...
               cPhi = (cPhi+69)%72 + 1; // cPhi -= 2 mod 72
               int cEta = iEta;
               if(negativeEta) cEta = -iEta;

               HcalTrigTowerDetId id(cEta, cPhi);
               id.setVersion(1); // To not process these 1x1 HF TPGs with RCT
               const auto tp = hcalTPGs->find(id);
               if ( tp != hcalTPGs->end() ) {
                  ctp7Data.setET(cType, negativeEta, iEta, iPhi, tp->SOI_compressedEt());
                  ctp7Data.setFB(cType, negativeEta, iEta, iPhi, ((tp->SOI_fineGrain(1)<<1) | tp->SOI_fineGrain(0)));
               }
            }
         }
      }
   }

   void 
   CaloLayer1Packer::makeRegions(uint32_t lPhi, UCTCTP7RawData& ctp7Data, const L1CaloRegionCollection* regions) {
      for(uint32_t side = 0; side <= 1; side++) {
         bool negativeEta = false;
         if(side == 0) negativeEta = true;
         for(uint32_t region = 0; region <= 6; region++) {
            uint32_t lEta = 10 - region; // GCT eta goes 0-21, 0-3 -HF, 4-10 -B/E, 11-17 +B/E, 18-21 +HF
            if(!negativeEta) lEta = region + 11;

            L1CaloRegionDetId id(lEta, lPhi);
            // Can't use find since not an edm::SortedCollection
            // const L1CaloRegion& rtp = *regions->find(id);
            for (const auto& rtp : *regions) {
               if ( rtp.id() == id ) {
                  ctp7Data.setRegionSummary(negativeEta, region, rtp.raw());
                  break;
               }
            }
         }
      }
   }


}
}

DEFINE_L1T_PACKER(l1t::stage2::CaloLayer1Packer);

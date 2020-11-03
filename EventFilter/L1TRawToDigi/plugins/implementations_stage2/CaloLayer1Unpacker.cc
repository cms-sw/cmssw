#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "CaloLayer1Unpacker.h"

using namespace edm;

namespace l1t {
  namespace stage2 {

    // max_iEta_HcalTP = 41; // barrel <= 16, endcap <= 29, hf <= 41
    // there are two TT29’s: one in HE readout in TT28 and another in HF readout in TT30
    // max_iPhi_HcalTP = 72;

    bool CaloLayer1Unpacker::unpack(const Block& block, UnpackerCollections* coll) {
      LogDebug("L1T") << "Block size = " << block.header().getSize();
      LogDebug("L1T") << "Board ID = " << block.amc().getBoardID();

      auto res = static_cast<CaloLayer1Collections*>(coll);

      auto ctp7_phi = block.amc().getBoardID();
      const uint32_t* ptr = block.payload().data();

      int N_BX = (block.header().getFlags() >> 16) & 0xf;
      //      std::cout << " N_BX calculated " << N_BX << std::endl;

      if (N_BX == 1) {
        UCTCTP7RawData ctp7Data(ptr);
        makeECalTPGs(ctp7_phi, ctp7Data, res->getEcalDigis());
        makeHCalTPGs(ctp7_phi, ctp7Data, res->getHcalDigis());
        makeHFTPGs(ctp7_phi, ctp7Data, res->getHcalDigis());
        makeRegions(ctp7_phi, ctp7Data, res->getRegions());
      } else if (N_BX == 5) {
        const uint32_t* ptr5 = ptr;
        UCTCTP7RawData ctp7Data(ptr);
        makeECalTPGs(ctp7_phi, ctp7Data, res->getEcalDigis());
        makeHCalTPGs(ctp7_phi, ctp7Data, res->getHcalDigis());
        makeHFTPGs(ctp7_phi, ctp7Data, res->getHcalDigis());
        makeRegions(ctp7_phi, ctp7Data, res->getRegions());
        for (int i = 0; i < 5; i++) {
          UCTCTP7RawData ctp7Data(ptr5);
          makeECalTPGs(ctp7_phi, ctp7Data, res->getEcalDigisBx(i));
          ptr5 += 192;
        }
      } else {
        LogError("CaloLayer1Unpacker") << "Number of BXs to unpack is not 1 or 5, stop here !!! " << N_BX << std::endl;
        return false;
      }

      return true;
    }

    void CaloLayer1Unpacker::makeECalTPGs(uint32_t lPhi,
                                          UCTCTP7RawData& ctp7Data,
                                          EcalTrigPrimDigiCollection* ecalTPGs) {
      UCTCTP7RawData::CaloType cType = UCTCTP7RawData::EBEE;
      for (uint32_t iPhi = 0; iPhi < 4; iPhi++) {  // Loop over all four phi divisions on card
        int cPhi = -1 + lPhi * 4 + iPhi;           // Calorimeter phi index
        if (cPhi == 0)
          cPhi = 72;
        else if (cPhi == -1)
          cPhi = 71;
        else if (cPhi < -1) {
          LogError("CaloLayer1Unpacker") << "Major error in makeECalTPGs" << std::endl;
          return;
        }
        for (int cEta = -28; cEta <= 28; cEta++) {  // Calorimeter Eta indices (HB/HE for now)
          if (cEta != 0) {                          // Calorimeter eta = 0 is invalid
            bool negativeEta = false;
            if (cEta < 0)
              negativeEta = true;
            uint32_t iEta = abs(cEta);
            // This code is fragile! Note that towerDatum is packed as is done in EcalTriggerPrimitiveSample
            // Bottom 8-bits are ET
            // Then finegrain feature bit
            // Then three bits have ttBits, which I have no clue about (not available on ECAL links so not set)
            // Then there is a spare FG Veto bit, which is used for L1 spike detection (not available on ECAL links so not set)
            // Top three bits seem to be unused. So, we steal those to set the tower masking, link masking and link status information
            // To decode these custom three bits use ((EcalTriggerPrimitiveSample::raw() >> 13) & 0x7)
            uint32_t towerDatum = ctp7Data.getET(cType, negativeEta, iEta, iPhi);
            if (ctp7Data.getFB(cType, negativeEta, iEta, iPhi) != 0)
              towerDatum |= 0x0100;
            if (ctp7Data.isTowerMasked(cType, negativeEta, iEta, iPhi))
              towerDatum |= 0x2000;
            if (ctp7Data.isLinkMasked(cType, negativeEta, iEta, iPhi))
              towerDatum |= 0x4000;
            if (ctp7Data.isLinkMisaligned(cType, negativeEta, iEta, iPhi) ||
                ctp7Data.isLinkInError(cType, negativeEta, iEta, iPhi) ||
                ctp7Data.isLinkDown(cType, negativeEta, iEta, iPhi))
              towerDatum |= 0x8000;
            EcalTriggerPrimitiveSample sample(towerDatum);
            int zSide = cEta / ((int)iEta);
            // As far as I can tell, the ECal unpacker only uses barrel and endcap IDs, never EcalTriggerTower
            const EcalSubdetector ecalTriggerTower =
                (iEta > 17) ? EcalSubdetector::EcalEndcap : EcalSubdetector::EcalBarrel;
            EcalTrigTowerDetId id(zSide, ecalTriggerTower, iEta, cPhi);
            EcalTriggerPrimitiveDigi tpg(id);
            tpg.setSize(1);
            tpg.setSample(0, sample);
            ecalTPGs->push_back(tpg);
          }
        }
      }
    }

    void CaloLayer1Unpacker::makeHCalTPGs(uint32_t lPhi,
                                          UCTCTP7RawData& ctp7Data,
                                          HcalTrigPrimDigiCollection* hcalTPGs) {
      UCTCTP7RawData::CaloType cType = UCTCTP7RawData::HBHE;
      for (uint32_t iPhi = 0; iPhi < 4; iPhi++) {  // Loop over all four phi divisions on card
        int cPhi = -1 + lPhi * 4 + iPhi;           // Calorimeter phi index
        if (cPhi == 0)
          cPhi = 72;
        else if (cPhi == -1)
          cPhi = 71;
        else if (cPhi < -1) {
          LogError("CaloLayer1Unpacker") << "Major error in makeHCalTPGs" << std::endl;
          return;
        }
        for (int cEta = -28; cEta <= 28; cEta++) {  // Calorimeter Eta indices (HB/HE for now)
          if (cEta != 0) {                          // Calorimeter eta = 0 is invalid
            bool negativeEta = false;
            if (cEta < 0)
              negativeEta = true;
            uint32_t iEta = abs(cEta);
            // This code is fragile! Note that towerDatum is packed as is done in HcalTriggerPrimitiveSample
            // Bottom 8-bits are ET
            // Then feature bit
            // The remaining bits are undefined presently
            // We use next three bits for link details, which we did not have room in EcalTriggerPrimitiveSample case
            // We use next three bits to set the tower masking, link masking and link status information as done for Ecal
            // To decode these custom six bits use ((EcalTriggerPrimitiveSample::raw() >> 9) & 0x77)
            uint32_t towerDatum = ctp7Data.getET(cType, negativeEta, iEta, iPhi);
            if (ctp7Data.getFB(cType, negativeEta, iEta, iPhi) != 0)
              towerDatum |= 0x0100;
            if (ctp7Data.isLinkMisaligned(cType, negativeEta, iEta, iPhi))
              towerDatum |= 0x0200;
            if (ctp7Data.isLinkInError(cType, negativeEta, iEta, iPhi))
              towerDatum |= 0x0400;
            if (ctp7Data.isLinkDown(cType, negativeEta, iEta, iPhi))
              towerDatum |= 0x0800;
            if (ctp7Data.isTowerMasked(cType, negativeEta, iEta, iPhi))
              towerDatum |= 0x2000;
            if (ctp7Data.isLinkMasked(cType, negativeEta, iEta, iPhi))
              towerDatum |= 0x4000;
            if (ctp7Data.isLinkMisaligned(cType, negativeEta, iEta, iPhi) ||
                ctp7Data.isLinkInError(cType, negativeEta, iEta, iPhi) ||
                ctp7Data.isLinkDown(cType, negativeEta, iEta, iPhi))
              towerDatum |= 0x8000;
            HcalTriggerPrimitiveSample sample(towerDatum);
            HcalTrigTowerDetId id(cEta, cPhi);
            HcalTriggerPrimitiveDigi tpg(id);
            tpg.setSize(1);
            tpg.setSample(0, sample);
            hcalTPGs->push_back(tpg);
          }
        }
      }
    }

    void CaloLayer1Unpacker::makeHFTPGs(uint32_t lPhi, UCTCTP7RawData& ctp7Data, HcalTrigPrimDigiCollection* hcalTPGs) {
      UCTCTP7RawData::CaloType cType = UCTCTP7RawData::HF;
      for (uint32_t side = 0; side <= 1; side++) {
        bool negativeEta = false;
        if (side == 0)
          negativeEta = true;
        for (uint32_t iEta = 30; iEta <= 40; iEta++) {
          for (uint32_t iPhi = 0; iPhi < 2; iPhi++) {
            if (iPhi == 1 && iEta == 40)
              iEta = 41;
            int cPhi = 1 + lPhi * 4 + iPhi * 2;  // Calorimeter phi index: 1, 3, 5, ... 71
            if (iEta == 41)
              cPhi -= 2;                  // Last two HF are 3, 7, 11, ...
            cPhi = (cPhi + 69) % 72 + 1;  // cPhi -= 2 mod 72
            int cEta = iEta;
            if (negativeEta)
              cEta = -iEta;
            // This code is fragile! Note that towerDatum is packed as is done in HcalTriggerPrimitiveSample
            // Bottom 8-bits are ET
            // Then feature bit
            // Then minBias ADC count bit
            // The remaining bits are undefined presently
            // We use next three bits for link details, which we did not have room in EcalTriggerPrimitiveSample case
            // We use next three bits to set the tower masking, link masking and link status information as done for Ecal
            // To decode these custom six bits use ((EcalTriggerPrimitiveSample::raw() >> 9) & 0x77)
            uint32_t towerDatum = ctp7Data.getET(cType, negativeEta, iEta, iPhi);
            towerDatum |= ctp7Data.getFB(cType, negativeEta, iEta, iPhi) << 8;
            if (ctp7Data.isLinkMisaligned(cType, negativeEta, iEta, iPhi))
              towerDatum |= 0x0400;
            if (ctp7Data.isLinkInError(cType, negativeEta, iEta, iPhi))
              towerDatum |= 0x0800;
            if (ctp7Data.isLinkDown(cType, negativeEta, iEta, iPhi))
              towerDatum |= 0x1000;
            if (ctp7Data.isTowerMasked(cType, negativeEta, iEta, iPhi))
              towerDatum |= 0x2000;
            if (ctp7Data.isLinkMasked(cType, negativeEta, iEta, iPhi))
              towerDatum |= 0x4000;
            if (ctp7Data.isLinkMisaligned(cType, negativeEta, iEta, iPhi) ||
                ctp7Data.isLinkInError(cType, negativeEta, iEta, iPhi) ||
                ctp7Data.isLinkDown(cType, negativeEta, iEta, iPhi))
              towerDatum |= 0x8000;
            HcalTriggerPrimitiveSample sample(towerDatum);
            HcalTrigTowerDetId id(cEta, cPhi);
            id.setVersion(1);  // To not process these 1x1 HF TPGs with RCT
            HcalTriggerPrimitiveDigi tpg(id);
            tpg.setSize(1);
            tpg.setSample(0, sample);
            hcalTPGs->push_back(tpg);
          }
        }
      }
    }

    void CaloLayer1Unpacker::makeRegions(uint32_t lPhi, UCTCTP7RawData& ctp7Data, L1CaloRegionCollection* regions) {
      for (uint32_t side = 0; side <= 1; side++) {
        bool negativeEta = false;
        if (side == 0)
          negativeEta = true;
        for (uint32_t region = 0; region <= 6; region++) {
          uint32_t regionData = ctp7Data.getRegionSummary(negativeEta, region);
          uint32_t lEta = 10 - region;  // GCT eta goes 0-21, 0-3 -HF, 4-10 -B/E, 11-17 +B/E, 18-21 +HF
          if (!negativeEta)
            lEta = region + 11;
          regions->push_back(L1CaloRegion((uint16_t)regionData, (unsigned)lEta, (unsigned)lPhi, (int16_t)0));
        }
      }
    }

  }  // namespace stage2
}  // namespace l1t

DEFINE_L1T_UNPACKER(l1t::stage2::CaloLayer1Unpacker);

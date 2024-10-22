#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "L1TObjectCollections.h"

#include "L1TStage2Layer2Constants.h"
#include "EtSumUnpacker_0x10010057.h"

namespace l1t {
  namespace stage2 {
    EtSumUnpacker_0x10010057::EtSumUnpacker_0x10010057() : EtSumCopy_(0) {}

    bool EtSumUnpacker_0x10010057::unpack(const Block& block, UnpackerCollections* coll) {
      using namespace l1t::stage2::layer2;

      LogDebug("L1T") << "Block ID  = " << block.header().getID() << " size = " << block.header().getSize();

      int nBX = int(
          ceil(block.header().getSize() / demux::nOutputFramePerBX));  // Since there 6 frames per demux output event
      // expect the first four frames to be the first 4 EtSum objects reported per event (see CMS IN-2013/005)

      // Find the central, first and last BXs
      int firstBX = -(ceil((double)nBX / 2.) - 1);
      int lastBX;
      if (nBX % 2 == 0) {
        lastBX = ceil((double)nBX / 2.);
      } else {
        lastBX = ceil((double)nBX / 2.) - 1;
      }

      auto res_ = static_cast<L1TObjectCollections*>(coll)->getEtSums(EtSumCopy_);
      res_->setBXRange(firstBX, lastBX);

      LogDebug("L1T") << "nBX = " << nBX << " first BX = " << firstBX << " lastBX = " << lastBX;

      // Loop over multiple BX and fill EtSums collection
      for (int bx = firstBX; bx <= lastBX; bx++) {
        // ET
        int iFrame = (bx - firstBX) * demux::nOutputFramePerBX;

        uint32_t raw_data = block.payload().at(iFrame);

        l1t::EtSum et{l1t::EtSum::kTotalEt};

        et.setHwPt(raw_data & 0xFFF);
        et.setP4(l1t::CaloTools::p4Demux(&et));

        LogDebug("L1T") << "ET: pT " << et.hwPt() << " bx " << bx;

        res_->push_back(bx, et);

        // ET EM

        l1t::EtSum etem{l1t::EtSum::kTotalEtEm};

        etem.setHwPt((raw_data >> 12) & 0xFFF);
        etem.setP4(l1t::CaloTools::p4Demux(&etem));

        LogDebug("L1T") << "ETEM: pT " << etem.hwPt() << " bx " << bx;

        res_->push_back(bx, etem);

        // MBHFPT0

        l1t::EtSum mbp0{l1t::EtSum::kMinBiasHFP0};
        mbp0.setHwPt((raw_data >> 28) & 0xf);

        res_->push_back(bx, mbp0);

        // HT

        raw_data = block.payload()[iFrame + 1];

        l1t::EtSum ht{l1t::EtSum::kTotalHt};

        ht.setHwPt(raw_data & 0xFFF);
        ht.setP4(l1t::CaloTools::p4Demux(&ht));

        LogDebug("L1T") << "HT: pT " << ht.hwPt();

        res_->push_back(bx, ht);

        //MBHFMT0

        l1t::EtSum mbm0{l1t::EtSum::kMinBiasHFM0};
        mbm0.setHwPt((raw_data >> 28) & 0xf);

        res_->push_back(bx, mbm0);

        //  MET (no HF)

        raw_data = block.payload()[iFrame + 2];

        l1t::EtSum met{l1t::EtSum::kMissingEt};

        met.setHwPt(raw_data & 0xFFF);
        met.setHwPhi((raw_data >> 12) & 0xFF);
        met.setP4(l1t::CaloTools::p4Demux(&met));

        LogDebug("L1T") << "MET: phi " << met.hwPhi() << " pT " << met.hwPt() << " bx " << bx;

        res_->push_back(bx, met);

        // MBHFPT1

        l1t::EtSum mbp1{l1t::EtSum::kMinBiasHFP1};
        mbp1.setHwPt((raw_data >> 28) & 0xf);

        res_->push_back(bx, mbp1);

        // ET ASYMM

        l1t::EtSum etAsym{l1t::EtSum::kAsymEt};
        etAsym.setHwPt((raw_data >> 20) & 0xFF);

        res_->push_back(bx, etAsym);

        // MHT

        raw_data = block.payload()[iFrame + 3];

        l1t::EtSum mht{l1t::EtSum::kMissingHt};

        mht.setHwPt(raw_data & 0xFFF);
        mht.setHwPhi((raw_data >> 12) & 0xFF);
        mht.setP4(l1t::CaloTools::p4Demux(&mht));

        LogDebug("L1T") << "MHT: phi " << mht.hwPhi() << " pT " << mht.hwPt() << " bx " << bx;

        res_->push_back(bx, mht);

        // MBHFMT1

        l1t::EtSum mbm1{l1t::EtSum::kMinBiasHFM1};
        mbm1.setHwPt((raw_data >> 28) & 0xf);

        res_->push_back(bx, mbm1);

        // HT ASYMM

        l1t::EtSum htAsym{l1t::EtSum::kAsymHt};
        htAsym.setHwPt((raw_data >> 20) & 0xFF);

        res_->push_back(bx, htAsym);

        //  MET (with HF)

        raw_data = block.payload()[iFrame + 4];

        l1t::EtSum methf{l1t::EtSum::kMissingEtHF};

        methf.setHwPt(raw_data & 0xFFF);
        methf.setHwPhi((raw_data >> 12) & 0xFF);
        methf.setP4(l1t::CaloTools::p4Demux(&methf));

        LogDebug("L1T") << "METHF: phi " << methf.hwPhi() << " pT " << methf.hwPt() << " bx " << bx;

        res_->push_back(bx, methf);

        // ETHF ASYMM

        l1t::EtSum etHFAsym{l1t::EtSum::kAsymEtHF};
        etHFAsym.setHwPt((raw_data >> 20) & 0xFF);

        res_->push_back(bx, etHFAsym);

        // CENTRALITY LOWER NIB

        uint32_t centLN = ((raw_data >> 28) & 0xF);

        // MHT with HF

        raw_data = block.payload()[iFrame + 5];

        l1t::EtSum mhthf{l1t::EtSum::kMissingHtHF};

        mhthf.setHwPt(raw_data & 0xFFF);
        mhthf.setHwPhi((raw_data >> 12) & 0xFF);
        mhthf.setP4(l1t::CaloTools::p4Demux(&mhthf));

        LogDebug("L1T") << "MHThf: phi " << mhthf.hwPhi() << " pT " << mhthf.hwPt() << " bx " << bx;

        res_->push_back(bx, mhthf);

        // HTHF ASYMM

        l1t::EtSum htHFAsym{l1t::EtSum::kAsymHtHF};
        htHFAsym.setHwPt((raw_data >> 20) & 0xFF);

        res_->push_back(bx, htHFAsym);

        // CENTRALITY UPPER NIB

        uint32_t centUN = ((raw_data >> 28) & 0xF);

        l1t::EtSum cent{l1t::EtSum::kCentrality};
        cent.setHwPt((centUN << 4) | centLN);

        res_->push_back(bx, cent);

        //HI-SUM

        raw_data = block.payload()[iFrame + 1];

        l1t::EtSum towCount{l1t::EtSum::kTowerCount};
        towCount.setHwPt((raw_data >> 12) & 0x1FFF);
        towCount.setP4(l1t::CaloTools::p4Demux(&towCount));

        res_->push_back(bx, towCount);
      }

      return true;
    }
  }  // namespace stage2
}  // namespace l1t

DEFINE_L1T_UNPACKER(l1t::stage2::EtSumUnpacker_0x10010057);

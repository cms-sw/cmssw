#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "L1Trigger/L1TMuon/interface/MuonRawDigiTranslator.h"

#include "GMTCollections.h"
#include "IntermediateMuonUnpacker.h"

namespace l1t {
  namespace stage2 {
    IntermediateMuonUnpacker::IntermediateMuonUnpacker() : res1_(nullptr), res2_(nullptr), coll1Cnt_(0) {}

    bool IntermediateMuonUnpacker::unpack(const Block& block, UnpackerCollections* coll) {
      LogDebug("L1T") << "Block ID  = " << block.header().getID() << " size = " << block.header().getSize();
      // process only if there is a payload
      // If all BX block were zero suppressed the block header size is 0.
      if (block.header().getSize() < 1) {
        return true;
      }

      auto payload = block.payload();

      int nBX, firstBX, lastBX;
      // Check if per BX zero suppression was enabled
      bool bxZsEnabled = ((block.header().getFlags() >> bxzs_enable_shift_) & 0x1) == 1;
      // Calculate the total number of BXs
      if (bxZsEnabled) {
        BxBlockHeader bxHeader(payload.at(0));
        nBX = bxHeader.getTotalBx();
      } else {
        nBX = int(ceil(block.header().getSize() / nWords_));
      }
      getBXRange(nBX, firstBX, lastBX);

      // decide which collections to use according to the link ID
      unsigned int linkId = (block.header().getID() - 1) / 2;
      // Intermediate muons come on uGMT output links 24-31.
      // Each link can transmit 3 muons and we receive 4 intermediate muons from
      // EMTF/OMTF on each detector side and 8 intermediate muons from BMTF.
      // Therefore, the muon at a certain position on a link has to be filled
      // in a specific collection. The order is from links 24-31:
      // 4 muons from EMTF pos, 4 from OMTF pos, 8 from BMTF, 4 from OMTF neg,
      // and 4 from EMTF neg.
      switch (linkId) {
        case 24:
          res1_ = static_cast<GMTCollections*>(coll)->getImdMuonsEMTFPos();
          res2_ = static_cast<GMTCollections*>(coll)->getImdMuonsEMTFPos();
          coll1Cnt_ = 3;
          break;
        case 25:
          res1_ = static_cast<GMTCollections*>(coll)->getImdMuonsEMTFPos();
          res2_ = static_cast<GMTCollections*>(coll)->getImdMuonsOMTFPos();
          coll1Cnt_ = 1;
          break;
        case 26:
          res1_ = static_cast<GMTCollections*>(coll)->getImdMuonsOMTFPos();
          res2_ = static_cast<GMTCollections*>(coll)->getImdMuonsBMTF();
          coll1Cnt_ = 2;
          break;
        case 27:
          res1_ = static_cast<GMTCollections*>(coll)->getImdMuonsBMTF();
          res2_ = static_cast<GMTCollections*>(coll)->getImdMuonsBMTF();
          coll1Cnt_ = 3;
          break;
        case 28:
          res1_ = static_cast<GMTCollections*>(coll)->getImdMuonsBMTF();
          res2_ = static_cast<GMTCollections*>(coll)->getImdMuonsBMTF();
          coll1Cnt_ = 3;
          break;
        case 29:
          res1_ = static_cast<GMTCollections*>(coll)->getImdMuonsBMTF();
          res2_ = static_cast<GMTCollections*>(coll)->getImdMuonsOMTFNeg();
          coll1Cnt_ = 1;
          break;
        case 30:
          res1_ = static_cast<GMTCollections*>(coll)->getImdMuonsOMTFNeg();
          res2_ = static_cast<GMTCollections*>(coll)->getImdMuonsEMTFNeg();
          coll1Cnt_ = 2;
          break;
        case 31:
          res1_ = static_cast<GMTCollections*>(coll)->getImdMuonsEMTFNeg();
          res2_ = static_cast<GMTCollections*>(coll)->getImdMuonsEMTFNeg();
          coll1Cnt_ = 3;
          break;
        default:
          edm::LogWarning("L1T") << "Block ID " << block.header().getID()
                                 << " not associated with intermediate muons. Skip.";
          return false;
      }
      res1_->setBXRange(firstBX, lastBX);
      res2_->setBXRange(firstBX, lastBX);
      LogDebug("L1T") << "nBX = " << nBX << " first BX = " << firstBX << " lastBX = " << lastBX;

      // Get the BX blocks and unpack them
      auto bxBlocks = block.getBxBlocks(nWords_, bxZsEnabled);
      for (const auto& bxBlock : bxBlocks) {
        unpackBx(bxBlock.header().getBx(), bxBlock.payload());
      }
      return true;
    }

    void IntermediateMuonUnpacker::unpackBx(int bx, const std::vector<uint32_t>& payload, unsigned int startIdx) {
      unsigned int i = startIdx;
      // Check if there are enough words left in the payload
      if (i + nWords_ <= payload.size()) {
        unsigned int muonCnt = 0;
        for (unsigned nWord = 0; nWord < nWords_; nWord += 2, ++muonCnt) {
          uint32_t raw_data_00_31 = payload[i++];
          uint32_t raw_data_32_63 = payload[i++];
          LogDebug("L1T") << "raw_data_00_31 = 0x" << hex << raw_data_00_31 << " raw_data_32_63 = 0x" << raw_data_32_63;
          // skip empty muons (hwPt == 0)
          if (((raw_data_00_31 >> l1t::MuonRawDigiTranslator::ptShift_) & l1t::MuonRawDigiTranslator::ptMask_) == 0) {
            LogDebug("L1T") << "Muon hwPt zero. Skip.";
            continue;
          }

          Muon mu;

          // The intermediate muons of the uGMT (FED number 1402) do not
          // have coordinates estimated at the vertex in the RAW data.
          // The corresponding bits are set to zero.
          MuonRawDigiTranslator::fillIntermediateMuon(mu, raw_data_00_31, raw_data_32_63, getAlgoVersion());

          LogDebug("L1T") << "Mu" << nWord / 2 << ": eta " << mu.hwEta() << " phi " << mu.hwPhi() << " pT " << mu.hwPt()
                          << " iso " << mu.hwIso() << " qual " << mu.hwQual() << " charge " << mu.hwCharge()
                          << " charge valid " << mu.hwChargeValid();

          if (muonCnt < coll1Cnt_) {
            res1_->push_back(bx, mu);
          } else {
            res2_->push_back(bx, mu);
          }
        }
      } else {
        edm::LogWarning("L1T") << "Only " << payload.size() - i << " 32 bit words in this BX but " << nWords_
                               << " are required. Not unpacking the data for BX " << bx << ".";
      }
    }

  }  // namespace stage2
}  // namespace l1t

DEFINE_L1T_UNPACKER(l1t::stage2::IntermediateMuonUnpacker);

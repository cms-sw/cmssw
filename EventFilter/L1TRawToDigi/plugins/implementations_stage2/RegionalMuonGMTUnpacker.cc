#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "L1Trigger/L1TMuon/interface/RegionalMuonRawDigiTranslator.h"
#include "RegionalMuonGMTUnpacker.h"

namespace l1t {
  namespace stage2 {
    bool RegionalMuonGMTUnpacker::unpack(const Block& block, UnpackerCollections* coll) {
      unsigned int blockId = block.header().getID();
      LogDebug("L1T") << "Block ID  = " << blockId << " size = " << block.header().getSize();
      // Process only if there is a payload.
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

      // decide which collection to use according to the link ID
      unsigned int linkId = blockId / 2;
      int processor;
      RegionalMuonCandBxCollection* regionalMuonCollection;
      RegionalMuonShowerBxCollection* regionalMuonShowerCollection;
      tftype trackFinder;
      if (linkId > 47 && linkId < 60) {
        regionalMuonCollection = static_cast<GMTCollections*>(coll)->getRegionalMuonCandsBMTF();
        regionalMuonShowerCollection =
            new RegionalMuonShowerBxCollection();  // To avoid warning re uninitialised collection
        trackFinder = tftype::bmtf;
        processor = linkId - 48;
      } else if (linkId > 41 && linkId < 66) {
        regionalMuonCollection = static_cast<GMTCollections*>(coll)->getRegionalMuonCandsOMTF();
        regionalMuonShowerCollection =
            new RegionalMuonShowerBxCollection();  // To avoid warning re uninitialised collection
        if (linkId < 48) {
          trackFinder = tftype::omtf_pos;
          processor = linkId - 42;
        } else {
          trackFinder = tftype::omtf_neg;
          processor = linkId - 60;
        }
      } else if (linkId > 35 && linkId < 72) {
        regionalMuonCollection = static_cast<GMTCollections*>(coll)->getRegionalMuonCandsEMTF();
        regionalMuonShowerCollection = static_cast<GMTCollections*>(coll)->getRegionalMuonShowersEMTF();
        if (linkId < 42) {
          trackFinder = tftype::emtf_pos;
          processor = linkId - 36;
        } else {
          trackFinder = tftype::emtf_neg;
          processor = linkId - 66;
        }
      } else {
        edm::LogError("L1T") << "No TF muon expected for link " << linkId;
        return false;
      }
      regionalMuonCollection->setBXRange(firstBX, lastBX);
      regionalMuonShowerCollection->setBXRange(firstBX, lastBX);

      LogDebug("L1T") << "nBX = " << nBX << " first BX = " << firstBX << " lastBX = " << lastBX;

      // Get the BX blocks and unpack them
      auto bxBlocks = block.getBxBlocks(nWords_, bxZsEnabled);
      for (const auto& bxBlock : bxBlocks) {
        // Throw an exception if finding a corrupt BX header with out of range BX numbers
        const auto bx = bxBlock.header().getBx();
        if (bx < firstBX || bx > lastBX) {
          throw cms::Exception("CorruptData") << "Corrupt RAW data from AMC " << block.amc().getAMCNumber()
                                              << ". BX number " << bx << " in BX header is outside of the BX range ["
                                              << firstBX << "," << lastBX << "] defined in the block header.";
        }
        // Check if there are enough words left in the BX block payload
        auto bxPayload = bxBlock.payload();
        if (nWords_ <= bxPayload.size()) {
          for (unsigned nWord = 0; nWord < nWords_; nWord += 2) {
            uint32_t raw_data_00_31 = bxPayload[nWord];
            uint32_t raw_data_32_63 = bxPayload[nWord + 1];
            LogDebug("L1T") << "raw_data_00_31 = 0x" << hex << setw(8) << setfill('0') << raw_data_00_31
                            << " raw_data_32_63 = 0x" << setw(8) << setfill('0') << raw_data_32_63;
            // skip empty muons (hwPt == 0)
            if (((raw_data_00_31 >> l1t::RegionalMuonRawDigiTranslator::ptShift_) &
                 l1t::RegionalMuonRawDigiTranslator::ptMask_) == 0) {
              LogDebug("L1T") << "Muon hwPt zero. Skip.";
              continue;
            }
            // Detect and ignore comma events
            if (raw_data_00_31 == 0x505050bc || raw_data_32_63 == 0x505050bc) {
              edm::LogWarning("L1T") << "Comma detected in raw data stream. Orbit number: "
                                     << block.amc().getOrbitNumber() << ", BX ID: " << block.amc().getBX()
                                     << ", BX: " << bx << ", linkId: " << linkId << ", Raw data: 0x" << hex << setw(8)
                                     << setfill('0') << raw_data_32_63 << setw(8) << setfill('0') << raw_data_00_31
                                     << dec << ". Skip.";
              continue;
            }

            RegionalMuonCand mu;
            mu.setMuIdx(nWord / 2);

            RegionalMuonRawDigiTranslator::fillRegionalMuonCand(
                mu, raw_data_00_31, raw_data_32_63, processor, trackFinder, isKbmtf_, useEmtfDisplacementInfo_);

            LogDebug("L1T") << "Mu" << nWord / 2 << ": eta " << mu.hwEta() << " phi " << mu.hwPhi() << " pT "
                            << mu.hwPt() << " qual " << mu.hwQual() << " sign " << mu.hwSign() << " sign valid "
                            << mu.hwSignValid() << " unconstrained pT " << mu.hwPtUnconstrained();

            regionalMuonCollection->push_back(bx, mu);
          }
          // Fill RegionalMuonShower objects. For this we need to look at all six words together.
          RegionalMuonShower muShower;
          if (RegionalMuonRawDigiTranslator::fillRegionalMuonShower(
                  muShower, bxPayload, processor, trackFinder, useEmtfShowers_)) {
            regionalMuonShowerCollection->push_back(bx, muShower);
          }
        } else {
          unsigned int nWords =
              nWords_;  // This seems unnecessary but it prevents an 'undefined reference' linker error that occurs when using nWords_ directly with LogWarning.
          edm::LogWarning("L1T") << "Only " << bxPayload.size() << " 32 bit words in this BX but " << nWords
                                 << " are required. Not unpacking the data for BX " << bx << ".";
        }
      }
      return true;
    }
  }  // namespace stage2
}  // namespace l1t

DEFINE_L1T_UNPACKER(l1t::stage2::RegionalMuonGMTUnpacker);

#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "L1Trigger/L1TMuon/interface/RegionalMuonRawDigiTranslator.h"
#include "GMTCollections.h"

namespace l1t {
   namespace stage2 {
      class RegionalMuonGMTUnpacker : public Unpacker {
         public:
            virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
      };
   }
}

// Implementation
namespace l1t {
   namespace stage2 {
      bool
      RegionalMuonGMTUnpacker::unpack(const Block& block, UnpackerCollections *coll)
      {
         unsigned int blockId = block.header().getID();
         LogDebug("L1T") << "Block ID  = " << blockId << " size = " << block.header().getSize();

         auto payload = block.payload();

         unsigned int nWords = 6; // every link transmits 6 words (3 muons) per bx
         int nBX, firstBX, lastBX;
         nBX = int(ceil(block.header().getSize() / nWords));
         getBXRange(nBX, firstBX, lastBX);
         // only use central BX for now
         //firstBX = 0;
         //lastBX = 0;
         //LogDebug("L1T") << "BX override. Set first BX = lastBX = 0.";

         // decide which collection to use according to the link ID
         unsigned int linkId = blockId / 2;
         int processor;
         RegionalMuonCandBxCollection* res;
         tftype trackFinder;
         if (linkId > 47 && linkId < 60) {
            res = static_cast<GMTCollections*>(coll)->getRegionalMuonCandsBMTF();
            trackFinder = tftype::bmtf;
            processor = linkId - 48;
         } else if (linkId > 41 && linkId < 66) {
            res = static_cast<GMTCollections*>(coll)->getRegionalMuonCandsOMTF();
            if (linkId < 48) {
               trackFinder = tftype::omtf_pos;
               processor = linkId - 42;
            } else {
               trackFinder = tftype::omtf_neg;
               processor = linkId - 60;
            }
         } else if (linkId > 35 && linkId < 72) {
            res = static_cast<GMTCollections*>(coll)->getRegionalMuonCandsEMTF();
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
         res->setBXRange(firstBX, lastBX);

         LogDebug("L1T") << "nBX = " << nBX << " first BX = " << firstBX << " lastBX = " << lastBX;

         // Initialise index
         int unsigned i = 0;

         // Loop over multiple BX and then number of muons filling muon collection
         for (int bx = firstBX; bx <= lastBX; ++bx) {
            for (unsigned nWord = 0; nWord < nWords && i < block.header().getSize(); nWord += 2) {
               uint32_t raw_data_00_31 = payload[i++];
               uint32_t raw_data_32_63 = payload[i++];        
               LogDebug("L1T") << "raw_data_00_31 = 0x" << hex << setw(8) << setfill('0') << raw_data_00_31 << " raw_data_32_63 = 0x" << setw(8) << setfill('0') << raw_data_32_63;
               // skip empty muons (hwPt == 0)
               //// the msb are reserved for global information
               //if ((raw_data_00_31 & 0x7FFFFFFF) == 0 && (raw_data_32_63 & 0x7FFFFFFF) == 0) {
               if (((raw_data_00_31 >> l1t::RegionalMuonRawDigiTranslator::ptShift_) & l1t::RegionalMuonRawDigiTranslator::ptMask_) == 0) {
                  LogDebug("L1T") << "Muon hwPt zero. Skip.";
                  continue;
               }
               // Detect and ignore comma events
               if (raw_data_00_31 == 0x505050bc || raw_data_32_63 == 0x505050bc) {
                  edm::LogWarning("L1T") << "Comma detected in raw data stream. Orbit number: " << block.amc().getOrbitNumber() << ", BX ID: " << block.amc().getBX() << ", BX: " << bx << ", linkId: " << linkId << ", Raw data: 0x" << hex << setw(8) << setfill('0') << raw_data_32_63 << setw(8) << setfill('0') << raw_data_00_31 << dec << ". Skip.";
                  continue;
               }
 
               RegionalMuonCand mu;
 
               RegionalMuonRawDigiTranslator::fillRegionalMuonCand(mu, raw_data_00_31, raw_data_32_63, processor, trackFinder);

               LogDebug("L1T") << "Mu" << nWord/2 << ": eta " << mu.hwEta() << " phi " << mu.hwPhi() << " pT " << mu.hwPt() << " qual " << mu.hwQual() << " sign " << mu.hwSign() << " sign valid " << mu.hwSignValid();

               res->push_back(bx, mu);
            }
         }
         return true;
      }
   }
}

DEFINE_L1T_UNPACKER(l1t::stage2::RegionalMuonGMTUnpacker);

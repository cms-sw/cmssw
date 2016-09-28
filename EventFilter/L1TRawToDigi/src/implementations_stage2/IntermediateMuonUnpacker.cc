#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "L1Trigger/L1TMuon/interface/MuonRawDigiTranslator.h"

#include "GMTCollections.h"

namespace l1t {
   namespace stage2 {
      class IntermediateMuonUnpacker : public Unpacker {
         public:
            virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
      };
   }
}

// Implementation
namespace l1t {
   namespace stage2 {
      bool
      IntermediateMuonUnpacker::unpack(const Block& block, UnpackerCollections *coll)
      {
         LogDebug("L1T") << "Block ID  = " << block.header().getID() << " size = " << block.header().getSize();

         auto payload = block.payload();

         unsigned int nWords = 6; // every link transmits 6 words (3 muons) per bx
         int nBX, firstBX, lastBX;
         nBX = int(ceil(block.header().getSize() / nWords));
         getBXRange(nBX, firstBX, lastBX);

         // decide which collections to use according to the link ID
         unsigned int linkId = (block.header().getID() - 1) / 2;
         unsigned int coll1Cnt = 0;
         MuonBxCollection* res1;
         MuonBxCollection* res2;
         // Intermediate muons come on uGMT output links 24-31.
         // Each link can transmit 3 muons and we receive 4 intermediate muons from
         // EMTF/OMTF on each detector side and 8 intermediate muons from BMTF.
         // Therefore, the muon at a certain position on a link has to be filled
         // in a specific collection. The order is from links 24-31:
         // 4 muons from EMTF pos, 4 from OMTF pos, 8 from BMTF, 4 from OMTF neg,
         // and 4 from EMTF neg.
         switch (linkId) {
           case 24:
             res1 = static_cast<GMTCollections*>(coll)->getImdMuonsEMTFPos();
             res2 = static_cast<GMTCollections*>(coll)->getImdMuonsEMTFPos();
             coll1Cnt = 3;
             break;
           case 25:
             res1 = static_cast<GMTCollections*>(coll)->getImdMuonsEMTFPos();
             res2 = static_cast<GMTCollections*>(coll)->getImdMuonsOMTFPos();
             coll1Cnt = 1;
             break;
           case 26:
             res1 = static_cast<GMTCollections*>(coll)->getImdMuonsOMTFPos();
             res2 = static_cast<GMTCollections*>(coll)->getImdMuonsBMTF();
             coll1Cnt = 2;
             break;
           case 27:
             res1 = static_cast<GMTCollections*>(coll)->getImdMuonsBMTF();
             res2 = static_cast<GMTCollections*>(coll)->getImdMuonsBMTF();
             coll1Cnt = 3;
             break;
           case 28:
             res1 = static_cast<GMTCollections*>(coll)->getImdMuonsBMTF();
             res2 = static_cast<GMTCollections*>(coll)->getImdMuonsBMTF();
             coll1Cnt = 3;
             break;
           case 29:
             res1 = static_cast<GMTCollections*>(coll)->getImdMuonsBMTF();
             res2 = static_cast<GMTCollections*>(coll)->getImdMuonsOMTFNeg();
             coll1Cnt = 1;
             break;
           case 30:
             res1 = static_cast<GMTCollections*>(coll)->getImdMuonsOMTFNeg();
             res2 = static_cast<GMTCollections*>(coll)->getImdMuonsEMTFNeg();
             coll1Cnt = 2;
             break;
           case 31:
             res1 = static_cast<GMTCollections*>(coll)->getImdMuonsEMTFNeg();
             res2 = static_cast<GMTCollections*>(coll)->getImdMuonsEMTFNeg();
             coll1Cnt = 3;
             break;
           default:
             edm::LogWarning("L1T") << "Block ID " << block.header().getID() << " not associated with intermediate muons. Skip.";
             return false;
         }
         res1->setBXRange(firstBX, lastBX);
         res2->setBXRange(firstBX, lastBX);

         LogDebug("L1T") << "nBX = " << nBX << " first BX = " << firstBX << " lastBX = " << lastBX;

         // Initialise indices
         unsigned int i = 0;
         unsigned int muonCnt = 0;

         // Loop over multiple BX and then number of muons filling muon collection
         for (int bx = firstBX; bx <= lastBX; ++bx) {
            for (unsigned nWord = 0; nWord < nWords && i < block.header().getSize(); nWord += 2, ++muonCnt) {
               uint32_t raw_data_00_31 = payload[i++];
               uint32_t raw_data_32_63 = payload[i++];        
               LogDebug("L1T") << "raw_data_00_31 = 0x" << hex << raw_data_00_31 << " raw_data_32_63 = 0x" << raw_data_32_63;
               // skip empty muons (hwPt == 0)
               if (((raw_data_00_31 >> l1t::MuonRawDigiTranslator::ptShift_) & l1t::MuonRawDigiTranslator::ptMask_) == 0) {
                  LogDebug("L1T") << "Muon hwPt zero. Skip.";
                  continue;
               }

               Muon mu;
                   
               MuonRawDigiTranslator::fillMuon(mu, raw_data_00_31, raw_data_32_63);

               LogDebug("L1T") << "Mu" << nWord/2 << ": eta " << mu.hwEta() << " phi " << mu.hwPhi() << " pT " << mu.hwPt() << " iso " << mu.hwIso() << " qual " << mu.hwQual() << " charge " << mu.hwCharge() << " charge valid " << mu.hwChargeValid();

               if (muonCnt < coll1Cnt) { 
                 res1->push_back(bx, mu);
               } else {
                 res2->push_back(bx, mu);
               }
            }
         }
         return true;
      }
   }
}

DEFINE_L1T_UNPACKER(l1t::stage2::IntermediateMuonUnpacker);

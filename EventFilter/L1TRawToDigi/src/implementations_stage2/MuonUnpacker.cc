#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "L1Trigger/L1TMuon/interface/MuonRawDigiTranslator.h"

#include "L1TObjectCollections.h"
//#include "GMTCollections.h"

namespace l1t {
   namespace stage2 {
      class MuonUnpacker : public Unpacker {
         public:
            virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
      };
   }
}

// Implementation
namespace l1t {
   namespace stage2 {
      bool
      MuonUnpacker::unpack(const Block& block, UnpackerCollections *coll)
      {
         LogDebug("L1T") << "Block ID  = " << block.header().getID() << " size = " << block.header().getSize();

         auto payload = block.payload();

         unsigned int nWords = 6; // every link transmits 6 words (3 muons) per bx
         int nBX, firstBX, lastBX;
         nBX = int(ceil(block.header().getSize() / nWords));
         getBXRange(nBX, firstBX, lastBX);
         // only use central BX for now
         //firstBX = 0;
         //lastBX = 0;
         //LogDebug("L1T") << "BX override. Set first BX = lastBX = 0.";

         auto res = static_cast<L1TObjectCollections*>(coll)->getMuons();
         res->setBXRange(firstBX, lastBX);

         LogDebug("L1T") << "nBX = " << nBX << " first BX = " << firstBX << " lastBX = " << lastBX;

         // Initialise index
         int unsigned i = 0;

         // Loop over multiple BX and then number of muons filling muon collection
         for (int bx = firstBX; bx <= lastBX; ++bx) {
            for (unsigned nWord = 0; nWord < nWords && i < block.header().getSize(); nWord += 2) {
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

               res->push_back(bx, mu);
            }
         }
         return true;
      }
   }
}

DEFINE_L1T_UNPACKER(l1t::stage2::MuonUnpacker);

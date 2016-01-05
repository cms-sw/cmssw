#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

#include "L1Trigger/L1TMuon/interface/MuonRawDigiTranslator.h"
#include "GMTTokens.h"

namespace l1t {
   namespace stage2 {
      class MuonPacker : public Packer {
         public:
            virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
         private:
            typedef std::map<unsigned int, std::vector<uint32_t>> PayloadMap;
      };
   }
}

// Implementation
namespace l1t {
   namespace stage2 {
      Blocks
      MuonPacker::pack(const edm::Event& event, const PackerTokens* toks)
      {
         edm::Handle<MuonBxCollection> muons;
         event.getByToken(static_cast<const GMTTokens*>(toks)->getMuonToken(), muons);

         PayloadMap payloadMap;

         for (int i = muons->getFirstBX(); i <= muons->getLastBX(); ++i) {
            // the first muon in every BX and every block id is 0
            for (unsigned int blkId = 1; blkId < 8; blkId += 2) {
               payloadMap[blkId].push_back(0);
               payloadMap[blkId].push_back(0);
            }

            unsigned int blkId = 1;
            int muCtr = 1;
            for (auto mu = muons->begin(i); mu != muons->end(i) && muCtr <= 8; ++mu, ++muCtr) {
               uint32_t msw = 0;
               uint32_t lsw = 0;

               MuonRawDigiTranslator::generatePackedDataWords(*mu, lsw, msw);

               payloadMap[blkId].push_back(lsw);
               payloadMap[blkId].push_back(msw);

               // go to next block id after two muons
               if (muCtr%2 == 0) {
                  blkId += 2;
               }
            }

            // padding empty muons to reach 3 muons per block id per BX
            for (auto &kv : payloadMap) {
               while (kv.second.size()%6 != 0) {
                  kv.second.push_back(0);
               }
            }
         }

         Blocks blocks;
         // push everything in the blocks vector
         for (auto &kv : payloadMap) {
            blocks.push_back(Block(kv.first, kv.second));
         }
         return blocks;
      }
   }
}

DEFINE_L1T_PACKER(l1t::stage2::MuonPacker);

#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

#include "L1Trigger/L1TMuon/interface/MuonRawDigiTranslator.h"
#include "GMTTokens.h"

namespace l1t {
   namespace stage2 {
      class MuonPacker : public Packer {
         public:
	    MuonPacker(unsigned b1) : b1_(b1) {}
            virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
            unsigned b1_;
         private:
            typedef std::map<unsigned int, std::vector<uint32_t>> PayloadMap;
      };

      class GTMuonPacker : public MuonPacker {
         public:
             GTMuonPacker() : MuonPacker(0) {}
      };
      class GMTMuonPacker : public MuonPacker {
         public:
	     GMTMuonPacker() : MuonPacker(1) {}
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
         event.getByToken(static_cast<const CommonTokens*>(toks)->getMuonToken(), muons);

         PayloadMap payloadMap;

         for (int i = muons->getFirstBX(); i <= muons->getLastBX(); ++i) {
            // the first muon in every BX and every block id is 0
            for (unsigned int blkId = b1_; blkId < b1_+7; blkId += 2) {
               payloadMap[blkId].push_back(0);
               payloadMap[blkId].push_back(0);
            }

            unsigned int blkId = b1_;
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
	    //cout << kv.first << ":  " << kv.second.size() << kv.second[0] << "\n";
            blocks.push_back(Block(kv.first, kv.second));
         }
         return blocks;
      }
   }
}

DEFINE_L1T_PACKER(l1t::stage2::GTMuonPacker);
DEFINE_L1T_PACKER(l1t::stage2::GMTMuonPacker);

#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

#include "L1Trigger/L1TMuon/interface/RegionalMuonRawDigiTranslator.h"
#include "GMTTokens.h"

namespace l1t {
   namespace stage2 {
      class RegionalMuonGMTPacker : public Packer {
         public:
            virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
         private:
            typedef std::map<unsigned int, std::vector<uint32_t>> PayloadMap;
            void packTF(const edm::Event&, const edm::EDGetTokenT<RegionalMuonCandBxCollection>&, Blocks&, const std::vector<unsigned int>&);
      };
   }
}

// Implementation
namespace l1t {
   namespace stage2 {
      Blocks
      RegionalMuonGMTPacker::pack(const edm::Event& event, const PackerTokens* toks)
      {	 

	 //auto bmtfToken = static_cast<const GMTTokens*>(toks)->getRegionalMuonCandTokenBMTF();
         //auto omtfToken = static_cast<const GMTTokens*>(toks)->getRegionalMuonCandTokenOMTF();
	 //auto emtfToken = static_cast<const GMTTokens*>(toks)->getRegionalMuonCandTokenEMTF();

         Blocks blocks;

         // link ids for the different TFs
         //std::vector<unsigned int> bmtfLinks {48,49,50,51,52,53,54,55,56,57,58,59};
         //std::vector<unsigned int> omtfLinks {42,43,44,45,46,47,60,61,62,63,64,65};
         //std::vector<unsigned int> emtfLinks {36,37,38,39,40,41,66,67,68,69,70,71};

         // pack the muons for each TF in blocks
         //packTF(event, bmtfToken, blocks, bmtfLinks);
         //packTF(event, omtfToken, blocks, omtfLinks);
         //packTF(event, emtfToken, blocks, emtfLinks);

         return blocks;
      }

      void
      RegionalMuonGMTPacker::packTF(const edm::Event& event, const edm::EDGetTokenT<RegionalMuonCandBxCollection>& tfToken, Blocks &blocks, const std::vector<unsigned int>& links)
      {
         edm::Handle<RegionalMuonCandBxCollection> muons;
         event.getByToken(tfToken, muons);
   
         PayloadMap payloadMap;

         unsigned bxCtr = 0;
         for (int i = muons->getFirstBX(); i <= muons->getLastBX(); ++i, ++bxCtr) {
            for (auto mu = muons->begin(i); mu != muons->end(i); ++mu) {
               uint32_t msw = 0;
               uint32_t lsw = 0;

               RegionalMuonRawDigiTranslator::generatePackedDataWords(*mu, lsw, msw);

               payloadMap[mu->link()*2].push_back(lsw);
               payloadMap[mu->link()*2].push_back(msw);
            }

            // muons are expected to come on a range of links depending on the the TF
            // but even if there was no muon coming from a processor the block should be generated
            // so add these links without muons to the map as well so that they will be filled with zeros
            for (const auto &link : links) {
               if (payloadMap.count(link*2) == 0) {
                  payloadMap[link*2].push_back(0);
               } else {
                  // if the key was already created in a previous BX then seed an entry for the padding if nothing was filled in this bx
                  if (payloadMap[link*2].size() == bxCtr * 6) {
                     payloadMap[link*2].push_back(0);
                  }
               }
            }

            // padding to 3 muons per block id (link) per BX
            for (auto &kv : payloadMap) {
               while (kv.second.size() % 6 != 0) {
                  kv.second.push_back(0);
               }
            }
         }

         // push everything in the blocks vector
         for (auto &kv : payloadMap) {
            blocks.push_back(Block(kv.first, kv.second));
         }
      }
   }
}

DEFINE_L1T_PACKER(l1t::stage2::RegionalMuonGMTPacker);

#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

#include "CaloTokens.h"

namespace l1t {
   namespace stage1 {
      class EtSumPacker : public Packer {
         public:
            virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
      };
   }
}

// Implementation

namespace l1t {
namespace stage1 {
   Blocks
   EtSumPacker::pack(const edm::Event& event, const PackerTokens* toks)
   {
      edm::Handle<EtSumBxCollection> etSums;
      event.getByToken(static_cast<const CaloTokens*>(toks)->getEtSumToken(), etSums);

        std::vector<uint32_t> load;

        for (int i = etSums->getFirstBX(); i <= etSums->getLastBX(); ++i) {
          int n = 0;
          
          uint16_t objectTotalEt=0;
          uint16_t objectTotalHt=0;
          uint16_t objectMissingEt=0;
          uint16_t objectMissingEtPhi=0;

          for (auto j = etSums->begin(i); j != etSums->end(i) && n < 4; ++j, ++n) {
 
            if (j->getType()==l1t::EtSum::kTotalEt){
               objectTotalEt=\
                            std::min(j->hwPt(), 0xFFF);
            }
          
            else if (j->getType()==l1t::EtSum::kTotalHt){
               objectTotalHt=\
                            std::min(j->hwPt(), 0xFFF);
            }
             
            else if (j->getType()==l1t::EtSum::kMissingEt){
               objectMissingEt=\
                            std::min(j->hwPt(), 0xFFF);
               objectMissingEtPhi=\
                            std::min(j->hwPhi(), 0x3F);               
            }
          }
          
          uint32_t word0=(objectTotalEt & 0xFFFF) | ((objectMissingEt & 0xFFFF) << 16);
          uint32_t word1=(objectTotalHt & 0xFFFF) | ((objectMissingEtPhi & 0xFFFF) << 16);
          
          word0 |= (1 << 31) | (1 << 15);
          word1 |= ((i == 0) << 31) | ((i == 0) << 15);

          load.push_back(word0);
          load.push_back(word1);
        }



      return {Block(6, load)};
   }
}
}

DEFINE_L1T_PACKER(l1t::stage1::EtSumPacker);

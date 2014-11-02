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
                            std::min(j->hwPt(), 0xFFF) |
                            (0 & 0x1) << 12 |      
                            (1 & 0x1) << 15;   
            }
          
            else if (j->getType()==l1t::EtSum::kTotalHt){
               objectTotalHt=\
                            std::min(j->hwPt(), 0xFFF) |
                            (0 & 0x1) << 12;
               if (i==0) objectTotalHt = objectTotalHt | ((1 & 0x1) << 15);
            }
             
            else if (j->getType()==l1t::EtSum::kMissingEt){
               objectMissingEt=\
                            std::min(j->hwPt(), 0xFFF) |
                            (0 & 0x1) << 12 |      
                            (1 & 0x1) << 15;
               objectMissingEtPhi=\
                            std::min(j->hwPhi(), 0x3F);
               if (i==0) objectMissingEtPhi = objectTotalHt | ((1 & 0x1) << 15);               
               
            }
          }
          
          uint32_t word0=(objectTotalEt & 0xFFFF) | ((objectTotalHt & 0xFFFF) << 16);
          uint32_t word1=(objectMissingEt & 0xFFFF) | ((objectMissingEtPhi & 0xFFFF) << 16);

          std::cout << word0 << std::endl;
          std::cout << word1 << std::endl;

          load.push_back(word0);
          load.push_back(word1);
        }



      return {Block(6, load)};
   }
}
}

DEFINE_L1T_PACKER(l1t::stage1::EtSumPacker);

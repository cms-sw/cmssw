#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

#include "CaloTokens.h"

namespace l1t {
  namespace stage1 {
    class CaloSpareHFPacker : public Packer {
      public:
        virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
    };
  }
}

// Implementation

namespace l1t {
  namespace stage1 {
    Blocks
      CaloSpareHFPacker::pack(const edm::Event& event, const PackerTokens* toks)
      {
        edm::Handle<CaloSpareBxCollection> calosparesHFBitCounts;
        event.getByToken(static_cast<const CaloTokens*>(toks)->getCaloSpareHFBitCountsToken(), calosparesHFBitCounts);
        
        edm::Handle<CaloSpareBxCollection> calosparesHFRingSums;
        event.getByToken(static_cast<const CaloTokens*>(toks)->getCaloSpareHFRingSumsToken(), calosparesHFRingSums);

        std::vector<uint32_t> load;

        for (int i = calosparesHFBitCounts->getFirstBX(); i <= calosparesHFBitCounts->getLastBX(); ++i) {
          int n = 0;

          int hfbitcount=0;
          int hfringsum=0;
                    
          for (auto j = calosparesHFBitCounts->begin(i); j != calosparesHFBitCounts->end(i) && n < 2; ++j, ++n) {
              hfbitcount=std::min(j->hwPt(),0xFFF);
          } 
                    
          n=0;

          for (auto j = calosparesHFRingSums->begin(i); j != calosparesHFRingSums->end(i) && n < 2; ++j, ++n) {
              hfringsum=std::min(j->hwPt(),0xFFF);
          } 
                 
          uint16_t object[2]={0,0};
          
          object[0]=hfbitcount|((hfringsum & 0x7) << 12);
          object[1]=((hfringsum>>3) & 0x1FF) |(0x1)<<10 | (0x1)<<12 | (0x1)<<14;               
                            
          uint32_t word0=(object[0] & 0xFFFF);
          uint32_t word1=(object[1] & 0xFFFF);

          word0 |= (1 << 15);
          word1 |= ((i == 0) << 15);

          load.push_back(word0);
          load.push_back(word1);
        }

        return {Block(97, load)};
      }
  }
}

DEFINE_L1T_PACKER(l1t::stage1::CaloSpareHFPacker);

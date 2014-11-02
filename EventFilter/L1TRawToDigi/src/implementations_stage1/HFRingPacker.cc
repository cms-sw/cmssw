#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

#include "CaloTokens.h"

namespace l1t {
  namespace stage1 {
    class HFRingPacker : public Packer {
      public:
        virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
    };
  }
}

// Implementation

namespace l1t {
  namespace stage1 {
    Blocks
      HFRingPacker::pack(const edm::Event& event, const PackerTokens* toks)
      {
        edm::Handle<EtSumBxCollection> etSums;
        event.getByToken(static_cast<const CaloTokens*>(toks)->getEtSumToken(), etSums);

        edm::Handle<CaloSpareBxCollection> calospares;
        event.getByToken(static_cast<const CaloTokens*>(toks)->getCaloSpareToken(), calospares);

        std::vector<uint32_t> load;

        for (int i = etSums->getFirstBX(); i <= etSums->getLastBX(); ++i) {
          int n = 0;

          int hfbitcount=0;
          int hfringsum=0;
          int htmissphi=0;
          int htmiss=0;
          
          for (auto j = etSums->begin(i); j != etSums->end(i) && n < 4; ++j, ++n) {
            if (j->getType()==l1t::EtSum::kMissingHt){
              htmiss=std::min(j->hwPt(),0x7F);
              htmissphi=std::min(j->hwPhi(),0x1F);
            }
          }
          
          n=0;
          
          for (auto m = calospares->begin(i); m != calospares->end(i) && n < 2; ++m, ++n) {
            if (m->getType()==l1t::CaloSpare::HFBitCount){
              hfbitcount=std::min(m->hwPt(),0xFFF);
            }
            
            else if (m->getType()==l1t::CaloSpare::HFRingSum){
              hfringsum=std::min(m->hwPt(),0xFFF);
            }
          }
          
          uint16_t object[4]={0,0,0,0};
          
          object[0]=\
                            hfbitcount|
                            ((hfringsum & 0x7) << 12) |
                            (0x1 << 15);
          object[1]=\
                            ((hfringsum>>3) & 0x1FF) |
                            (0x0)<<9 | (0x1)<<10 | (0x0)<<11 | (0x1)<<12 | (0x0)<<13 | (0x1)<<14;
                            if (i==0) object[1] = object[1] | ((0x1) << 15);
                            
          object[2]=\
                            htmissphi|
                            ((htmiss & 0x7F) << 5 ) |
                            (0x0 << 12) |
                            (0x0 << 13) |
                            (0x1 << 14) |
                            (0x1 << 15);
          object[3]=\
                             0x1 | (0x0 << 1) | (0x1 << 2) | (0x0 << 3) | (0x1 << 4) | (0x0 << 5) | (0x1 << 6) |
                            (0x0 << 7) | (0x1 << 8) | (0x0 << 9) | (0x1 << 10) | (0x0 << 11) | (0x1 << 12) |
                            (0x0 << 13) | (0x1 << 14);
                            if (i==0) object[3] = object[3] | ((0x1) << 15);
                            
          uint32_t word0=(object[0] & 0xFFFF) | ((object[2] & 0xFFFF) << 16);
          uint32_t word1=(object[1] & 0xFFFF) | ((object[3] & 0xFFFF) << 16);

          std::cout << word0 << std::endl;
          std::cout << word1 << std::endl;

          load.push_back(word0);
          load.push_back(word1);
        }

        return {Block(7, load)};
      }
  }
}

DEFINE_L1T_PACKER(l1t::stage1::HFRingPacker);

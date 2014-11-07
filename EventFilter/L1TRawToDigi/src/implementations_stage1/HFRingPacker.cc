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

        edm::Handle<CaloSpareBxCollection> calosparesHFBitCounts;
        event.getByToken(static_cast<const CaloTokens*>(toks)->getCaloSpareHFBitCountsToken(), calosparesHFBitCounts);
        
        edm::Handle<CaloSpareBxCollection> calosparesHFRingSums;
        event.getByToken(static_cast<const CaloTokens*>(toks)->getCaloSpareHFRingSumsToken(), calosparesHFRingSums);

        std::vector<uint32_t> load;

        for (int i = etSums->getFirstBX(); i <= etSums->getLastBX(); ++i) {
          int n = 0;

          int hfbitcount=0;
          int hfringsum=0;
          int htmissphi=0;
          int htmiss=0;
          
          int flaghtmiss=0;
          
          for (auto j = etSums->begin(i); j != etSums->end(i) && n < 4; ++j, ++n) {
            if (j->getType()==l1t::EtSum::kMissingHt){
            
              flaghtmiss=j->hwQual() & 0x1;
              htmiss=std::min(j->hwPt(),0x7F);              
              htmissphi=std::min(j->hwPhi(),0x1F);
            }
          }
          
          n=0;
          
          for (auto j = calosparesHFBitCounts->begin(i); j != calosparesHFBitCounts->end(i) && n < 2; ++j, ++n) {
              hfbitcount=std::min(j->hwPt(),0xFFF);
          } 
                    
          n=0;

          for (auto j = calosparesHFRingSums->begin(i); j != calosparesHFRingSums->end(i) && n < 2; ++j, ++n) {
              hfringsum=std::min(j->hwPt(),0xFFF);
          } 
                 
          uint16_t object[4]={0,0,0,0};
          
          object[0]=hfbitcount|((hfringsum & 0x7) << 12);
          object[1]=htmissphi|((htmiss & 0x7F) << 5 ) |(flaghtmiss<<12)|(0x1 << 14);
          object[2]=((hfringsum>>3) & 0x1FF) |(0x1)<<10 | (0x1)<<12 | (0x1)<<14;               
          object[3]= 0x1 | (0x1 << 2) | (0x1 << 4) | (0x1 << 6) |(0x1 << 8) | (0x1 << 10) | (0x1 << 12) | (0x1 << 14);
                            
          uint32_t word0=(object[0] & 0xFFFF) | ((object[1] & 0xFFFF) << 16);
          uint32_t word1=(object[2] & 0xFFFF) | ((object[3] & 0xFFFF) << 16);

          word0 |= (1 << 31) | (1 << 15);
          word1 |= ((i == 0) << 31) | ((i == 0) << 15);

          load.push_back(word0);
          load.push_back(word1);
        }

        return {Block(7, load)};
      }
  }
}

DEFINE_L1T_PACKER(l1t::stage1::HFRingPacker);

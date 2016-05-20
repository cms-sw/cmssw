#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

#include "CaloTokens.h"

#include "L1TStage2Layer2Constants.h"

namespace l1t {
    namespace stage2 {
      class EtSumPacker : public Packer {
         public:
	    EtSumPacker(int b1) : b1_(b1) {}
            virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
            int b1_;
      };
      class GTEtSumPacker : public EtSumPacker {
         public:
             GTEtSumPacker() : EtSumPacker(20) {}
      };
      class CaloEtSumPacker : public EtSumPacker {
         public:
	     CaloEtSumPacker() : EtSumPacker(21) {}
      };

   }
}

// Implementation

namespace l1t {
namespace stage2 {
   Blocks
   EtSumPacker::pack(const edm::Event& event, const PackerTokens* toks)
   {
      edm::Handle<EtSumBxCollection> etSums;
      event.getByToken(static_cast<const CommonTokens*>(toks)->getEtSumToken(), etSums);

      uint32_t et_word = 0;
      uint32_t ht_word = 0;
      uint32_t met_word = 0;
      uint32_t mht_word = 0;

      for (int i = etSums->getFirstBX(); i <= etSums->getLastBX(); ++i) {
         for (auto j = etSums->begin(i); j != etSums->end(i); ++j) {
	   uint32_t word = std::min(j->hwPt(), 0xFFF);
	   if ((j->getType()==l1t::EtSum::kMissingEt) || (j->getType()==l1t::EtSum::kMissingHt))
	     word = word | ((j->hwPhi() & 0xFF) << 12);
	   
	   if (j->getType()==l1t::EtSum::kTotalEt)   et_word = word;
	   if (j->getType()==l1t::EtSum::kTotalHt)   ht_word = word;
	   if (j->getType()==l1t::EtSum::kMissingEt) met_word = word;
	   if (j->getType()==l1t::EtSum::kMissingHt) mht_word = word;
         }
      }

      std::vector<uint32_t> load;
      load.push_back(et_word);
      load.push_back(ht_word);
      load.push_back(met_word);
      load.push_back(mht_word);
      while (load.size()<l1t::stage2::layer2::demux::nOutputFramePerBX) load.push_back(0);

      return {Block(b1_, load)};
   }
}
}

DEFINE_L1T_PACKER(l1t::stage2::CaloEtSumPacker);
DEFINE_L1T_PACKER(l1t::stage2::GTEtSumPacker);


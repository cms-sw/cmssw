#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

#include "CaloTokens.h"

namespace l1t {
   namespace stage2 {
      class EtSumPacker : public Packer {
         public:
            virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
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
      event.getByToken(static_cast<const CaloTokens*>(toks)->getEtSumToken(), etSums);

      std::vector<uint32_t> load;

      for (int i = etSums->getFirstBX(); i <= etSums->getLastBX(); ++i) {
         for (auto j = etSums->begin(i); j != etSums->end(i); ++j) {
	   uint32_t word = std::min(j->hwPt(), 0xFFF);
	   if ((j->getType()==l1t::EtSum::kMissingEt) || (j->getType()==l1t::EtSum::kMissingHt))
	     word = word | ((j->hwPhi() & 0xFF) << 12);
	   load.push_back(word);
         }
      }

      return {Block(3, load)};
   }
}
}

DEFINE_L1T_PACKER(l1t::stage2::EtSumPacker);

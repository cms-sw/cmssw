#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

#include "CaloTokens.h"

namespace l1t {
   namespace stage2 {
      class EGammaPacker : public Packer {
         public:
            virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
      };
   }
}

// Implementation

namespace l1t {
namespace stage2 {
   Blocks
   EGammaPacker::pack(const edm::Event& event, const PackerTokens* toks)
   {
      edm::Handle<EGammaBxCollection> egs;
      event.getByToken(static_cast<const CaloTokens*>(toks)->getEGammaToken(), egs);

      std::vector<uint32_t> load;

      for (int i = egs->getFirstBX(); i <= egs->getLastBX(); ++i) {
         int n = 0;
         for (auto j = egs->begin(i); j != egs->end(i) && n < 12; ++j, ++n) {
            uint32_t word = \
                            std::min(j->hwPt(), 0x1FF) |
                            (abs(j->hwEta()) & 0x7F) << 9 |
                            ((j->hwEta() < 0) & 0x1) << 16 |
                            (j->hwPhi() & 0xFF) << 17 |
                            (j->hwIso() & 0x1) << 25 |
                            (j->hwQual() & 0x7) << 26;
            load.push_back(word);
         }

         // pad for up to 12 egammas
         for (; n < 12; ++n)
            load.push_back(0);
      }

      return {Block(1, load)};
   }
}
}

DEFINE_L1T_PACKER(l1t::stage2::EGammaPacker);

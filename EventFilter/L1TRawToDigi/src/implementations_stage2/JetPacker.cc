#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

#include "CaloTokens.h"

namespace l1t {
   namespace stage2 {
      class JetPacker : public Packer {
         public:
            virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
      };
   }
}

// Implementation

namespace l1t {
namespace stage2 {
   Blocks
   JetPacker::pack(const edm::Event& event, const PackerTokens* toks)
   {
      edm::Handle<JetBxCollection> jets;
      event.getByToken(static_cast<const CaloTokens*>(toks)->getJetToken(), jets);

      std::vector<uint32_t> load;

      for (int i = jets->getFirstBX(); i <= jets->getLastBX(); ++i) {
         int n = 0;
         for (auto j = jets->begin(i); j != jets->end(i) && n < 12; ++j, ++n) {
            uint32_t word = \
                            std::min(j->hwPt(), 0x7FF) |
                            (abs(j->hwEta()) & 0x7F) << 11 |
                            ((j->hwEta() < 0) & 0x1) << 18 |
                            (j->hwPhi() & 0xFF) << 19 |
                            (j->hwQual() & 0x7) << 27;
            load.push_back(word);
         }

         for (; n < 12; ++n)
            load.push_back(0);
      }

      return {Block(5, load)};
   }
}
}

DEFINE_L1T_PACKER(l1t::stage2::JetPacker);

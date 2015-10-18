#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

#include "GTTokens.h"

namespace l1t {
   namespace stage2 {
      class MuonPacker : public Packer {
         public:
            virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
      };
   }
}

// Implementation

namespace l1t {
namespace stage2 {
   Blocks
   MuonPacker::pack(const edm::Event& event, const PackerTokens* toks)
   {
      edm::Handle<MuonBxCollection> mus;
      event.getByToken(static_cast<const GTTokens*>(toks)->getMuonToken(), mus);

      std::vector<uint32_t> load;

      for (int i = mus->getFirstBX(); i <= mus->getLastBX(); ++i) {
         int n = 0;
         for (auto j = mus->begin(i); j != mus->end(i) && n < 8; ++j, ++n) {
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
         for (; n < 8; ++n)
            load.push_back(0);
      }

      return {Block(1, load)};
   }
}
}

DEFINE_L1T_PACKER(l1t::stage2::MuonPacker);

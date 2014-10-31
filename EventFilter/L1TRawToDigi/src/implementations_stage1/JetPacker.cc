#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

#include "CaloTokens.h"

namespace l1t {
  namespace stage1 {
    class JetPacker : public Packer {
      public:
        virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
    };
  }
}

// Implementation

namespace l1t {
  namespace stage1 {
    Blocks
      JetPacker::pack(const edm::Event& event, const PackerTokens* toks)
      {
        edm::Handle<JetBxCollection> jets;
        event.getByToken(static_cast<const CaloTokens*>(toks)->getJetToken(), jets);

        std::vector<uint32_t> load;

        for (int i = jets->getFirstBX(); i <= jets->getLastBX(); ++i) {
          int n = 0;
          uint16_t jetbit[4];

          for (auto j = jets->begin(i); j != jets->end(i) && n < 4; ++j, ++n) {
            std::cout << j->hwPt() << " @ " << j->hwEta() << ", " << j->hwPhi() << std::endl;
            jetbit[n] = \
                            std::min(j->hwPt(), 0x3F) |
                            (abs(j->hwEta()) & 0x7) << 6 |
                            ((j->hwEta() < 0) & 0x1) << 9 |
                            (j->hwPhi() & 0x1F) << 10 |
                            (j->hwQual() & 0x1) << 15;
            std::cout << jetbit[n] << std::endl;
            
          }
          uint32_t word0=(jetbit[0] & 0xFFFF) | ((jetbit[1] & 0xFFFF) << 16);
          uint32_t word1=(jetbit[2] & 0xFFFF) | ((jetbit[3] & 0xFFFF) << 16);

          std::cout << word0 << std::endl;
          std::cout << word1 << std::endl;

          load.push_back(word0);
          load.push_back(word1);
        }

        return {Block(3, load)};
      }
  }
}

DEFINE_L1T_PACKER(l1t::stage1::JetPacker);

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

        std::vector<uint32_t> load;

        return {};
      }
  }
}

DEFINE_L1T_PACKER(l1t::stage1::HFRingPacker);

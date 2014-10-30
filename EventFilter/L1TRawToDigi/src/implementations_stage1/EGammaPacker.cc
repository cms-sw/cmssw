#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

#include "CaloTokens.h"

namespace l1t {
   namespace stage1 {
      class EGammaPacker : public Packer {
         public:
            virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
      };
   }
}

// Implementation

namespace l1t {
namespace stage1 {
   Blocks
   EGammaPacker::pack(const edm::Event& event, const PackerTokens* toks)
   {
      edm::Handle<EGammaBxCollection> egs;
      event.getByToken(static_cast<const CaloTokens*>(toks)->getEGammaToken(), egs);

      return {};
   }
}
}

DEFINE_L1T_PACKER(l1t::stage1::EGammaPacker);

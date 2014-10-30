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

      return {};
   }
}
}

DEFINE_L1T_PACKER(l1t::stage1::JetPacker);

#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "CaloTokens.h"

namespace l1t {
   namespace stage1 {
      class CaloTowerPacker : public Packer {
         public:
            virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
      };
   }
}

// Implementation

namespace l1t {
namespace stage1 {
   Blocks
   CaloTowerPacker::pack(const edm::Event& event, const PackerTokens* toks)
   {

      edm::Handle<CaloTowerBxCollection> towers;
      event.getByToken(static_cast<const CaloTokens*>(toks)->getCaloTowerToken(), towers);

      Blocks res;
      return res;
   }
}
}

DEFINE_L1T_PACKER(l1t::stage1::CaloTowerPacker);

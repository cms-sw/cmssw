#ifndef L1T_PACKER_STAGE2_INTERMEDIATEMUONPACKER_H
#define L1T_PACKER_STAGE2_INTERMEDIATEMUONPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

namespace l1t {
   namespace stage2 {
      class IntermediateMuonPacker : public Packer {
         public:
            virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
      };
   }
}

#endif

#ifndef L1T_PACKER_STAGE2_CALOTOWERPACKER_H
#define L1T_PACKER_STAGE2_CALOTOWERPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

namespace l1t {
   namespace stage2 {
      class CaloTowerPacker : public Packer {
         public:
            Blocks pack(const edm::Event&, const PackerTokens*) override;
      };
   }
}

#endif

#ifndef L1T_PACKER_STAGE1_ETSUMPACKER_H
#define L1T_PACKER_STAGE1_ETSUMPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

namespace l1t {
   namespace stage1 {
      class EtSumPacker : public Packer {
         public:
            Blocks pack(const edm::Event&, const PackerTokens*) override;
      };
   }
}

#endif

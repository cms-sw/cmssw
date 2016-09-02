#ifndef L1T_PACKER_STAGE1_CALOSPAREHFPACKER_H
#define L1T_PACKER_STAGE1_CALOSPAREHFPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

namespace l1t {
  namespace stage1 {
    class CaloSpareHFPacker : public Packer {
      public:
        virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
    };
  }
}

#endif

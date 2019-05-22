#ifndef L1T_PACKER_STAGE1_CALOSPAREHFPACKER_H
#define L1T_PACKER_STAGE1_CALOSPAREHFPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

namespace l1t {
  namespace stage1 {
    class CaloSpareHFPacker : public Packer {
    public:
      Blocks pack(const edm::Event&, const PackerTokens*) override;
    };
  }  // namespace stage1
}  // namespace l1t

#endif

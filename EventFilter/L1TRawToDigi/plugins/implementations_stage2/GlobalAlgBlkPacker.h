#ifndef L1T_PACKER_STAGE2_GLOBALALGBLKPACKER_H
#define L1T_PACKER_STAGE2_GLOBALALGBLKPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

namespace l1t {
  namespace stage2 {
    class GlobalAlgBlkPacker : public Packer {
    public:
      Blocks pack(const edm::Event&, const PackerTokens*) override;
    };
  }  // namespace stage2
}  // namespace l1t

#endif

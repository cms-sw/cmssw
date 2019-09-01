#ifndef L1T_PACKER_STAGE2_MPUNPACKER_0X10010033_H
#define L1T_PACKER_STAGE2_MPUNPACKER_0X10010033_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
  namespace stage2 {
    class MPUnpacker_0x10010033 : public Unpacker {
    public:
      bool unpack(const Block& block, UnpackerCollections* coll) override;
    };
  }  // namespace stage2
}  // namespace l1t

#endif

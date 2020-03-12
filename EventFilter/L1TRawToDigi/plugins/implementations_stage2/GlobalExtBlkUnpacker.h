#ifndef L1T_PACKER_STAGE2_GLOBALEXTBLKUNPACKER_H
#define L1T_PACKER_STAGE2_GLOBALEXTBLKUNPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
  namespace stage2 {
    class GlobalExtBlkUnpacker : public Unpacker {
    public:
      bool unpack(const Block& block, UnpackerCollections* coll) override;
    };
  }  // namespace stage2
}  // namespace l1t

#endif

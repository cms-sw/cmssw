#ifndef L1T_PACKER_STAGE1_HFRINGUNPACKER_H
#define L1T_PACKER_STAGE1_HFRINGUNPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
  namespace stage1 {
    class HFRingUnpacker : public Unpacker {
      public:
        bool unpack(const Block& block, UnpackerCollections *coll) override;
    };
  }
}

#endif

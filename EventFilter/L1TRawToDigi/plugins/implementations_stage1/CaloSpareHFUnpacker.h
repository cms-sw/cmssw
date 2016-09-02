#ifndef L1T_PACKER_STAGE1_CALOSPAREHFUNPACKER_H
#define L1T_PACKER_STAGE1_CALOSPAREHFUNPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
  namespace stage1 {
    class CaloSpareHFUnpacker : public Unpacker {
      public:
        virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
    };
  }
}

#endif

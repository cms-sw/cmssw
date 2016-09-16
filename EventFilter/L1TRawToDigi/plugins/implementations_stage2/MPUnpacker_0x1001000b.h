#ifndef L1T_PACKER_STAGE2_MPUNPACKER_0X1001000B_H
#define L1T_PACKER_STAGE2_MPUNPACKER_0X1001000B_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
  namespace stage2 {
    class MPUnpacker_0x1001000b : public Unpacker {
    public:
      enum { BLK_TOT_POS=123, BLK_X_POS=121, BLK_Y_POS=127, BLK_TOT_NEG=125, BLK_X_NEG=131, BLK_Y_NEG=129};
      virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
    private:
      int etaSign(int blkId);
    };
  }
}

#endif

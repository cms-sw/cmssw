#ifndef L1T_PACKER_STAGE2_MPUNPACKER_0X10010010_H
#define L1T_PACKER_STAGE2_MPUNPACKER_0X10010010_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
   namespace stage2 {
      class MPUnpacker_0x10010010 : public Unpacker {
         public:
            bool unpack(const Block& block, UnpackerCollections *coll) override;
      };
   }
}

#endif

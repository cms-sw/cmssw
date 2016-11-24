#ifndef L1T_PACKER_STAGE2_EGAMMAUNPACKER_H
#define L1T_PACKER_STAGE2_EGAMMAUNPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
   namespace stage2 {
      class EGammaUnpacker : public Unpacker {
         public:
            virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
      };
   }
}

#endif

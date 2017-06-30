#ifndef L1T_PACKER_STAGE2_ETSUMUNPACKER_H
#define L1T_PACKER_STAGE2_ETSUMUNPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
   namespace stage2 {
      class CaloLayer1Unpacker : public Unpacker {
         public:
            virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
      };
   }
}

#endif

#ifndef L1T_PACKER_STAGE1_LEGACYETSUMUNPACKER_H
#define L1T_PACKER_STAGE1_LEGACYETSUMUNPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
   namespace stage1 {
      namespace legacy {
         class EtSumUnpacker : public Unpacker {
            public:
               virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
         };
      }
   }
}

#endif

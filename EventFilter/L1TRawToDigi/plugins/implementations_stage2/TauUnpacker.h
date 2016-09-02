#ifndef L1T_PACKER_STAGE2_TAUUNPACKER_H
#define L1T_PACKER_STAGE2_TAUUNPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
   namespace stage2 {
      class TauUnpacker : public Unpacker {
         public:
            virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
      };
   }
}

#endif

#ifndef EventFilter_L1TRawToDigi_stage2_IntermediateMuonUnpacker_h
#define EventFilter_L1TRawToDigi_stage2_IntermediateMuonUnpacker_h

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
   namespace stage2 {
      class IntermediateMuonUnpacker : public Unpacker {
         public:
            virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
      };
   }
}

#endif

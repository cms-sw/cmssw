#ifndef EventFilter_L1TRawToDigi_stage2_IntermediateMuonPacker_h
#define EventFilter_L1TRawToDigi_stage2_IntermediateMuonPacker_h

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

namespace l1t {
   namespace stage2 {
      class IntermediateMuonPacker : public Packer {
         public:
            Blocks pack(const edm::Event&, const PackerTokens*) override;
      };
   }
}

#endif

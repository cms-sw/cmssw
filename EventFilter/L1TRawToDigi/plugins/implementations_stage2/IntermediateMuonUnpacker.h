#ifndef EventFilter_L1TRawToDigi_stage2_IntermediateMuonUnpacker_h
#define EventFilter_L1TRawToDigi_stage2_IntermediateMuonUnpacker_h

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"
#include <map>
#include "DataFormats/L1Trigger/interface/Muon.h"

namespace l1t {
   namespace stage2 {
      class IntermediateMuonUnpacker : public Unpacker {
         public:
            IntermediateMuonUnpacker();
            ~IntermediateMuonUnpacker() override {};

            bool unpack(const Block& block, UnpackerCollections *coll) override;

         private:
            static constexpr unsigned nWords_ = 6; // every link transmits 6 words (3 muons) per bx
            static constexpr unsigned bxzs_enable_shift_ = 1;

            MuonBxCollection* res1_;
            MuonBxCollection* res2_;
            unsigned int coll1Cnt_;

            void unpackBx(int bx, const std::vector<uint32_t>& payload, unsigned int startIdx=0);
      };
   }
}

#endif

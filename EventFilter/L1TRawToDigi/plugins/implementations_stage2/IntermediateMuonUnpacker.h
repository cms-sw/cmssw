#ifndef EventFilter_L1TRawToDigi_stage2_IntermediateMuonUnpacker_h
#define EventFilter_L1TRawToDigi_stage2_IntermediateMuonUnpacker_h

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
   namespace stage2 {
      class IntermediateMuonUnpacker : public Unpacker {
         public:
            IntermediateMuonUnpacker();
            ~IntermediateMuonUnpacker() {};

            virtual bool unpack(const Block& block, UnpackerCollections *coll) override;

            inline unsigned int getAlgoVersion() { return algoVersion_; };
            inline void setAlgoVersion(const unsigned int version) { algoVersion_ = version; };

         private:
            static const unsigned int nWords_ = 6; // every link transmits 6 words (3 muons) per bx
            static const unsigned int bxzs_enable_shift_ = 1;

            MuonBxCollection* res1_;
            MuonBxCollection* res2_;
            unsigned int algoVersion_;
            unsigned int coll1Cnt_;

            void unpackBx(int bx, const std::vector<uint32_t>& payload, unsigned int startIdx=0);
      };
   }
}

#endif

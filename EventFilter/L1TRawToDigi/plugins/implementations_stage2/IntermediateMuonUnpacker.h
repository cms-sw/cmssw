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
            unsigned int algoVersion_;
      };
   }
}

#endif

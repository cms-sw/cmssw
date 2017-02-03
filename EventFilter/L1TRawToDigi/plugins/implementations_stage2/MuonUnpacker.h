#ifndef L1T_PACKER_STAGE2_MUONUNPACKER_H
#define L1T_PACKER_STAGE2_MUONUNPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
   namespace stage2 {
      class MuonUnpacker : public Unpacker {
         public:
            MuonUnpacker();
            ~MuonUnpacker() {};

            virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
            unsigned int getAlgoVersion();
            void setAlgoVersion(unsigned int version);
         private:
            unsigned int algoVersion_;

      };
   }
}

#endif

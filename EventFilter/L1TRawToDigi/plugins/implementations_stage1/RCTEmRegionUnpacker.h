#ifndef L1T_PACKER_STAGE1_RCTUNPACKER_H
#define L1T_PACKER_STAGE1_RCTUNPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
  namespace stage1 {
    class RCTEmRegionUnpacker : public Unpacker {
      public:
        bool unpack(const Block& block, UnpackerCollections *coll) override;
      private:
        unsigned int counter_ = 0;
    };
  }
}

#endif

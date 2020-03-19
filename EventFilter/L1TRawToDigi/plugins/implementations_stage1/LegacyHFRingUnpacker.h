#ifndef L1T_PACKER_STAGE1_LEGACYHFRINGUNPACKER_H
#define L1T_PACKER_STAGE1_LEGACYHFRINGUNPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
  namespace stage1 {
    namespace legacy {
      class HFRingUnpacker : public Unpacker {
      public:
        bool unpack(const Block& block, UnpackerCollections* coll) override;
      };
    }  // namespace legacy
  }    // namespace stage1
}  // namespace l1t

#endif

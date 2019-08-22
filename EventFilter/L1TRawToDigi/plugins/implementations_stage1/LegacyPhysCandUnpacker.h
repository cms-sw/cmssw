#ifndef L1T_PACKER_STAGE1_LEGACYPHYSCANDUNPACKER_H
#define L1T_PACKER_STAGE1_LEGACYPHYSCANDUNPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
  namespace stage1 {
    namespace legacy {
      class IsoEGammaUnpacker : public Unpacker {
      public:
        bool unpack(const Block& block, UnpackerCollections* coll) override;
      };

      class NonIsoEGammaUnpacker : public Unpacker {
      public:
        bool unpack(const Block& block, UnpackerCollections* coll) override;
      };

      class CentralJetUnpacker : public Unpacker {
      public:
        bool unpack(const Block& block, UnpackerCollections* coll) override;
      };

      class ForwardJetUnpacker : public Unpacker {
      public:
        bool unpack(const Block& block, UnpackerCollections* coll) override;
      };

      class TauUnpacker : public Unpacker {
      public:
        bool unpack(const Block& block, UnpackerCollections* coll) override;
      };

      class IsoTauUnpacker : public Unpacker {
      public:
        bool unpack(const Block& block, UnpackerCollections* coll) override;
      };
    }  // namespace legacy
  }    // namespace stage1
}  // namespace l1t

#endif

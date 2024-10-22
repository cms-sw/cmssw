#ifndef L1T_PACKER_STAGE1_PHYSCANDUNPACKER_H
#define L1T_PACKER_STAGE1_PHYSCANDUNPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
  namespace stage1 {
    class IsoEGammaUnpackerLeft : public Unpacker {
    public:
      bool unpack(const Block& block, UnpackerCollections* coll) override;
    };

    class NonIsoEGammaUnpackerLeft : public Unpacker {
    public:
      bool unpack(const Block& block, UnpackerCollections* coll) override;
    };

    class CentralJetUnpackerLeft : public Unpacker {
    public:
      bool unpack(const Block& block, UnpackerCollections* coll) override;
    };

    class ForwardJetUnpackerLeft : public Unpacker {
    public:
      bool unpack(const Block& block, UnpackerCollections* coll) override;
    };

    class TauUnpackerLeft : public Unpacker {
    public:
      bool unpack(const Block& block, UnpackerCollections* coll) override;
    };

    class IsoTauUnpackerLeft : public Unpacker {
    public:
      bool unpack(const Block& block, UnpackerCollections* coll) override;
    };

    class IsoEGammaUnpackerRight : public Unpacker {
    public:
      bool unpack(const Block& block, UnpackerCollections* coll) override;
    };

    class NonIsoEGammaUnpackerRight : public Unpacker {
    public:
      bool unpack(const Block& block, UnpackerCollections* coll) override;
    };

    class CentralJetUnpackerRight : public Unpacker {
    public:
      bool unpack(const Block& block, UnpackerCollections* coll) override;
    };

    class ForwardJetUnpackerRight : public Unpacker {
    public:
      bool unpack(const Block& block, UnpackerCollections* coll) override;
    };

    class TauUnpackerRight : public Unpacker {
    public:
      bool unpack(const Block& block, UnpackerCollections* coll) override;
    };

    class IsoTauUnpackerRight : public Unpacker {
    public:
      bool unpack(const Block& block, UnpackerCollections* coll) override;
    };

  }  // namespace stage1
}  // namespace l1t

#endif

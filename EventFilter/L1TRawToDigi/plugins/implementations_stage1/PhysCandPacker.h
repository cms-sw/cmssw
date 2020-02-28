#ifndef L1T_PACKER_STAGE1_PHYSCANDPACKER_H
#define L1T_PACKER_STAGE1_PHYSCANDPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

namespace l1t {
  namespace stage1 {
    class IsoEGammaPacker : public Packer {
    public:
      Blocks pack(const edm::Event&, const PackerTokens*) override;
    };

    class NonIsoEGammaPacker : public Packer {
    public:
      Blocks pack(const edm::Event&, const PackerTokens*) override;
    };

    class CentralJetPacker : public Packer {
    public:
      Blocks pack(const edm::Event&, const PackerTokens*) override;
    };

    class ForwardJetPacker : public Packer {
    public:
      Blocks pack(const edm::Event&, const PackerTokens*) override;
    };

    class TauPacker : public Packer {
    public:
      Blocks pack(const edm::Event&, const PackerTokens*) override;
    };

    class IsoTauPacker : public Packer {
    public:
      Blocks pack(const edm::Event&, const PackerTokens*) override;
    };
  }  // namespace stage1
}  // namespace l1t

#endif

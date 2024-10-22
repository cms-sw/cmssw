#ifndef L1T_PACKER_STAGE2_EGAMMAPACKER_H
#define L1T_PACKER_STAGE2_EGAMMAPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

namespace l1t {
  namespace stage2 {
    class EGammaPacker : public Packer {
    public:
      EGammaPacker(int b1, int b2) : b1_(b1), b2_(b2) {}
      Blocks pack(const edm::Event&, const PackerTokens*) override;
      int b1_, b2_;
    };

    class GTEGammaPacker : public EGammaPacker {
    public:
      GTEGammaPacker() : EGammaPacker(8, 10) {}
    };
    class CaloEGammaPacker : public EGammaPacker {
    public:
      CaloEGammaPacker() : EGammaPacker(9, 11) {}
    };
  }  // namespace stage2
}  // namespace l1t

#endif

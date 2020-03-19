#ifndef L1T_PACKER_STAGE2_ETSUMPACKER_H
#define L1T_PACKER_STAGE2_ETSUMPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

namespace l1t {
  namespace stage2 {
    class EtSumPacker : public Packer {
    public:
      EtSumPacker(int b1) : b1_(b1) {}
      Blocks pack(const edm::Event&, const PackerTokens*) override;
      int b1_;
    };
    class GTEtSumPacker : public EtSumPacker {
    public:
      GTEtSumPacker() : EtSumPacker(20) {}
    };
    class CaloEtSumPacker : public EtSumPacker {
    public:
      CaloEtSumPacker() : EtSumPacker(21) {}
    };

  }  // namespace stage2
}  // namespace l1t

#endif

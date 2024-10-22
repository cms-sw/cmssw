#ifndef L1T_PACKER_STAGE2_ETSUMPACKER_H
#define L1T_PACKER_STAGE2_ETSUMPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

namespace l1t {
  namespace stage2 {
    class ZDCPacker : public Packer {
    public:
      ZDCPacker(int b1) : b1_(b1) {}
      Blocks pack(const edm::Event&, const PackerTokens*) override;
      int b1_;
    };
    class GTEtSumZDCPacker : public ZDCPacker {
    public:
      GTEtSumZDCPacker() : ZDCPacker(142) {}
    };
    // class CaloEtSumZDCPacker : public ZDCPacker {
    // public:
    //   CaloEtSumZDCPacker() : ZDCPacker(143) {}
    // };

  }  // namespace stage2
}  // namespace l1t

#endif

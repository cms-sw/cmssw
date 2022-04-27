#ifndef L1T_PACKER_STAGE2_ETSUMUNPACKER_0X10010057_H
#define L1T_PACKER_STAGE2_ETSUMUNPACKER_0X10010057_H

#include "EventFilter/L1TRawToDigi/plugins/implementations_stage2/EtSumUnpacker.h"

namespace l1t {
  namespace stage2 {
    class EtSumUnpacker_0x10010057 : public EtSumUnpacker {
    public:
      EtSumUnpacker_0x10010057();
      ~EtSumUnpacker_0x10010057() override{};

      bool unpack(const Block& block, UnpackerCollections* coll) override;

      inline void setEtSumCopy(const unsigned int copy) override { EtSumCopy_ = copy; };

    private:
      unsigned int EtSumCopy_;
    };
  }  // namespace stage2
}  // namespace l1t

#endif

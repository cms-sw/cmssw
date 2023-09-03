#ifndef L1T_PACKER_STAGE2_ZDCUnpacker_H
#define L1T_PACKER_STAGE2_ZDCUnpacker_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
  namespace stage2 {
    class ZDCUnpacker : public Unpacker {
    public:
      ZDCUnpacker();
      ~ZDCUnpacker() override{};

      bool unpack(const Block& block, UnpackerCollections* coll) override;

      // kept setEtSumCopy and EtSumCopy_ because zdc uses EtSum class
      virtual inline void setEtSumCopy(const unsigned int copy) { EtSumCopy_ = copy; };

    private:
      unsigned int EtSumCopy_;
    };
  }  // namespace stage2
}  // namespace l1t

#endif
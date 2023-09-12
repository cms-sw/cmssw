#ifndef L1T_PACKER_STAGE2_ZDCUnpacker_H
#define L1T_PACKER_STAGE2_ZDCUnpacker_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
  namespace stage2 {
    class ZDCUnpacker : public Unpacker {
    public:
      ZDCUnpacker();
      ~ZDCUnpacker() override = default;

      bool unpack(const Block& block, UnpackerCollections* coll) override;

      inline void setEtSumZDCCopy(const unsigned int copy) { EtSumZDCCopy_ = copy; };

    private:
      unsigned int EtSumZDCCopy_;
    };
  }  // namespace stage2
}  // namespace l1t

#endif

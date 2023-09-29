#ifndef L1T_PACKER_STAGE2_REGIONALMUONGMTUNPACKER_H
#define L1T_PACKER_STAGE2_REGIONALMUONGMTUNPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"
#include "EventFilter/L1TRawToDigi/interface/Block.h"
#include "GMTCollections.h"

namespace l1t {
  namespace stage2 {
    class RegionalMuonGMTUnpacker : public Unpacker {
    public:
      bool unpack(const Block& block, UnpackerCollections* coll) override;
      void setIsKbmtf() { isKbmtf_ = true; }
      void setUseEmtfDisplacementInfo() { useEmtfDisplacementInfo_ = true; }
      void setUseEmtfNominalTightShowers() { useEmtfNominalTightShowers_ = true; }
      void setUseEmtfLooseShowers() { useEmtfLooseShowers_ = true; }

    private:
      static constexpr unsigned nWords_ = 6;  // every link transmits 6 words (3 muons) per bx
      static constexpr unsigned bxzs_enable_shift_ = 1;

      bool isKbmtf_{false};
      bool useEmtfDisplacementInfo_{false};
      bool useEmtfNominalTightShowers_{false};
      bool useEmtfLooseShowers_{false};
    };
  }  // namespace stage2
}  // namespace l1t

#endif
